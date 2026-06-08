"""JF single-forward block-decode reimplementation with per-internal-iter
newly-accepted logging. Faithful re-cast of
`modeling.cllm2_qwen2_modeling_kv_terminate_on_eos_improved.jacobi_forward_greedy`
without KV-cache micro-management — uses full forward each iter to keep the
loop short and inspectable.

Per block:
  out = block_tokens (length K). Initial = [anchor, noise, noise, ..., noise]
  iter = 0
  while total_accepted < K:
      forward(committed + out) -> logits at the last K positions
      greedy_tokens = argmax(logits[:K-1])    # predicts positions 1..K-1 of out
      mismatch[i]   = (greedy_tokens[i] != out[i+1])
      num_acc       = leading run of (mismatch == 0) + 1
      commit out[:num_acc] (the always-accepted anchor + matched prefix)
      if num_acc < K-total_acc:
          # rebuild the unaccepted tail using THIS iter's argmax tail
          next_anchor = greedy_tokens[num_acc-1] (= argmax at first mismatch)
          new_tail    = argmax(logits[num_acc : K-1])  # K - num_acc - 1 tokens
          out = [next_anchor, *new_tail]               # length K - total_acc - num_acc + 1
      else:
          # whole block accepted; bonus token = argmax(logits[-1]) starts next block
          ...
      iter += 1

For ce_noisy_decay_k3 training: noise_source=uniform → init each new block's
tail with K-1 uniform-random vocab tokens. The first position (anchor) is the
bonus token from the previous block (or argmax over prompt last logit for the
very first block).

Output: aggregate per-iter newly_accepted across all blocks across all prompts.
The headline answer is the iter-1 mean — does it match the user's "≈ 6" claim?

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 scripts/_diag_jf_blockdecode_per_iter.py \\
        --model PATH --prompts_jsonl PATH --K 32 --max_new 256 --n_prompts 16
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompts_jsonl", required=True)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--max_new", type=int, default=256)
    p.add_argument("--max_block_iters", type=int, default=64)
    p.add_argument("--vocab_size", type=int, default=152064)
    p.add_argument("--n_prompts", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--noise_source", choices=["uniform", "prompt_sample"],
                   default="uniform")
    p.add_argument("--out_jsonl", type=str, default="")
    return p.parse_args()


def _sample_noise(committed: list[int], n: int, vocab_size: int,
                   noise_source: str, rng: random.Random) -> list[int]:
    if noise_source == "uniform":
        return [rng.randrange(vocab_size) for _ in range(n)]
    if not committed:
        return [0] * n
    return [committed[rng.randrange(len(committed))] for _ in range(n)]


@torch.no_grad()
def _forward_last_K(model, committed_t: torch.Tensor, block_t: torch.Tensor) -> torch.Tensor:
    """Return logits over the last K positions = the block. Shape (K, V)."""
    inp = torch.cat([committed_t, block_t], dim=1)
    L = committed_t.shape[1]
    K = block_t.shape[1]
    logits = model(input_ids=inp).logits[0, L - 1 : L - 1 + K, :]
    return logits


@torch.no_grad()
def decode_one_prompt(model, tok, prompt_ids: list[int], args,
                       rng: random.Random,
                       per_iter_acc: dict[int, list[int]]) -> dict:
    device = model.device
    K = args.K
    eos_id = tok.eos_token_id
    alt_eos_id = 151645  # Qwen2.5-Instruct alternate eos

    committed = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    prompt_len = committed.shape[1]
    total_new = 0
    total_forwards = 0
    n_blocks = 0
    eos_hit = False

    # Anchor for the first block = bonus from prompt-prefill argmax. Do one
    # prefill forward to get it (this forward IS counted, as in JF reference).
    prefill_logits = model(input_ids=committed).logits[0, -1, :]
    anchor = int(prefill_logits.argmax().item())
    total_forwards += 1
    total_new += 1
    committed = torch.cat([committed, torch.tensor([[anchor]], dtype=torch.long, device=device)], dim=1)
    if anchor in (eos_id, alt_eos_id):
        return {"n_tokens": total_new, "n_forwards": total_forwards, "n_blocks": 0,
                "tpf": total_new / total_forwards,
                "completion_preview": tok.decode([anchor], skip_special_tokens=False)}

    while total_new < args.max_new and not eos_hit:
        # Build the K-position block: anchor at position 0, noise at 1..K-1.
        # Then committed at this point already includes the anchor; we pop it
        # back to fit JF's convention where the block "covers" the anchor as its
        # first position (KV is grown by the block-1 fresh positions).
        # SIMPLER: keep committed as committed-before-anchor, treat block_tokens
        # as [anchor, noise...] of length K, forward over committed + block_tokens.
        committed_no_anchor = committed[:, :-1]
        committed_list = committed[0].cpu().tolist()
        noise_tail = _sample_noise(committed_list, K - 1, args.vocab_size,
                                    args.noise_source, rng)
        out = torch.tensor([[anchor] + noise_tail], dtype=torch.long, device=device)

        total_acc_in_block = 1  # the anchor is "always accepted" (JF +1)
        block_committed_ids: list[int] = [anchor]
        block_iter = 0
        next_anchor_for_next_block: int | None = None

        while total_acc_in_block < K and block_iter < args.max_block_iters:
            logits_K = _forward_last_K(model, committed_no_anchor, out)  # (K, V)
            total_forwards += 1
            argmax_K = logits_K.argmax(dim=-1)  # (K,)

            # JF mismatch: greedy_tokens[i] = argmax(logits[i]) predicts out[i+1].
            greedy_tokens = argmax_K[:-1]  # (K-1,) — these predict positions 1..K-1
            mismatch = (greedy_tokens != out[0, 1:])
            # leading run of mismatch=False
            n_leading = int((mismatch.cumsum(0) == 0).sum().item())
            num_acc_raw = n_leading + 1  # always +1 for the anchor

            # Cap by remaining block slots.
            new_acc = min(num_acc_raw, K - total_acc_in_block + 1)
            # Note: total_acc_in_block already counts the anchor (+1) at iter 0,
            # so new_acc here counts ONLY newly-accepted-this-iter tokens
            # (including the anchor on the FIRST iter). Subtract +1 for anchor on
            # the very first iter so we report the # of NEW positions filled
            # this iter, not double-counting the anchor.
            if block_iter == 0:
                newly_added = num_acc_raw - 1  # subtract anchor
            else:
                newly_added = num_acc_raw - 1  # in subsequent iters out[0] is the
                                                # previous iter's next_anchor (also
                                                # "always accepted" but already
                                                # committed at the end of last iter)
            # Actually JF jacobi_forward_greedy treats every iter the same way:
            # num_accepted = leading_match + 1, and out[0] becomes part of
            # accepted_n_gram each iter. To stay consistent with the reference,
            # we just write num_acc_raw fresh slots this iter (clip at remaining).
            slots_remaining = K - total_acc_in_block
            newly_added = min(slots_remaining, num_acc_raw if block_iter == 0
                              else num_acc_raw)
            # Subtract the anchor-already-committed on iters > 0
            if block_iter > 0:
                newly_added = max(0, newly_added - 1)

            # Log per-iter newly_added (matches user's "per-iter TPF" framing).
            per_iter_acc[block_iter].append(newly_added)

            # Append newly_added tokens to the block commit.
            if block_iter == 0:
                # out[1 : num_acc_raw] are the matched draft positions (those we
                # KEEP from the draft tail because they match the verify argmax).
                add_ids = out[0, 1 : 1 + newly_added].cpu().tolist()
            else:
                add_ids = out[0, 1 : 1 + newly_added].cpu().tolist()
            block_committed_ids.extend(add_ids)
            total_acc_in_block += newly_added

            # Check EOS in accepted.
            if eos_id in add_ids or alt_eos_id in add_ids:
                eos_hit = True
                break
            if total_acc_in_block >= K:
                # block fully filled; the bonus for the NEXT block = argmax at
                # the last logit position (one beyond block end).
                next_anchor_for_next_block = int(argmax_K[-1].item())
                break

            # has_rejected: True in normal case. Build new `out`:
            # next_anchor = greedy_tokens[num_acc_raw - 1] = argmax at first mismatch
            next_anchor_id = int(greedy_tokens[num_acc_raw - 1].item())
            # New tail = argmax over the unaccepted positions (rebuild from this
            # iter's predictions). Length = K - 1 - num_acc_raw.
            tail = argmax_K[num_acc_raw : K - 1].cpu().tolist()
            new_out_list = [next_anchor_id] + tail
            # Pad with fresh noise if still short.
            while len(new_out_list) < K - total_acc_in_block:
                pass  # never expected
            # OUT shrinks by num_acc_raw each iter, but we keep a constant K-length
            # view by padding new noise at the END to fill back to K total slots.
            short_by = K - len(new_out_list)
            if short_by > 0:
                new_out_list = new_out_list + _sample_noise(
                    committed_list, short_by, args.vocab_size, args.noise_source, rng)
            new_out_list = new_out_list[:K]
            out = torch.tensor([new_out_list], dtype=torch.long, device=device)

            block_iter += 1

        # Commit the block to `committed`.
        if eos_hit:
            # block_committed_ids may include EOS; truncate after first occurrence.
            cut = next(
                (i + 1 for i, t in enumerate(block_committed_ids)
                 if t in (eos_id, alt_eos_id)),
                len(block_committed_ids))
            block_committed_ids = block_committed_ids[:cut]
        # Pop the anchor (it's already in committed from previous step).
        new_tokens_to_add = block_committed_ids[1:]
        if new_tokens_to_add:
            committed = torch.cat(
                [committed, torch.tensor([new_tokens_to_add], dtype=torch.long, device=device)],
                dim=1)
            total_new += len(new_tokens_to_add)
        n_blocks += 1

        if eos_hit:
            break
        if next_anchor_for_next_block is None:
            # block exited without full fill — shouldn't happen in well-formed runs
            break
        anchor = next_anchor_for_next_block
        committed = torch.cat(
            [committed, torch.tensor([[anchor]], dtype=torch.long, device=device)],
            dim=1)
        total_new += 1
        if anchor in (eos_id, alt_eos_id):
            break

    tpf = (total_new / total_forwards) if total_forwards else 0.0
    return {
        "n_tokens": total_new,
        "n_forwards": total_forwards,
        "n_blocks": n_blocks,
        "tpf": tpf,
        "completion_preview": tok.decode(
            committed[0, prompt_len:][:200].cpu().tolist(),
            skip_special_tokens=False),
    }


def main() -> None:
    args = parse_args()
    print(f"[jf-block-diag] loading {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()

    prompts = [json.loads(l) for l in open(args.prompts_jsonl) if l.strip()]
    prompts = prompts[: args.n_prompts]
    print(f"[jf-block-diag] {len(prompts)} prompts, K={args.K} max_new={args.max_new} "
          f"noise={args.noise_source}", flush=True)

    rng = random.Random(args.seed)
    per_iter_acc: dict[int, list[int]] = defaultdict(list)
    out_fp = open(args.out_jsonl, "w") if args.out_jsonl else None

    rows = []
    for i, pdata in enumerate(prompts):
        chat = [{"role": "user", "content": pdata["input"]}]
        prompt_text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        prompt_ids = tok(prompt_text, return_tensors="pt").input_ids[0].tolist()
        r = decode_one_prompt(model, tok, prompt_ids, args, rng, per_iter_acc)
        rows.append(r)
        if out_fp:
            out_fp.write(json.dumps({**r, "batch_idx": i}) + "\n")
            out_fp.flush()
        print(f"[jf-block-diag] [{i+1}/{len(prompts)}] "
              f"n_tok={r['n_tokens']:4d} forwards={r['n_forwards']:4d} "
              f"blocks={r['n_blocks']:3d} TPF={r['tpf']:.3f}",
              flush=True)

    if out_fp:
        out_fp.close()

    # Aggregate per-iter stats
    print("\n=== Per-iter newly-accepted counts ===")
    print(f"{'iter':>5s} {'n_obs':>6s} {'mean':>9s} {'std':>7s} {'min':>5s} {'max':>5s}")
    for it in sorted(per_iter_acc.keys()):
        vals = per_iter_acc[it]
        n = len(vals)
        m = sum(vals) / n
        s = (sum((x - m) ** 2 for x in vals) / max(1, n - 1)) ** 0.5
        print(f"{it:5d} {n:6d} {m:9.3f} {s:7.3f} {min(vals):5d} {max(vals):5d}")
    print(f"\n[jf-block-diag] iter-1 mean = "
          f"{sum(per_iter_acc.get(1, []))/max(1,len(per_iter_acc.get(1, []))):.3f}  "
          f"(user expected ≈ 6)")


if __name__ == "__main__":
    main()
