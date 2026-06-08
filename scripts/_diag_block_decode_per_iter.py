"""Block-decode Jacobi simulator with PER-INTERNAL-ITER trace.

Replicates JF reference / cllm2_qwen2 `jacobi_forward_greedy`'s inner loop:
within a single block, init draft = K pure-noise tokens, then iterate Jacobi
forwards (each iter: forward → argmax → accept matching prefix → rebuild
unaccepted tail from this iter's argmax). Stop when total_accepted == K or
EOS appears in accepted prefix.

User's claim under test (2026-06-07): for the ce_noisy_decay_k3 model,
average per-iter committed tokens at iter-index 1 should be ≈ 6.

For each block we log: per-iter "newly_accepted" counts (the # tokens
committed in just that iter), the iter_count to converge, and the final
committed K-block.

Aggregates iter-0 mean, iter-1 mean, iter-2 mean across all blocks across
all prompts.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 scripts/_diag_block_decode_per_iter.py \\
        --model PATH --prompts_jsonl PATH --K 32 --max_new 256 --n_prompts 8
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
    p.add_argument("--K", type=int, default=32, help="block size = n_token_seq_len")
    p.add_argument("--max_new", type=int, default=256)
    p.add_argument("--max_block_iters", type=int, default=64,
                   help="hard cap on Jacobi iters within a single block")
    p.add_argument("--vocab_size", type=int, default=152064)
    p.add_argument("--n_prompts", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _argmax_K_at_tail(model, prefix_ids: torch.Tensor, draft_K: torch.Tensor) -> torch.Tensor:
    """Forward on (prefix + draft). Return argmax at the last K positions.

    prefix_ids: (1, L)   — already-committed tokens
    draft_K:    (1, K)   — draft (will be replaced each iter)

    Returns (1, K) argmax. argmax[j] predicts token at position L+j.
    """
    inp = torch.cat([prefix_ids, draft_K], dim=1)
    L = prefix_ids.shape[1]
    K = draft_K.shape[1]
    with torch.no_grad():
        logits = model(input_ids=inp).logits[0, L - 1 : L - 1 + K, :]
    return logits.argmax(dim=-1).unsqueeze(0)  # (1, K)


@torch.no_grad()
def decode_one_prompt(model, tok, prompt_ids: list[int], args, rng: random.Random,
                       per_iter_acc: dict[int, list[int]]) -> dict:
    device = model.device
    K = args.K
    eos_id = tok.eos_token_id

    committed = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    prompt_len = committed.shape[1]
    total_new = 0
    total_iters = 0
    block_idx = 0

    while total_new < args.max_new:
        # Init draft = K pure-noise tokens (matches CONSISTENCY_NOISE_SOURCE=uniform).
        draft = torch.tensor(
            [[rng.randrange(args.vocab_size) for _ in range(K)]],
            dtype=torch.long, device=device,
        )
        block_total_acc = 0
        block_iter = 0
        block_committed_tokens: list[int] = []

        while block_total_acc < K and block_iter < args.max_block_iters:
            # FORWARD: predict argmax at K positions (one per draft slot).
            argmax_K = _argmax_K_at_tail(model, committed, draft)[0]  # (K,)
            draft_row = draft[0]  # (K,)

            # JF-style verification: at position 0 we always accept (bonus). For
            # positions j in [1..K-1], check whether argmax at the PREVIOUS
            # position (= prediction of token at j) matches draft[j].
            # Equivalently: greedy_tokens[j-1] = argmax[j-1], compared to
            # draft[j]. Build mismatch over j=1..K-1.
            mismatch = (argmax_K[:-1] != draft_row[1:])  # (K-1,)
            # accepted_prefix = leading run of mismatch=False, then +1 for the
            # always-accepted first position.
            non_match = mismatch.cumsum(dim=-1) != 0  # (K-1,)
            n_leading_match = (~non_match).sum().item()
            num_accepted = n_leading_match + 1  # includes position 0

            # Commit the newly-accepted prefix to committed_block tokens. The
            # first position takes draft[0] which is meaningless for the very
            # first iter (it's noise); but JF reference handles this via
            # `first_correct_token` mechanism. To stay faithful we record
            # block_committed_tokens up to block_total_acc + num_accepted (capped
            # at K), using a mix of draft[0] (or this iter's argmax) for the
            # first slot. For SIMULATION SIMPLICITY: write argmax_K[:num_accepted]
            # which is the model's own prediction at the accepted positions
            # (since argmax at position j predicts token at j, and we already
            # know argmax[j-1] = draft[j] for accepted j).
            # NOTE: this assumes the noise at position 0 of the first iter is
            # over-written by the model's argmax — which is what JF reference
            # does via `next_token = argmax(logits[:,num_accepted_raw-1,:])`.
            new_acc = min(num_accepted, K - block_total_acc)
            # Write commit positions
            for j in range(new_acc):
                pos = block_total_acc + j
                if pos < K:
                    block_committed_tokens.append(int(argmax_K[pos].item())
                                                   if pos > 0
                                                   else int(argmax_K[0].item()))

            # Log per-iter newly-accepted count BEFORE updating draft.
            per_iter_acc[block_iter].append(new_acc)

            block_total_acc += new_acc
            block_iter += 1
            total_iters += 1

            if block_total_acc >= K:
                break

            # Build next iter's draft. JF reference: positions [0..K-1] for next
            # iter = [next_token, argmax[num_accepted..K-1], ... refill if needed].
            # Equivalent shorter form: just use argmax_K as new draft (shifted so
            # position 0 = argmax_K[block_total_acc-1] = the latest "next_token",
            # and the tail = noise refill).
            # SIMPLIFIED: rebuild ENTIRE draft from this iter's argmax, plus
            # fresh noise refill where the accepted prefix was eaten — but
            # since we still have K slots in the block, we just feed
            # argmax_K[num_accepted-1:K] padded with fresh noise.
            # Most faithful single-block emulation: keep the FIRST K - new_acc
            # remaining positions = argmax tail, refill rest with noise.
            n_remaining = K - block_total_acc
            tail = argmax_K[new_acc : new_acc + n_remaining]
            refill = torch.tensor(
                [rng.randrange(args.vocab_size) for _ in range(K - n_remaining)],
                dtype=torch.long, device=device,
            )
            new_draft = torch.cat([tail, refill], dim=0).unsqueeze(0)  # (1, K)
            draft = new_draft

        # Commit the block to `committed`.
        # block_committed_tokens may be shorter than K if we hit max_block_iters
        # without convergence. Pad with the iter's last argmax if needed.
        if len(block_committed_tokens) < K:
            # Pad with whatever we have in argmax_K from the last iter.
            pad_n = K - len(block_committed_tokens)
            block_committed_tokens.extend(int(t.item()) for t in argmax_K[-pad_n:])
        block_committed_tokens = block_committed_tokens[:K]

        # Stop the block at first EOS.
        if eos_id is not None and eos_id in block_committed_tokens:
            cut = block_committed_tokens.index(eos_id) + 1
            block_committed_tokens = block_committed_tokens[:cut]

        commit_t = torch.tensor([block_committed_tokens], dtype=torch.long, device=device)
        committed = torch.cat([committed, commit_t], dim=1)
        total_new += len(block_committed_tokens)
        block_idx += 1

        if eos_id is not None and eos_id in block_committed_tokens:
            break

    overall_tpf = (total_new / total_iters) if total_iters else 0.0
    return {
        "n_tokens": total_new,
        "n_iters": total_iters,
        "n_blocks": block_idx,
        "tpf_overall": overall_tpf,
        "completion_preview": tok.decode(
            committed[0, prompt_len:][:200].cpu().tolist(),
            skip_special_tokens=False),
    }


def main() -> None:
    args = parse_args()
    print(f"[block-diag] loading {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()

    prompts = [json.loads(l) for l in open(args.prompts_jsonl) if l.strip()]
    prompts = prompts[: args.n_prompts]
    rng = random.Random(args.seed)
    print(f"[block-diag] {len(prompts)} prompts, K={args.K} max_new={args.max_new}", flush=True)

    # per_iter_acc[iter_idx] = list of newly-accepted counts at that iter
    # (across all blocks across all prompts).
    per_iter_acc: dict[int, list[int]] = defaultdict(list)
    summaries = []

    for i, pdata in enumerate(prompts):
        chat = [{"role": "user", "content": pdata["input"]}]
        prompt_text = tok.apply_chat_template(chat, tokenize=False,
                                              add_generation_prompt=True)
        prompt_ids = tok(prompt_text, return_tensors="pt").input_ids[0].tolist()

        r = decode_one_prompt(model, tok, prompt_ids, args, rng, per_iter_acc)
        summaries.append(r)
        print(f"[block-diag] [{i+1}/{len(prompts)}] "
              f"n_tok={r['n_tokens']:4d} iters={r['n_iters']:4d} "
              f"blocks={r['n_blocks']:3d} overall_TPF={r['tpf_overall']:.3f}",
              flush=True)

    print("\n=== Per-iter newly-accepted counts (block-decode internal iters) ===")
    print(f"{'iter':>5s} {'n_obs':>6s} {'mean_new_acc':>13s} {'std':>7s}")
    for it in sorted(per_iter_acc.keys()):
        vals = per_iter_acc[it]
        n = len(vals)
        m = sum(vals) / n
        s = (sum((x - m) ** 2 for x in vals) / max(1, n - 1)) ** 0.5
        print(f"{it:5d} {n:6d} {m:13.3f} {s:7.3f}")
    print(f"\n[block-diag] If iter=1 mean is ~6, user's claim is verified — the")
    print(f"[block-diag] model DOES predict ~6 clean tokens given its OWN iter-0")
    print(f"[block-diag] argmax (computed on pure noise) as the next draft.")


if __name__ == "__main__":
    main()
