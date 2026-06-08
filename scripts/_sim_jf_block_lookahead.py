"""JacobiForcing-style BLOCK-MODE Jacobi simulator with lookahead.

Block mode (JF-reference style):
  - Maintain a "current_block" of K positions.
  - Each iter: forward [prompt | committed | current_block]; get target_argmax K.
  - Convergence test: if target_argmax == current_block exactly → commit the
    whole block + the bonus token (model's argmax at position L+K, computed
    via one extra cheap forward), start new block.
  - Otherwise: current_block ← target_argmax (Jacobi fixed-point step).

  TPF = (K * num_blocks_committed) / (total_iters across all blocks).

Lookahead variant:
  At each refinement iter, instead of just argmax-ing, we ALSO sample K_alt
  parallel alternative refinements from the same logits. For each candidate
  next_block (vanilla argmax + K_alt sampled), we run a SECOND batched forward
  to measure how many positions of cand match its own next-iter argmax. Pick
  the cand that's CLOSEST TO ITS OWN FIXPOINT (smallest distance from candidate
  to its argmax) — i.e. the cand that's most converged. Use that as next iter's
  block.

The intuition (per user): in block mode, choosing a better intermediate block
narrows the per-iter "confusion zone" and accelerates convergence to fixpoint
→ fewer iters to commit the K-block → higher TPF.

Two TPF metrics reported per prompt:
  - vanilla block-mode TPF (no resampling)
  - lookahead block-mode TPF (K_alt resampling + best-by-self-fixpoint-distance)

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 scripts/_sim_jf_block_lookahead.py \\
        --model PATH --prompts_jsonl ... --K_alt 8 --out_jsonl ...
"""
from __future__ import annotations

import argparse
import json
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompts_jsonl", required=True)
    p.add_argument("--out_jsonl", required=True)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--K_alt", type=int, default=8)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--max_new", type=int, default=256)
    p.add_argument("--max_iters_per_block", type=int, default=128)
    p.add_argument("--max_blocks", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def init_block_prompt_sample(committed: list[int], K: int, rng: random.Random) -> list[int]:
    if not committed:
        return [0] * K
    return [committed[rng.randrange(len(committed))] for _ in range(K)]


@torch.no_grad()
def forward_logits_K(model, committed: list[int], block: list[int]) -> torch.Tensor:
    """Return logits at K positions: each predicts position L+j given context."""
    device = model.device
    L = len(committed)
    K = len(block)
    inp = torch.tensor([committed + block], dtype=torch.long, device=device)
    return model(input_ids=inp).logits[0, L - 1 : L - 1 + K, :]


@torch.no_grad()
def block_lookahead_run(model, tok, prompt_ids: list[int], args,
                        gen: torch.Generator, rng: random.Random) -> dict:
    """Block-mode Jacobi with lookahead resampling per refinement iter."""
    device = model.device
    K = args.K
    eos_id = tok.eos_token_id
    committed = list(prompt_ids)
    prompt_len = len(prompt_ids)
    total_iters = 0
    total_new = 0
    blocks_done = 0
    iters_per_block: list[int] = []

    while total_new < args.max_new and blocks_done < args.max_blocks:
        block = init_block_prompt_sample(committed, K, rng)
        iters = 0
        converged = False
        while iters < args.max_iters_per_block:
            iters += 1
            total_iters += 1
            logits_K = forward_logits_K(model, committed, block)
            target = logits_K.argmax(dim=-1).cpu().tolist()
            if target == block:
                converged = True
                break

            # Vanilla candidate = target_argmax (the standard Jacobi step).
            vanilla_next = list(target)

            # K_alt parallel samples (NEW reuses logits — no extra forward)
            if args.temp > 0:
                probs = torch.softmax(logits_K.float() / args.temp, dim=-1)
            else:
                probs = torch.zeros_like(logits_K)
                probs[range(K), torch.argmax(logits_K, dim=-1)] = 1.0
            samples = torch.multinomial(probs, num_samples=args.K_alt,
                                        replacement=True, generator=gen)  # (K, K_alt)
            alts = [[int(samples[j, k].item()) for j in range(K)] for k in range(args.K_alt)]

            candidates = [vanilla_next] + alts

            # Batched forward to evaluate each candidate's own fixpoint-distance.
            L_now = len(committed)
            batch_ids = torch.tensor([committed + c for c in candidates],
                                     dtype=torch.long, device=device)
            batch_logits = model(input_ids=batch_ids).logits  # (B, L_now+K, V)
            cand_argmax = batch_logits[:, L_now - 1 : L_now - 1 + K, :].argmax(dim=-1).cpu().tolist()

            # Score: how many positions of cand match its own argmax under cand's context.
            # The cand whose argmax == itself is at the fixpoint. Higher match = closer.
            best_idx = 0
            best_match = 0
            for i, cand in enumerate(candidates):
                m = sum(1 for a, b in zip(cand, cand_argmax[i]) if a == b)
                if m > best_match:
                    best_match = m
                    best_idx = i
            block = list(candidates[best_idx])

        # End of block: commit K tokens + 1 bonus (model's argmax at L+K under final block).
        # If EOS appears INSIDE the converged block, truncate and stop.
        if eos_id in block:
            first_eos = block.index(eos_id)
            committed.extend(block[: first_eos + 1])
            total_new += first_eos + 1
            blocks_done += 1
            iters_per_block.append(iters)
            break
        # Compute bonus token at position L+K
        bonus_inp = torch.tensor([committed + block], dtype=torch.long, device=device)
        bonus_logit = model(input_ids=bonus_inp).logits[0, -1, :]
        bonus = int(bonus_logit.argmax().item())

        committed.extend(block)
        committed.append(bonus)
        total_new += K + 1
        blocks_done += 1
        iters_per_block.append(iters)

        if bonus == eos_id:
            break

    tpf = total_new / total_iters if total_iters else 0.0
    return {
        "tpf": tpf,
        "n_tokens": total_new,
        "n_iters": total_iters,
        "blocks": blocks_done,
        "iters_per_block": iters_per_block,
        "mean_iters_per_block": (sum(iters_per_block)/len(iters_per_block)) if iters_per_block else 0,
        "completion_preview": tok.decode(committed[prompt_len:][:200], skip_special_tokens=False),
    }


@torch.no_grad()
def block_vanilla_run(model, tok, prompt_ids: list[int], args, rng: random.Random) -> dict:
    """Block-mode Jacobi vanilla (just the deterministic fixed-point loop)."""
    device = model.device
    K = args.K
    eos_id = tok.eos_token_id
    committed = list(prompt_ids)
    prompt_len = len(prompt_ids)
    total_iters = 0
    total_new = 0
    blocks_done = 0
    iters_per_block: list[int] = []
    while total_new < args.max_new and blocks_done < args.max_blocks:
        block = init_block_prompt_sample(committed, K, rng)
        iters = 0
        while iters < args.max_iters_per_block:
            iters += 1
            total_iters += 1
            target = forward_logits_K(model, committed, block).argmax(dim=-1).cpu().tolist()
            if target == block:
                break
            block = target
        # Honor EOS inside block.
        if eos_id in block:
            first_eos = block.index(eos_id)
            committed.extend(block[: first_eos + 1])
            total_new += first_eos + 1
            blocks_done += 1
            iters_per_block.append(iters)
            break
        # bonus
        bonus_inp = torch.tensor([committed + block], dtype=torch.long, device=device)
        bonus = int(model(input_ids=bonus_inp).logits[0, -1].argmax().item())
        committed.extend(block)
        committed.append(bonus)
        total_new += K + 1
        blocks_done += 1
        iters_per_block.append(iters)
        if bonus == eos_id:
            break
    tpf = total_new / total_iters if total_iters else 0.0
    return {
        "tpf": tpf,
        "n_tokens": total_new,
        "n_iters": total_iters,
        "blocks": blocks_done,
        "iters_per_block": iters_per_block,
        "mean_iters_per_block": (sum(iters_per_block)/len(iters_per_block)) if iters_per_block else 0,
        "completion_preview": tok.decode(committed[prompt_len:][:200], skip_special_tokens=False),
    }


def main() -> None:
    args = parse_args()
    print(f"[jf-block] loading {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()

    prompts = [json.loads(l) for l in open(args.prompts_jsonl)]
    print(f"[jf-block] {len(prompts)} prompts, K_alt={args.K_alt} temp={args.temp}", flush=True)

    gen = torch.Generator(device="cuda:0").manual_seed(args.seed)
    rng_cpu = random.Random(args.seed)
    out_fp = open(args.out_jsonl, "w")
    rows = []
    for i, pdata in enumerate(prompts):
        chat = [{"role": "user", "content": pdata["input"]}]
        prompt_text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        prompt_ids = tok(prompt_text, return_tensors="pt").input_ids[0].tolist()
        v = block_vanilla_run(model, tok, prompt_ids, args, rng_cpu)
        l = block_lookahead_run(model, tok, prompt_ids, args, gen, rng_cpu)
        r = {"batch_idx": i,
             "tpf_vanilla": v["tpf"], "iters_v": v["n_iters"], "tok_v": v["n_tokens"],
             "mean_iters_per_block_v": v["mean_iters_per_block"], "blocks_v": v["blocks"],
             "tpf_lookahead": l["tpf"], "iters_l": l["n_iters"], "tok_l": l["n_tokens"],
             "mean_iters_per_block_l": l["mean_iters_per_block"], "blocks_l": l["blocks"],
             "iters_per_block_v": v["iters_per_block"], "iters_per_block_l": l["iters_per_block"]}
        rows.append(r)
        out_fp.write(json.dumps(r) + "\n")
        out_fp.flush()
        print(f"[jf-block] [{i+1}/{len(prompts)}] "
              f"vanilla TPF={v['tpf']:.3f} (iters/block={v['mean_iters_per_block']:.1f}) "
              f"lookahead TPF={l['tpf']:.3f} (iters/block={l['mean_iters_per_block']:.1f}) "
              f"Δ={l['tpf']-v['tpf']:+.3f}", flush=True)

    n = len(rows)
    mv = sum(r['tpf_vanilla'] for r in rows)/n
    ml = sum(r['tpf_lookahead'] for r in rows)/n
    print(f"\n[jf-block] MEAN vanilla={mv:.3f}  lookahead={ml:.3f}  Δ={ml-mv:+.3f}", flush=True)
    out_fp.close()


if __name__ == "__main__":
    main()
