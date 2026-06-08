"""Multi-iteration Jacobi spec-decode simulator with controllable draft sharpness.

Hypothesis under test (2026-06-06): the bottleneck isn't iter N's n_acc (which
is fixed by causal attention given iter N's draft) — it's iter N+1's draft
QUALITY. Iter N+1's draft is the shifted tail of iter N's target_argmax. If
iter N's draft tail past rel=M is "confusing wrong" cascade content, the model's
predictions at those positions are polluted by that bad context, and iter N+1
inherits noisy draft tokens. If we instead REPLACE rel=M..K with clean uniform
noise — which the model can easily ignore — iter N's tail predictions become
sharper (closer to its AR continuation), iter N+1's draft has a NARROWER
confusion zone, and iter N+1 accepts more tokens.

Two simulation modes:
  natural: next draft = shifted target_argmax[n_acc+1:] + prompt_sample refill
  sharp-M: feed THIS iter a draft of [committed tail | M model preds | uniform
           noise]. Iter N's n_acc is unchanged (causal). The benefit lands at
           iter N+1, whose draft inherits iter N's (cleaner) tail predictions.

If TPF(sharp-M) > TPF(natural) for some M < TPF(natural), the hypothesis holds:
narrower confusion zones during draft-refresh compound into faster Jacobi
convergence.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 scripts/_sim_jacobi_sharpen.py \\
        --model PATH --prompts_jsonl ... --mode {natural,sharp1,sharp2,sharp3} \\
        --out_jsonl ...
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
    p.add_argument("--max_new", type=int, default=1024)
    p.add_argument("--max_iters", type=int, default=1024)
    p.add_argument("--mode", required=True,
                   choices=["natural", "sharp1", "sharp2", "sharp3", "sharp5", "sharp8"])
    p.add_argument("--vocab_size", type=int, default=152064)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def init_draft_prompt_sample(committed: list[int], K: int, rng: random.Random) -> list[int]:
    if not committed:
        return [0] * K
    return [committed[rng.randrange(len(committed))] for _ in range(K)]


def simulate(model, tok, prompt_ids: list[int], args, rng: random.Random) -> dict:
    device = model.device
    K = args.K
    sharpen_M: int | None
    if args.mode == "natural":
        sharpen_M = None
    elif args.mode == "sharp1":
        sharpen_M = 1
    elif args.mode == "sharp2":
        sharpen_M = 2
    elif args.mode == "sharp3":
        sharpen_M = 3
    elif args.mode == "sharp5":
        sharpen_M = 5
    elif args.mode == "sharp8":
        sharpen_M = 8
    else:
        raise ValueError(args.mode)

    committed = list(prompt_ids)
    prompt_len = len(committed)
    draft = init_draft_prompt_sample(committed, K, rng)

    total_new = 0
    n_iters = 0
    n_acc_history: list[int] = []
    eos_id = tok.eos_token_id

    while total_new < args.max_new and n_iters < args.max_iters:
        input_ids = torch.tensor([committed + draft], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(input_ids=input_ids)
        L = len(committed)
        # Logit at position L-1+j predicts the token at L+j (j=0..K-1).
        logits_K = out.logits[0, L - 1 : L - 1 + K, :]
        target_argmax = logits_K.argmax(dim=-1).cpu().tolist()

        n_acc = 0
        for j in range(K):
            if draft[j] == target_argmax[j]:
                n_acc += 1
            else:
                break
        n_acc_history.append(n_acc)

        # Commit n_acc accepted draft tokens + 1 bonus (target at first rejection).
        for j in range(n_acc):
            committed.append(int(draft[j]))
        if n_acc < K:
            bonus = int(target_argmax[n_acc])
            committed.append(bonus)
            total_new += n_acc + 1
        else:
            total_new += K
        n_iters += 1

        if committed[-1] == eos_id:
            break

        # Build next draft. natural and sharp-M differ in HOW the draft we'll
        # feed into iter N+1 is constructed. Both start from the same source:
        # target_argmax[n_acc+1:] is iter N's prediction at positions whose
        # context included the rejected draft.
        shifted_tail = list(target_argmax[n_acc + 1:])  # length K - n_acc - 1
        if sharpen_M is None:
            # Natural: refill the (n_acc + 1) freed slots at the end with
            # prompt_sample (matches JF-reference draft refresh).
            new_draft = list(shifted_tail)
            while len(new_draft) < K:
                new_draft.append(int(committed[rng.randrange(len(committed))]))
        else:
            # Sharpened: only keep the first M of shifted_tail as "natural"
            # draft content; the remaining K - M positions become uniform random
            # noise. Iter N+1 will then forward this draft, and its predictions
            # at rel > M will be informed by clean noise context — which the
            # model handles much better than mid-confidence wrong tokens.
            keep = shifted_tail[: max(0, sharpen_M)]
            new_draft = list(keep)
            while len(new_draft) < K:
                new_draft.append(rng.randrange(args.vocab_size))
        draft = new_draft[:K]

    tpf = total_new / n_iters if n_iters > 0 else 0.0
    return {
        "n_tokens": total_new,
        "n_iters": n_iters,
        "tpf": tpf,
        "n_acc_history": n_acc_history,
        "mean_n_acc": (sum(n_acc_history) / len(n_acc_history)) if n_acc_history else 0,
        "completion_preview": tok.decode(committed[prompt_len:][:200], skip_special_tokens=False),
    }


def main() -> None:
    args = parse_args()
    print(f"[sim] loading {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0"
    )
    model.eval()

    prompts = [json.loads(l) for l in open(args.prompts_jsonl)]
    print(f"[sim] {len(prompts)} prompts, mode={args.mode}", flush=True)

    rng = random.Random(args.seed)
    out_fp = open(args.out_jsonl, "w")
    rows = []
    for i, pdata in enumerate(prompts):
        chat = [{"role": "user", "content": pdata["input"]}]
        prompt_text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        prompt_ids = tok(prompt_text, return_tensors="pt").input_ids[0].tolist()
        r = simulate(model, tok, prompt_ids, args, rng)
        r["batch_idx"] = i
        rows.append(r)
        out_fp.write(json.dumps(r) + "\n")
        out_fp.flush()
        print(f"[sim] [{i+1}/{len(prompts)}] tok={r['n_tokens']:4d} iters={r['n_iters']:4d} TPF={r['tpf']:.3f}", flush=True)

    tpfs = [r["tpf"] for r in rows]
    mean_tpf = sum(tpfs) / len(tpfs)
    out_fp.close()
    print(f"\n[sim] MODE={args.mode}: mean TPF = {mean_tpf:.3f} (n={len(tpfs)})", flush=True)


if __name__ == "__main__":
    main()
