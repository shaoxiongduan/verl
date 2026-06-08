"""Simulate 'perfect iter-2 every iter' to test the ceiling on Jacobi TPF.

The natural block-decode iter-2 setup is: [prompt | 1 committed token | K-1
shifted noise + 1 new noise]. The model commits ~5.8 tokens on average (math_k3
on DS, K=32).

If every iter looked LIKE iter-2 — i.e., [prompt | accepted_so_far_committed |
K fresh-random-noise tokens] — would we sustain that 5.8 acceptance per iter?

This sim runs that exact protocol. Modes:
  - fresh: every iter, draft = K uniform-random tokens (no shifted refresh)
  - shifted: standard block-decode, K shifted + (n_acc+1) refilled with random
"""
from __future__ import annotations
import argparse, json, random, torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompts_jsonl", required=True)
    p.add_argument("--out_jsonl", required=True)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--max_new", type=int, default=512)
    p.add_argument("--max_iters", type=int, default=512)
    p.add_argument("--mode", choices=["fresh", "shifted"], required=True)
    p.add_argument("--vocab_size", type=int, default=152064)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def simulate(model, tok, prompt_ids: list[int], args, rng: random.Random) -> dict:
    device = model.device
    K = args.K
    eos_id = tok.eos_token_id
    alt_eos = 151645
    committed = list(prompt_ids)
    prompt_len = len(committed)

    # Initial draft: K uniform random
    draft = [rng.randrange(args.vocab_size) for _ in range(K)]
    total_new = 0
    n_iters = 0
    n_acc_history: list[int] = []

    while total_new < args.max_new and n_iters < args.max_iters:
        L = len(committed)
        inp = torch.tensor([committed + draft], dtype=torch.long, device=device)
        out = model(input_ids=inp)
        logits = out.logits[0, L - 1 : L - 1 + K, :]
        targ = logits.argmax(dim=-1).cpu().tolist()

        n_acc = 0
        for j in range(K):
            if draft[j] == targ[j]:
                n_acc += 1
            else:
                break
        n_acc_history.append(n_acc)
        for j in range(n_acc):
            committed.append(int(draft[j]))
        # Bonus token at position n_acc (target prediction at first reject)
        if n_acc < K:
            committed.append(int(targ[n_acc]))
            total_new += n_acc + 1
        else:
            total_new += K
        n_iters += 1
        if committed[-1] in (eos_id, alt_eos):
            break

        # Build next draft
        if args.mode == "fresh":
            # Always pure uniform-random K tokens — every iter mimics iter-2 setup
            draft = [rng.randrange(args.vocab_size) for _ in range(K)]
        else:  # shifted (block-decode-style)
            shifted = list(targ[n_acc + 1:])  # K - n_acc - 1
            draft = list(shifted)
            while len(draft) < K:
                draft.append(rng.randrange(args.vocab_size))
            draft = draft[:K]

    return {
        "n_tokens": total_new,
        "n_iters": n_iters,
        "tpf": total_new / max(1, n_iters),
        "mean_n_acc": sum(n_acc_history) / max(1, len(n_acc_history)),
        "n_acc_history": n_acc_history,
    }


def main():
    args = parse_args()
    print(f"[sim] loading {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()

    prompts = [json.loads(l) for l in open(args.prompts_jsonl)]
    print(f"[sim] mode={args.mode} K={args.K} max_new={args.max_new} n={len(prompts)}", flush=True)
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
        print(f"[sim] [{i+1}/{len(prompts)}] mode={args.mode} K={args.K} ntok={r['n_tokens']:4d} iters={r['n_iters']:4d} TPF={r['tpf']:.3f} mean_n_acc={r['mean_n_acc']:.2f}", flush=True)
    mean_tpf = sum(r["tpf"] for r in rows) / len(rows)
    mean_nacc = sum(r["mean_n_acc"] for r in rows) / len(rows)
    print(f"\n[sim] MODE={args.mode} K={args.K}: mean TPF={mean_tpf:.3f}, mean_n_acc per iter={mean_nacc:.3f}", flush=True)


if __name__ == "__main__":
    main()
