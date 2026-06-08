"""Warm-restart with imprecise boundary predictor.

Setup: a lightweight predictor identifies WHERE the clean prefix ends within
the K-block. The predictor outputs an "M" — number of leading warm positions
to trust. Beyond M, we don't trust the warm draft (predictor says "noise").

Each cycle:
  1. Warm forward: input `[committed | K random]` → warm_argmax (K model preds)
  2. Build verify draft:
       - First M positions = warm_argmax[0:M]   (the trusted-by-predictor part)
       - Last (K-M) positions = fresh uniform random noise
     This simulates: "predictor says first M tokens are clean candidates;
     beyond M is noise that needs another verify pass later"
  3. Verify forward: input `[committed | mixed_draft]` → verify_argmax
  4. Accept: standard greedy spec-decode prefix match between mixed_draft and
     verify_argmax. n_acc ≤ M because positions ≥ M are random noise that
     won't match.
  5. Commit n_acc + 1 tokens. Next cycle starts fresh.

If predictor is PERFECT (M = K = 32): equivalent to warm_restart → ~5.01 n_acc
If predictor is OFF (small M): n_acc capped at M, TPF ≤ M + 1
If predictor is "exactly right" (M ≈ 5 for math_k3): n_acc ≈ 5, TPF ≈ 6
If predictor over-shoots (M > 5): TPF still ~5 (model's denoising capacity is fixed)
If predictor under-shoots (M < 5): TPF drops to M+1 (we artificially throw away accept positions)

This tells us how MUCH PRECISION the boundary predictor needs.
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
    p.add_argument("--trust_M", type=int, required=True,
                   help="Number of leading warm positions the predictor trusts")
    p.add_argument("--max_new", type=int, default=512)
    p.add_argument("--max_cycles", type=int, default=128)
    p.add_argument("--vocab_size", type=int, default=152064)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def simulate(model, tok, prompt_ids, args, rng):
    device = model.device
    K = args.K
    M = args.trust_M
    eos_id = tok.eos_token_id
    alt_eos = 151645
    committed = list(prompt_ids)
    total_tok = 0
    n_cycles = 0
    n_acc_history = []

    while total_tok < args.max_new and n_cycles < args.max_cycles:
        L = len(committed)
        # Warm forward: K random
        random_draft = [rng.randrange(args.vocab_size) for _ in range(K)]
        inp1 = torch.tensor([committed + random_draft], dtype=torch.long, device=device)
        warm = model(input_ids=inp1).logits[0, L - 1 : L - 1 + K, :].argmax(dim=-1).cpu().tolist()

        # Build verify draft: keep first M of warm, replace rest with fresh random
        noise_tail = [rng.randrange(args.vocab_size) for _ in range(K - M)]
        mixed_draft = warm[:M] + noise_tail  # length K

        # Verify forward
        inp2 = torch.tensor([committed + mixed_draft], dtype=torch.long, device=device)
        verify = model(input_ids=inp2).logits[0, L - 1 : L - 1 + K, :].argmax(dim=-1).cpu().tolist()

        # Standard greedy spec-decode: prefix match
        n_acc = 0
        for j in range(K):
            if mixed_draft[j] == verify[j]:
                n_acc += 1
            else:
                break
        n_acc_history.append(n_acc)

        for j in range(n_acc):
            committed.append(int(mixed_draft[j]))
        if n_acc < K:
            committed.append(int(verify[n_acc]))
            total_tok += n_acc + 1
        else:
            total_tok += K
        n_cycles += 1
        if committed[-1] in (eos_id, alt_eos):
            break

    return {
        "n_tokens": total_tok,
        "n_cycles": n_cycles,
        "tpf_verify": total_tok / max(1, n_cycles),
        "tpf_all_forwards": total_tok / max(1, 2 * n_cycles),
        "mean_n_acc": sum(n_acc_history) / max(1, len(n_acc_history)),
        "n_acc_history": n_acc_history,
        "trust_M": M,
    }


def main():
    args = parse_args()
    print(f"[sim] loading {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    prompts = [json.loads(l) for l in open(args.prompts_jsonl)]
    print(f"[sim] warm_trust_boundary trust_M={args.trust_M} K={args.K} max_new={args.max_new} n={len(prompts)}", flush=True)
    rng = random.Random(args.seed)
    out_fp = open(args.out_jsonl, "w")
    rows = []
    for i, p in enumerate(prompts):
        chat = [{"role": "user", "content": p["input"]}]
        text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        pid = tok(text, return_tensors="pt").input_ids[0].tolist()
        r = simulate(model, tok, pid, args, rng)
        r["batch_idx"] = i
        rows.append(r)
        out_fp.write(json.dumps(r) + "\n")
        out_fp.flush()
        print(f"[sim] [{i+1}/{len(prompts)}] trust_M={args.trust_M} ntok={r['n_tokens']:4d} cycles={r['n_cycles']:4d} TPF_verify={r['tpf_verify']:.3f} TPF_all={r['tpf_all_forwards']:.3f} mean_n_acc={r['mean_n_acc']:.2f}", flush=True)
    mean_v = sum(r["tpf_verify"] for r in rows) / len(rows)
    mean_a = sum(r["tpf_all_forwards"] for r in rows) / len(rows)
    mean_nacc = sum(r["mean_n_acc"] for r in rows) / len(rows)
    print(f"\n[sim] trust_M={args.trust_M}: mean TPF_verify={mean_v:.3f}, TPF_all={mean_a:.3f}, mean_n_acc/cycle={mean_nacc:.3f}", flush=True)


if __name__ == "__main__":
    main()
