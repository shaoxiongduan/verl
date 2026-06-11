"""Warm-restart with imprecise boundary predictor — ERROR-OFFSET parameterization.

Setup: a lightweight predictor identifies WHERE the clean prefix ends within
the K-block. The predictor outputs a boundary, but might be OFF by `offset_M`
tokens. We simulate the predictor being M tokens off (corrupts the first M
positions of the warm draft with random noise — those are positions the
predictor mislabeled).

  offset_M=0 = perfect predictor → equivalent to warm_restart (max TPF ~6)
  offset_M=N = predictor off by N positions, first N warm preds replaced w/ noise

Each cycle:
  1. Warm forward: input `[committed | K random]` → warm_argmax (K model preds)
  2. Corrupt first offset_M positions with random (predictor's mislabeled tail):
       corrupted = [M_random] + warm_argmax[M:]  (length K)
  3. Verify forward: input `[committed | corrupted]` → verify_argmax
  4. Accept: standard greedy spec-decode prefix match. Random tokens at
     positions 0..M-1 won't match → n_acc < M usually.
     BUT: if model is robust, verify[j] for j>M may still match warm_argmax[j]
     even though the corrupted prefix poisoned the context.
  5. Commit n_acc + 1 tokens.
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
    p.add_argument("--offset_M", type=int, required=True,
                   help="Predictor error: corrupt first M positions of warm draft with random noise")
    p.add_argument("--max_new", type=int, default=512)
    p.add_argument("--max_cycles", type=int, default=128)
    p.add_argument("--vocab_size", type=int, default=152064)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def simulate(model, tok, prompt_ids, args, rng):
    device = model.device
    K = args.K
    M = args.offset_M
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

        # CORRUPT first M positions of warm with random noise (predictor's error).
        # Beyond M, the predictor was right, so we keep warm_argmax.
        # offset_M=0 → no corruption, identical to warm_restart.
        corrupt_prefix = [rng.randrange(args.vocab_size) for _ in range(M)]
        mixed_draft = corrupt_prefix + warm[M:K]  # length K

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
        "offset_M": M,
    }


def main():
    args = parse_args()
    print(f"[sim] loading {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    prompts = [json.loads(l) for l in open(args.prompts_jsonl)]
    print(f"[sim] warm_trust_boundary trust_M={args.offset_M} K={args.K} max_new={args.max_new} n={len(prompts)}", flush=True)
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
        print(f"[sim] [{i+1}/{len(prompts)}] trust_M={args.offset_M} ntok={r['n_tokens']:4d} cycles={r['n_cycles']:4d} TPF_verify={r['tpf_verify']:.3f} TPF_all={r['tpf_all_forwards']:.3f} mean_n_acc={r['mean_n_acc']:.2f}", flush=True)
    mean_v = sum(r["tpf_verify"] for r in rows) / len(rows)
    mean_a = sum(r["tpf_all_forwards"] for r in rows) / len(rows)
    mean_nacc = sum(r["mean_n_acc"] for r in rows) / len(rows)
    print(f"\n[sim] trust_M={args.offset_M}: mean TPF_verify={mean_v:.3f}, TPF_all={mean_a:.3f}, mean_n_acc/cycle={mean_nacc:.3f}", flush=True)


if __name__ == "__main__":
    main()
