"""Warm-restart sim with PREDICTOR-OFFSET error.

Tests error-tolerance: if a hypothetical "lightweight clean-prefix predictor"
produces M wrong tokens at the start of each cycle, can the model still
denoise the rest of the K-block and find useful AR continuations?

Per cycle:
  1. Warm forward: input `[committed | K_random]` → warm_argmax (K model preds)
  2. Corrupt the first M (offset) positions: corrupted = [M_random | warm_argmax[M:]]
     This simulates "predictor's first M positions are off"
  3. Verify forward: input `[committed | corrupted]` → verify_argmax
  4. Accept positions starting AT offset M: n_acc_offset = max j such that
     corrupted[M+j] == verify_argmax[M+j] for j in 0..K-M-1
     (i.e., prefix match starting from offset M)
  5. Commit n_acc + 1 tokens

Output:
  - mean_n_acc_after_offset: average accepted tokens past the M-token error
  - TPF_verify_post_offset = (mean_n_acc + 1) per verify forward
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
    p.add_argument("--offset", type=int, default=1,
                   help="Number of intentionally-wrong tokens at start of draft")
    p.add_argument("--max_new", type=int, default=512)
    p.add_argument("--max_cycles", type=int, default=128)
    p.add_argument("--vocab_size", type=int, default=152064)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def simulate(model, tok, prompt_ids, args, rng):
    device = model.device
    K = args.K
    M = args.offset
    eos_id = tok.eos_token_id
    alt_eos = 151645
    committed = list(prompt_ids)
    total_tok = 0
    n_cycles = 0
    n_acc_history = []

    while total_tok < args.max_new and n_cycles < args.max_cycles:
        L = len(committed)
        # Warm forward
        random_draft = [rng.randrange(args.vocab_size) for _ in range(K)]
        inp1 = torch.tensor([committed + random_draft], dtype=torch.long, device=device)
        warm = model(input_ids=inp1).logits[0, L - 1 : L - 1 + K, :].argmax(dim=-1).cpu().tolist()

        # Corrupt: replace first M positions with random
        corrupt_prefix = [rng.randrange(args.vocab_size) for _ in range(M)]
        corrupted = corrupt_prefix + warm[M:K]  # length K total

        # Verify forward
        inp2 = torch.tensor([committed + corrupted], dtype=torch.long, device=device)
        verify = model(input_ids=inp2).logits[0, L - 1 : L - 1 + K, :].argmax(dim=-1).cpu().tolist()

        # Accept starting from offset M
        n_acc = 0
        for j in range(M, K):
            if corrupted[j] == verify[j]:
                n_acc += 1
            else:
                break
        n_acc_history.append(n_acc)

        # Commit accepted tokens past offset + bonus
        for j in range(n_acc):
            committed.append(int(corrupted[M + j]))
        if M + n_acc < K:
            committed.append(int(verify[M + n_acc]))
            total_tok += n_acc + 1
        else:
            total_tok += K - M
        n_cycles += 1
        if committed[-1] in (eos_id, alt_eos):
            break

    return {
        "n_tokens": total_tok,
        "n_cycles": n_cycles,
        "tpf_verify_post_offset": total_tok / max(1, n_cycles),
        "tpf_all_forwards": total_tok / max(1, 2 * n_cycles),
        "mean_n_acc_post_offset": sum(n_acc_history) / max(1, len(n_acc_history)),
        "n_acc_history": n_acc_history,
        "offset": M,
    }


def main():
    args = parse_args()
    print(f"[sim] loading {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    prompts = [json.loads(l) for l in open(args.prompts_jsonl)]
    print(f"[sim] warm_offset offset={args.offset} K={args.K} max_new={args.max_new} n={len(prompts)}", flush=True)
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
        print(f"[sim] [{i+1}/{len(prompts)}] offset={args.offset} ntok={r['n_tokens']:4d} cycles={r['n_cycles']:4d} "
              f"TPF_verify_post={r['tpf_verify_post_offset']:.3f} TPF_all={r['tpf_all_forwards']:.3f} "
              f"mean_n_acc_post={r['mean_n_acc_post_offset']:.2f}", flush=True)
    mean_v = sum(r["tpf_verify_post_offset"] for r in rows) / len(rows)
    mean_a = sum(r["tpf_all_forwards"] for r in rows) / len(rows)
    mean_nacc = sum(r["mean_n_acc_post_offset"] for r in rows) / len(rows)
    print(f"\n[sim] offset={args.offset}: mean TPF_verify_post={mean_v:.3f}, TPF_all={mean_a:.3f}, mean_n_acc_post_offset={mean_nacc:.3f}", flush=True)


if __name__ == "__main__":
    main()
