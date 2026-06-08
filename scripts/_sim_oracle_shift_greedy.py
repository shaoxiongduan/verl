"""Oracle-shift Jacobi sim under greedy (T=0).

Protocol (one forward per iter):
  1. Maintain a K-token draft and committed prefix
  2. Iter N: forward [committed | draft] → get target_argmax[0..K-1]
  3. n_acc = first j where draft[j] != target_argmax[j]
  4. Commit draft[:n_acc] + target_argmax[n_acc]   (n_acc + 1 tokens)
  5. Build next draft:
     - First (K - n_acc - 1) positions = target_argmax[n_acc+1:K]
       (the "shifted" cascade continuation — what natural Jacobi would do)
     - The last (n_acc + 1) positions = ORACLE RESET to fresh uniform noise
       (this is the +0.85 trick: replace the refilled tail with clean noise)
  6. With --keep_M, keep M "cascade transition" tokens after the shifted cascade
     before reset, matching oracle_shift{0..3} variants:
       keep_M=0  → mostly cascade-warm front, fresh noise tail
       keep_M=K  → no reset (= natural Jacobi)

Output: TPF (tokens / forwards), mean_n_acc per iter.
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
    p.add_argument("--keep_M", type=int, default=0,
                   help="Keep this many cascade tokens past n_acc+1 before noise reset. "
                        "0 = oracle_shift0 (max reset), K = no reset (natural Jacobi).")
    p.add_argument("--vocab_size", type=int, default=152064)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def simulate(model, tok, prompt_ids, args, rng):
    device = model.device
    K = args.K
    eos_id = tok.eos_token_id
    alt_eos = 151645
    committed = list(prompt_ids)
    # Initial draft: K fresh random tokens
    draft = [rng.randrange(args.vocab_size) for _ in range(K)]
    total_tok = 0
    n_iters = 0
    n_acc_history: list[int] = []
    while total_tok < args.max_new and n_iters < args.max_iters:
        L = len(committed)
        inp = torch.tensor([committed + draft], dtype=torch.long, device=device)
        target_argmax = model(input_ids=inp).logits[0, L - 1 : L - 1 + K, :].argmax(dim=-1).cpu().tolist()
        # Accept: longest prefix where draft[j] == target_argmax[j]
        n_acc = 0
        for j in range(K):
            if draft[j] == target_argmax[j]:
                n_acc += 1
            else:
                break
        n_acc_history.append(n_acc)
        for j in range(n_acc):
            committed.append(int(draft[j]))
        if n_acc < K:
            committed.append(int(target_argmax[n_acc]))
            total_tok += n_acc + 1
        else:
            total_tok += K
        n_iters += 1
        if committed[-1] in (eos_id, alt_eos):
            break
        # Build next draft: shifted cascade + oracle-reset tail
        # The natural cascade-shifted draft would be target_argmax[n_acc + 1:K]
        # of length K - n_acc - 1. Keep first args.keep_M of these, then noise.
        shifted = list(target_argmax[n_acc + 1:])  # length K - n_acc - 1
        # Take min(keep_M, len(shifted)) as the cascade keep
        keep_len = min(args.keep_M, len(shifted))
        new_draft = list(shifted[:keep_len])
        while len(new_draft) < K:
            new_draft.append(rng.randrange(args.vocab_size))
        draft = new_draft[:K]
    return {
        "n_tokens": total_tok,
        "n_iters": n_iters,
        "tpf": total_tok / max(1, n_iters),
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
    print(f"[sim] oracle_shift_greedy keep_M={args.keep_M} K={args.K} max_new={args.max_new} n={len(prompts)}", flush=True)
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
        print(f"[sim] [{i+1}/{len(prompts)}] keep_M={args.keep_M} ntok={r['n_tokens']:4d} iters={r['n_iters']:4d} TPF={r['tpf']:.3f} mean_n_acc={r['mean_n_acc']:.2f}", flush=True)
    mean_tpf = sum(r["tpf"] for r in rows) / len(rows)
    mean_nacc = sum(r["mean_n_acc"] for r in rows) / len(rows)
    print(f"\n[sim] keep_M={args.keep_M}: mean TPF={mean_tpf:.3f}, mean_n_acc/iter={mean_nacc:.3f}", flush=True)


if __name__ == "__main__":
    main()
