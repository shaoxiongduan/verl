"""Perfect-iter-2-every-cycle sim.

Each outer "cycle" does:
  1. Warm forward: feed [committed | K random uniform tokens] through the model
     to get target_argmax_warm[0..K-1]. (This is "iter 1" of a fresh block-
     decode setup — converts random noise into a near-AR-continuation draft.)
  2. Verify forward: feed [committed | target_argmax_warm] back through the
     model to get target_argmax_verify[0..K-1]. Count n_acc = first j where
     target_argmax_warm[j] != target_argmax_verify[j].
  3. Commit target_argmax_warm[0..n_acc-1] + target_argmax_verify[n_acc]
     (i.e., n_acc + 1 tokens).
  4. Restart with a fresh K-random draft on the new committed sequence.

TPF metrics reported:
  - tpf_verify_only:   total_tokens / num_verify_forwards
    (treats the warm forward as "free" — like a draft model in real spec decode)
  - tpf_all_forwards:  total_tokens / (2 * num_verify_forwards)
    (counts BOTH forwards as Jacobi-equivalent iters — the apples-to-apples
    comparison vs block-decode where every forward is a single iter)

If the cons training's iter-2 burst is fully sustained, mean_n_acc should be
close to math_k3's reported iter-2 = 5.83 (and base ~4.4).

If it's significantly lower, the burst depends on something specific to natural
block-decode that this restart sim doesn't replicate.
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
    p.add_argument("--max_cycles", type=int, default=128)
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
    total_tok = 0
    n_cycles = 0
    n_acc_history = []
    while total_tok < args.max_new and n_cycles < args.max_cycles:
        L = len(committed)
        # Warm forward: random K-token draft
        random_draft = [rng.randrange(args.vocab_size) for _ in range(K)]
        inp1 = torch.tensor([committed + random_draft], dtype=torch.long, device=device)
        warm = model(input_ids=inp1).logits[0, L - 1 : L - 1 + K, :].argmax(dim=-1).cpu().tolist()
        # Verify forward: use warm as draft
        inp2 = torch.tensor([committed + warm], dtype=torch.long, device=device)
        verify = model(input_ids=inp2).logits[0, L - 1 : L - 1 + K, :].argmax(dim=-1).cpu().tolist()
        # n_acc = first j where warm[j] != verify[j]
        n_acc = 0
        for j in range(K):
            if warm[j] == verify[j]:
                n_acc += 1
            else:
                break
        n_acc_history.append(n_acc)
        for j in range(n_acc):
            committed.append(int(warm[j]))
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
        "tpf_verify_only": total_tok / max(1, n_cycles),
        "tpf_all_forwards": total_tok / max(1, 2 * n_cycles),
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
    print(f"[sim] warm_restart K={args.K} max_new={args.max_new} n={len(prompts)}", flush=True)
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
        print(f"[sim] [{i+1}/{len(prompts)}] ntok={r['n_tokens']:4d} cycles={r['n_cycles']:4d} "
              f"TPF_verify={r['tpf_verify_only']:.3f} TPF_all={r['tpf_all_forwards']:.3f} mean_n_acc={r['mean_n_acc']:.2f}", flush=True)
    mean_v = sum(r["tpf_verify_only"] for r in rows) / len(rows)
    mean_a = sum(r["tpf_all_forwards"] for r in rows) / len(rows)
    mean_nacc = sum(r["mean_n_acc"] for r in rows) / len(rows)
    print(f"\n[sim] K={args.K}: mean TPF_verify={mean_v:.3f}, TPF_all={mean_a:.3f}, mean_n_acc/cycle={mean_nacc:.3f}", flush=True)


if __name__ == "__main__":
    main()
