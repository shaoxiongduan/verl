"""Warm-restart with imperfect boundary predictor — LEAKED-TOKEN parameterization.

The user's protocol:
  - Each cycle = warm forward + verify forward (2 forwards per cycle).
  - TPF metric counts ONLY verify forwards (warm is "the lightweight predictor's
    free product"; in production it would come from a cheap model).
  - The boundary predictor identifies where iter-0's "correct" prefix ends.
  - shift=0 (PERFECT predictor): the next cycle's K-window is fully fresh
    random — predictor perfectly identifies all rejected positions and reinits
    them. Identical to warm_restart → ~5 n_acc + 1 = 6 TPF_verify.
  - shift=N (off by N): N of iter-0's REJECTED (wrong) predictions LEAK into
    the next cycle's K-window at positions 0..N-1. The predictor failed to
    identify those N as wrong, so they stay. The rest (K-N) is fresh random.

The leaked tokens are MODEL'S OWN WRONG PREDICTIONS (= warm[n_acc+1..K-1]),
NOT random noise — they're "in-distribution wrong" tokens that confuse the
next cycle's warm forward.

Each cycle:
  1. Warm forward: input `[committed | shift_leaked + (K-shift) random]`
     → warm_argmax (K model predictions on this mixed input)
  2. Verify forward: input `[committed | warm_argmax]`
     → verify_argmax
  3. Accept: greedy spec-decode prefix match warm == verify
  4. Commit n_acc + 1 tokens
  5. Compute LEAKED for next cycle: take warm[n_acc+1 : n_acc+1+shift]
     (= first `shift` of this cycle's REJECTED predictions)
  6. Build next cycle's mixed_draft: [leaked + (K-shift) fresh random]

Expected (math_k3, K=32):
  shift=0: TPF_verify ≈ 6.0 (= warm_restart baseline)
  shift=1..5: degraded TPF — quantifies tolerance to predictor error
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
    p.add_argument("--shift", type=int, required=True,
                   help="Predictor error: N wrong tokens leak from prev cycle into next mixed_draft")
    p.add_argument("--max_new", type=int, default=512)
    p.add_argument("--max_cycles", type=int, default=128)
    p.add_argument("--vocab_size", type=int, default=152064)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def simulate(model, tok, prompt_ids, args, rng):
    device = model.device
    K = args.K
    shift = args.shift
    eos_id = tok.eos_token_id
    alt_eos = 151645
    committed = list(prompt_ids)
    total_tok = 0
    n_cycles = 0
    n_acc_history = []
    leaked = []  # tokens leaked from prev cycle's rejected warm positions

    while total_tok < args.max_new and n_cycles < args.max_cycles:
        L = len(committed)
        # Build warm input: leaked (length up to shift) + fresh random for the rest
        if len(leaked) >= shift:
            leak_part = list(leaked[:shift])
        else:
            leak_part = list(leaked)
        fresh = [rng.randrange(args.vocab_size) for _ in range(K - len(leak_part))]
        mixed_draft = leak_part + fresh  # length K

        # Warm forward
        inp1 = torch.tensor([committed + mixed_draft], dtype=torch.long, device=device)
        warm = model(input_ids=inp1).logits[0, L - 1 : L - 1 + K, :].argmax(dim=-1).cpu().tolist()

        # Verify forward
        inp2 = torch.tensor([committed + warm], dtype=torch.long, device=device)
        verify = model(input_ids=inp2).logits[0, L - 1 : L - 1 + K, :].argmax(dim=-1).cpu().tolist()

        # Greedy spec-decode prefix acceptance
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

        # Compute next cycle's leaked tokens from THIS cycle's rejected warm positions
        # warm[n_acc+1 ... K-1] are the "wrong" predictions
        if n_acc + 1 < K:
            leaked = list(warm[n_acc + 1 : n_acc + 1 + shift])
        else:
            leaked = []

    return {
        "n_tokens": total_tok,
        "n_cycles": n_cycles,
        "tpf_verify": total_tok / max(1, n_cycles),
        "tpf_all_forwards": total_tok / max(1, 2 * n_cycles),
        "mean_n_acc": sum(n_acc_history) / max(1, len(n_acc_history)),
        "n_acc_history": n_acc_history,
        "shift": shift,
    }


def main():
    args = parse_args()
    print(f"[sim] loading {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    prompts = [json.loads(l) for l in open(args.prompts_jsonl)]
    print(f"[sim] warm_shift_leak shift={args.shift} K={args.K} max_new={args.max_new} n={len(prompts)}", flush=True)
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
        print(f"[sim] [{i+1}/{len(prompts)}] shift={args.shift} ntok={r['n_tokens']:4d} cycles={r['n_cycles']:4d} "
              f"TPF_verify={r['tpf_verify']:.3f} TPF_all={r['tpf_all_forwards']:.3f} mean_n_acc={r['mean_n_acc']:.2f}", flush=True)
    mean_v = sum(r["tpf_verify"] for r in rows) / len(rows)
    mean_a = sum(r["tpf_all_forwards"] for r in rows) / len(rows)
    mean_nacc = sum(r["mean_n_acc"] for r in rows) / len(rows)
    print(f"\n[sim] shift={args.shift}: mean TPF_verify={mean_v:.3f}, TPF_all={mean_a:.3f}, mean_n_acc/cycle={mean_nacc:.3f}", flush=True)


if __name__ == "__main__":
    main()
