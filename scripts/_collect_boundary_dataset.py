"""Collect boundary-predictor training data from warm_restart cycles.

For each cycle, dump per-position features (computed from the warm forward's
logits) and per-position labels (computed from the verify forward):

  Features (per-position j in [0, K)):
    - top1_prob: max softmax over vocab
    - top5_prob_sum: cumulative mass on top-5 (concentration proxy)
    - entropy: full distribution entropy in nats
    - margin_log: logit_top1 - logit_top2 (log-domain confidence)
    - top1_token: argmax (int, vocab idx — useful for token-identity features)
    - position: j in [0, K)
    - L_committed: context length before the K-block
  Label (per-position j):
    - accept[j] = int(warm_argmax[j] == verify_argmax[j])
  Cycle-level label:
    - n_acc: first j where accept[j] == 0 (or K if all accepted)

This data lets us fit lightweight boundary predictors (logistic regression on
top1_prob, small MLP, etc.) and ask: how well does a cheap feature like top-1
probability separate "accept" from "reject" positions?

Dumps a JSONL file: one line per cycle, with all K-position features + labels.
"""
from __future__ import annotations
import argparse, json, math, random, torch, torch.nn.functional as F
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
    p.add_argument("--shift", type=int, default=0,
                   help="leakage shift (0 = warm_restart = clean K-noise restart)")
    return p.parse_args()


@torch.no_grad()
def per_position_features(logits_K: torch.Tensor):
    """logits_K: (K, V).  Returns dict of per-position scalar features."""
    K, V = logits_K.shape
    fp32 = logits_K.float()
    probs = F.softmax(fp32, dim=-1)
    top5_vals, top5_idx = probs.topk(5, dim=-1)            # (K, 5)
    top1_prob = top5_vals[:, 0]                            # (K,)
    top5_sum = top5_vals.sum(dim=-1)                       # (K,)
    entropy = -(probs * (probs.clamp_min(1e-12).log())).sum(dim=-1)  # (K,)
    top2_logits, _ = fp32.topk(2, dim=-1)
    margin_log = top2_logits[:, 0] - top2_logits[:, 1]     # (K,)
    return {
        "top1_prob": top1_prob.cpu().tolist(),
        "top5_prob_sum": top5_sum.cpu().tolist(),
        "entropy": entropy.cpu().tolist(),
        "margin_log": margin_log.cpu().tolist(),
        "top1_token": top5_idx[:, 0].cpu().tolist(),
        "top5_tokens": top5_idx.cpu().tolist(),
        "top5_probs": top5_vals.cpu().tolist(),
    }


@torch.no_grad()
def collect_for_prompt(model, prompt_ids, args, rng, out_fp, prompt_idx):
    device = model.device
    K = args.K
    shift = args.shift
    eos_id = 151645
    pad_id = 151643
    stop_ids = {eos_id, pad_id}
    committed = list(prompt_ids)
    total_tok = 0
    n_cycles = 0
    leaked = []

    while total_tok < args.max_new and n_cycles < args.max_cycles:
        L = len(committed)
        leak_part = list(leaked[:shift]) if leaked else []
        fresh = [rng.randrange(args.vocab_size) for _ in range(K - len(leak_part))]
        mixed_draft = leak_part + fresh

        # Warm forward
        inp1 = torch.tensor([committed + mixed_draft], dtype=torch.long, device=device)
        warm_logits = model(input_ids=inp1).logits[0, L - 1 : L - 1 + K, :]
        warm_argmax = warm_logits.argmax(dim=-1).cpu().tolist()
        feats = per_position_features(warm_logits)

        # Verify forward
        inp2 = torch.tensor([committed + warm_argmax], dtype=torch.long, device=device)
        verify_argmax = model(input_ids=inp2).logits[0, L - 1 : L - 1 + K, :].argmax(dim=-1).cpu().tolist()

        # Per-position accept label
        accept = [int(warm_argmax[j] == verify_argmax[j]) for j in range(K)]
        # n_acc = longest prefix where accept[j]==1
        n_acc = 0
        for j in range(K):
            if accept[j]:
                n_acc += 1
            else:
                break

        # Dump one JSONL line for this cycle
        rec = {
            "prompt_idx": prompt_idx,
            "cycle_idx": n_cycles,
            "L_committed": L,
            "mixed_draft": mixed_draft,
            "warm_argmax": warm_argmax,
            "verify_argmax": verify_argmax,
            "accept": accept,
            "n_acc": n_acc,
            **feats,
        }
        out_fp.write(json.dumps(rec) + "\n")
        out_fp.flush()

        # Commit (same logic as warm_shift_leak)
        for j in range(n_acc):
            committed.append(int(warm_argmax[j]))
        if n_acc < K:
            committed.append(int(verify_argmax[n_acc]))
            total_tok += n_acc + 1
        else:
            total_tok += K
        n_cycles += 1
        if committed[-1] in stop_ids:
            break

        # Compute leaked for next cycle
        if n_acc + 1 < K:
            leaked = list(warm_argmax[n_acc + 1 : n_acc + 1 + shift])
        else:
            leaked = []

    return n_cycles, total_tok


def main():
    args = parse_args()
    print(f"[col] loading {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0"
    )
    model.eval()
    prompts = [json.loads(l) for l in open(args.prompts_jsonl)]
    rng = random.Random(args.seed)
    print(f"[col] K={args.K} shift={args.shift} max_new={args.max_new} n_prompts={len(prompts)}", flush=True)
    out_fp = open(args.out_jsonl, "w")
    total_cycles = 0
    total_tokens = 0
    for i, p in enumerate(prompts):
        chat = [{"role": "user", "content": p["input"]}]
        text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        pid = tok(text, return_tensors="pt").input_ids[0].tolist()
        nc, nt = collect_for_prompt(model, pid, args, rng, out_fp, i)
        total_cycles += nc
        total_tokens += nt
        print(f"[col] [{i+1}/{len(prompts)}] cycles={nc} tokens={nt} total_cycles={total_cycles}", flush=True)
    print(f"\n[col] DONE: {total_cycles} cycles dumped, {total_tokens} tokens generated", flush=True)


if __name__ == "__main__":
    main()
