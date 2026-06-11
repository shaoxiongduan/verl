"""Trace first N iters of the Jacobi-predictor-refresh protocol for one prompt.

Prints per-iter:
  - input draft (first 12 tokens decoded)
  - cur_argmax (forward output, first 12 tokens decoded)
  - per-position match (draft[j] == cur_argmax[j]) → n_acc
  - per-position top1_prob and predictor P(correct)
  - the refresh decision (keep / reinit) per position
  - new_draft for next iter (first 12 tokens decoded)
"""
from __future__ import annotations
import argparse, json, random
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


LOGREG5_WEIGHTS = {
    "math_k3": {
        "w": [1.0651507, 0.54727219, 0.86257177, -0.12074035, 0.41574884],
        "b": 0.20852359,
    },
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--model_key", default="math_k3")
    p.add_argument("--prompts_jsonl", required=True)
    p.add_argument("--feat_jsonl", required=True)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--n_iters", type=int, default=5)
    p.add_argument("--vocab_size", type=int, default=152064)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt_idx", type=int, default=0)
    p.add_argument("--refresh", choices=["none", "logit_p", "logreg5"], default="logreg5")
    p.add_argument("--leak", choices=["boundary", "per_position"], default="boundary")
    p.add_argument("--threshold", type=float, default=0.5)
    return p.parse_args()


def compute_mu_sd(feat_jsonl: str):
    feats = []
    for line in open(feat_jsonl):
        r = json.loads(line)
        K = len(r["accept"])
        for j in range(K):
            feats.append([r["top1_prob"][j], r["entropy"][j], r["margin_log"][j],
                          r["top5_prob_sum"][j], j / K])
    X = np.array(feats, dtype=np.float64)
    return X.mean(axis=0), X.std(axis=0) + 1e-6


def fmt_tok(t, tok, w=10):
    s = repr(tok.decode([int(t)]))
    if len(s) > w: s = s[: w - 1] + "…"
    return s.ljust(w)


@torch.no_grad()
def main():
    args = parse_args()
    print(f"loading {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()

    mu, sd = compute_mu_sd(args.feat_jsonl)
    print(f"feature mu={mu}", flush=True)
    print(f"feature sd={sd}", flush=True)

    weights = LOGREG5_WEIGHTS[args.model_key]
    prompts = [json.loads(l) for l in open(args.prompts_jsonl)]
    p = prompts[args.prompt_idx]
    chat = [{"role": "user", "content": p["input"]}]
    text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    pid = tok(text, return_tensors="pt").input_ids[0].tolist()
    print(f"\nprompt #{args.prompt_idx}: {p['input'][:80]!r}")
    print(f"committed length: {len(pid)}")

    rng = random.Random(args.seed)
    K = args.K
    committed = list(pid)
    draft = [rng.randrange(args.vocab_size) for _ in range(K)]

    print(f"\nrefresh={args.refresh}  leak={args.leak}  thr={args.threshold}  K={K}")
    print(f"\nInitial K-random draft (first 8): {[tok.decode([t]) for t in draft[:8]]!r}")

    total_tok = 0
    for iter_n in range(args.n_iters):
        L = len(committed)
        print(f"\n{'='*100}")
        print(f"ITER {iter_n}  (L_committed={L}, total_committed_tokens_this_run={total_tok})")
        print(f"{'='*100}")
        print(f"draft (first 12): {' '.join(fmt_tok(t, tok) for t in draft[:12])}")

        inp = torch.tensor([committed + draft], dtype=torch.long, device=model.device)
        logits_K = model(input_ids=inp).logits[0, L - 1 : L - 1 + K, :]
        cur_argmax = logits_K.argmax(dim=-1).cpu().tolist()
        print(f"cur_argmax (first 12): {' '.join(fmt_tok(t, tok) for t in cur_argmax[:12])}")

        # Features
        fp32 = logits_K.float()
        probs = F.softmax(fp32, dim=-1)
        top5_vals, _ = probs.topk(5, dim=-1)
        top1 = top5_vals[:, 0].cpu().numpy()
        top5_sum = top5_vals.sum(dim=-1).cpu().numpy()
        entropy = -(probs * (probs.clamp_min(1e-12).log())).sum(dim=-1).cpu().numpy()
        top2_logits, _ = fp32.topk(2, dim=-1)
        margin_log = (top2_logits[:, 0] - top2_logits[:, 1]).cpu().numpy()

        # Jacobi convergence
        n_acc = 0
        for j in range(K):
            if cur_argmax[j] == draft[j]:
                n_acc += 1
            else:
                break
        match_arr = [int(cur_argmax[j] == draft[j]) for j in range(K)]
        print(f"per-pos match (first 16): {match_arr[:16]}")
        print(f"n_acc = {n_acc}")

        if n_acc > 0:
            print(f"COMMIT {n_acc} tokens: {[tok.decode([cur_argmax[j]]) for j in range(n_acc)]!r}")
            for j in range(n_acc):
                committed.append(int(cur_argmax[j]))
            total_tok += n_acc

        # Compute predictor scores
        if args.refresh == "logit_p":
            p_correct = top1.copy()
        elif args.refresh == "logreg5":
            pos = np.arange(K, dtype=np.float64) / K
            X = np.stack([top1, entropy, margin_log, top5_sum, pos], axis=1)
            Xs = (X - mu) / sd
            z = Xs @ np.array(weights["w"]) + weights["b"]
            p_correct = 1.0 / (1.0 + np.exp(-z))
        else:
            p_correct = np.ones(K)

        # Show feature table for first 12 positions
        print(f"\n{'j':>3} {'cur':>10} {'top1_p':>7} {'entropy':>7} {'margin':>7} {'P_correct':>10}")
        for j in range(12):
            print(f"{j:>3} {fmt_tok(cur_argmax[j], tok, 10)} {top1[j]:>7.3f} {entropy[j]:>7.3f} {margin_log[j]:>7.3f} {p_correct[j]:>10.4f}")

        # Refresh after shift
        shifted = list(cur_argmax[n_acc:K])
        n_keep = K - n_acc
        if args.refresh == "none":
            keep = np.ones(n_keep, dtype=bool)
        else:
            # Recompute predictor on the SHIFTED window using features at original positions [n_acc, K)
            if args.refresh == "logit_p":
                pc = top1[n_acc:K]
            else:
                t1s = top1[n_acc:K]; ets = entropy[n_acc:K]; mgs = margin_log[n_acc:K]
                tss = top5_sum[n_acc:K]
                pos_orig = np.arange(n_acc, K, dtype=np.float64) / K
                Xs = (np.stack([t1s, ets, mgs, tss, pos_orig], axis=1) - mu) / sd
                z = Xs @ np.array(weights["w"]) + weights["b"]
                pc = 1.0 / (1.0 + np.exp(-z))
            if args.leak == "boundary":
                below = np.where(pc < args.threshold)[0]
                b = int(below[0]) if len(below) else n_keep
                keep = np.zeros(n_keep, dtype=bool); keep[:b] = True
            else:
                keep = pc > args.threshold

        new_draft = [0] * K
        for j in range(n_keep):
            if keep[j]:
                new_draft[j] = shifted[j]
            else:
                new_draft[j] = rng.randrange(args.vocab_size)
        for j in range(n_keep, K):
            new_draft[j] = rng.randrange(args.vocab_size)

        print(f"\nrefresh: n_keep={n_keep}, keep_mask first 12: {[int(k) for k in keep[:12]]}")
        print(f"new_draft (first 12): {' '.join(fmt_tok(t, tok) for t in new_draft[:12])}")
        # Indicate which positions are kept vs random
        if args.refresh != "none":
            tags = []
            for j in range(min(12, K)):
                if j >= n_keep:
                    tags.append("new")
                elif keep[j]:
                    tags.append("kep")
                else:
                    tags.append("RND")
            print(f"src tag      (first 12): {' '.join(t.center(10) for t in tags)}")

        draft = new_draft

    print(f"\n\nfinal committed length: {len(committed)}, gained {total_tok} tokens in {args.n_iters} iters")


if __name__ == "__main__":
    main()
