"""Warm-restart sim with a LEARNED boundary predictor in the leakage role.

Same 2-forward-per-cycle protocol as _sim_warm_shift_leak.py, but instead of
a fixed `shift` of leaked tokens, the predictor's per-position P(correct) is
used to decide which warm tokens to retain in the next cycle's mixed_draft.

Two predictor modes:
  --predictor logit_p   : Simple rule. Keep warm[j] if top1_prob[j] > thr.
                          (Best single-feature ceiling per AUC analysis.)
  --predictor logreg5   : 5-feature logistic regression. Apply std-scaled
                          weights to (top1, entropy, margin, top5sum, pos/K).
                          Threshold at 0.5 → keep if P_correct > 0.5.

Two leakage modes:
  --leak boundary       : "First-j-where-predicted-wrong" boundary. Keep
                          warm[0..b], reinit warm[b..K] to fresh random.
                          This is the shift-leak analog (shift = b).
  --leak per_position   : Independent per-position decision. Keep warm[j]
                          where P_correct[j] > thr; reinit elsewhere.

We compare to oracle (= use TRUE n_acc as boundary) which is the warm_restart
shift=0 baseline.
"""
from __future__ import annotations
import argparse, json, math, random
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# Logreg weights computed by _analyze_boundary_mlp.py on standardized features.
# Feature order: (top1_prob, entropy, margin, top5_sum, pos/32) + bias.
# These are on STANDARDIZED features (zero-mean, unit-std on training data).
LOGREG5_WEIGHTS = {
    "math_k3": {
        # 80/20 split val AUC=0.7662
        "w": [1.0651507, 0.54727219, 0.86257177, -0.12074035, 0.41574884],
        "b": 0.20852359,
        # Training-set feature means & stds (from collected data)
        "mu": None,  # Computed at runtime from a small dataset sample
        "sd": None,
    },
    "base": {
        # val AUC=0.6996
        "w": [0.60511182, -0.16665991, 0.63486387, 0.00439597, 0.50162427],
        "b": -0.23659919,
        "mu": None,
        "sd": None,
    },
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--model_key", required=True, choices=["math_k3", "base"])
    p.add_argument("--prompts_jsonl", required=True)
    p.add_argument("--feat_jsonl", required=True,
                   help="Path to a previously-collected boundary_data_*.jsonl "
                        "to derive feature normalization mu/sd from.")
    p.add_argument("--out_jsonl", required=True)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--max_new", type=int, default=512)
    p.add_argument("--max_cycles", type=int, default=128)
    p.add_argument("--vocab_size", type=int, default=152064)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--predictor", choices=["logit_p", "logreg5"], default="logreg5")
    p.add_argument("--leak", choices=["boundary", "per_position"], default="boundary")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="For logit_p: top1_prob threshold to keep. For logreg5: P_correct threshold.")
    return p.parse_args()


def compute_mu_sd(feat_jsonl: str):
    """Compute per-feature mu and sd from a collected dataset."""
    feats = []
    for line in open(feat_jsonl):
        r = json.loads(line)
        K = len(r["accept"])
        for j in range(K):
            feats.append([r["top1_prob"][j], r["entropy"][j], r["margin_log"][j],
                          r["top5_prob_sum"][j], j / K])
    X = np.array(feats, dtype=np.float64)
    return X.mean(axis=0), X.std(axis=0) + 1e-6


@torch.no_grad()
def per_position_features_np(logits_K: torch.Tensor):
    fp32 = logits_K.float()
    probs = F.softmax(fp32, dim=-1)
    top5_vals, _ = probs.topk(5, dim=-1)
    top1_prob = top5_vals[:, 0].cpu().numpy()
    top5_sum = top5_vals.sum(dim=-1).cpu().numpy()
    entropy = -(probs * (probs.clamp_min(1e-12).log())).sum(dim=-1).cpu().numpy()
    top2_logits, _ = fp32.topk(2, dim=-1)
    margin_log = (top2_logits[:, 0] - top2_logits[:, 1]).cpu().numpy()
    return top1_prob, entropy, margin_log, top5_sum


def predict_per_position(args, top1, ent, marg, t5s, K, w_dict):
    """Return per-position P(correct) in [0, 1]^K."""
    if args.predictor == "logit_p":
        return top1  # use raw top1_prob as P(correct) proxy
    pos = np.arange(K, dtype=np.float64) / K
    X = np.stack([top1, ent, marg, t5s, pos], axis=1)
    Xs = (X - w_dict["mu"]) / w_dict["sd"]
    z = Xs @ np.array(w_dict["w"]) + w_dict["b"]
    return 1 / (1 + np.exp(-z))


def decide_keep_mask(args, p_correct):
    """Return boolean mask of which warm[j] to KEEP. Length K."""
    if args.leak == "per_position":
        return p_correct > args.threshold
    # boundary mode: first j where p_correct < threshold is the boundary
    K = len(p_correct)
    below = np.where(p_correct < args.threshold)[0]
    b = int(below[0]) if len(below) else K
    mask = np.zeros(K, dtype=bool); mask[:b] = True
    return mask


@torch.no_grad()
def simulate(model, tok, prompt_ids, args, rng, w_dict):
    device = model.device
    K = args.K
    eos_id = 151645; pad_id = 151643
    stop_ids = {eos_id, pad_id}
    committed = list(prompt_ids)
    total_tok = 0; n_cycles = 0
    n_acc_history = []
    pred_boundary_history = []
    true_boundary_history = []
    # First cycle: all fresh random
    mixed_draft = [rng.randrange(args.vocab_size) for _ in range(K)]

    while total_tok < args.max_new and n_cycles < args.max_cycles:
        L = len(committed)
        # Warm forward
        inp1 = torch.tensor([committed + mixed_draft], dtype=torch.long, device=device)
        warm_logits = model(input_ids=inp1).logits[0, L - 1 : L - 1 + K, :]
        warm_argmax = warm_logits.argmax(dim=-1).cpu().tolist()
        top1, ent, marg, t5s = per_position_features_np(warm_logits)
        # Verify forward
        inp2 = torch.tensor([committed + warm_argmax], dtype=torch.long, device=device)
        verify_argmax = model(input_ids=inp2).logits[0, L - 1 : L - 1 + K, :].argmax(dim=-1).cpu().tolist()

        # Standard greedy spec-decode accept (same as warm_restart/shift_leak)
        n_acc = 0
        for j in range(K):
            if warm_argmax[j] == verify_argmax[j]:
                n_acc += 1
            else:
                break
        n_acc_history.append(n_acc)
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

        # Apply predictor to current warm output → decide what to retain.
        p_correct = predict_per_position(args, top1, ent, marg, t5s, K, w_dict)
        keep_mask = decide_keep_mask(args, p_correct)
        # Predicted boundary = max index that is True (boundary mode) or count(True) (per_position mode)
        if args.leak == "boundary":
            pred_b = int(keep_mask.sum())
        else:
            pred_b = int(keep_mask.sum())  # not really a boundary in per_position mode
        pred_boundary_history.append(pred_b)
        true_boundary_history.append(n_acc)

        # Build next cycle's mixed_draft.
        # Shift the K-window forward by (n_acc + 1) positions: positions
        # [n_acc+1, K) of THIS cycle's warm correspond to positions [0, K-n_acc-1)
        # of next cycle's K-window. The first K-n_acc-1 positions of next cycle's
        # warm input can reuse warm[n_acc+1..K-1] IF predictor says they're worth
        # keeping. Remaining positions = fresh random.
        next_draft = [0] * K
        shift_amount = n_acc + 1
        # For position j in next K-window, originally came from warm position (j + shift_amount)
        for j in range(K):
            src = j + shift_amount
            if src < K and keep_mask[src]:
                next_draft[j] = int(warm_argmax[src])
            else:
                next_draft[j] = rng.randrange(args.vocab_size)
        mixed_draft = next_draft

    return {
        "n_tokens": total_tok,
        "n_cycles": n_cycles,
        "tpf_verify": total_tok / max(1, n_cycles),
        "tpf_all_forwards": total_tok / max(1, 2 * n_cycles),
        "mean_n_acc": sum(n_acc_history) / max(1, len(n_acc_history)),
        "n_acc_history": n_acc_history,
        "pred_boundary_history": pred_boundary_history,
        "true_boundary_history": true_boundary_history,
        "predictor": args.predictor,
        "leak_mode": args.leak,
        "threshold": args.threshold,
    }


def main():
    args = parse_args()
    print(f"[predsim] loading {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    prompts = [json.loads(l) for l in open(args.prompts_jsonl)]
    rng = random.Random(args.seed)

    w_dict = dict(LOGREG5_WEIGHTS[args.model_key])
    print(f"[predsim] computing feature mu/sd from {args.feat_jsonl}", flush=True)
    w_dict["mu"], w_dict["sd"] = compute_mu_sd(args.feat_jsonl)
    print(f"[predsim] feature mu = {w_dict['mu']}", flush=True)
    print(f"[predsim] feature sd = {w_dict['sd']}", flush=True)
    print(f"[predsim] predictor={args.predictor}  leak={args.leak}  thr={args.threshold}  K={args.K}  max_new={args.max_new}", flush=True)

    out_fp = open(args.out_jsonl, "w")
    rows = []
    for i, p in enumerate(prompts):
        chat = [{"role": "user", "content": p["input"]}]
        text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        pid = tok(text, return_tensors="pt").input_ids[0].tolist()
        r = simulate(model, tok, pid, args, rng, w_dict)
        r["batch_idx"] = i
        rows.append(r)
        out_fp.write(json.dumps(r) + "\n")
        out_fp.flush()
        # Boundary calibration: mean signed error pred - true
        be = [pb - tb for pb, tb in zip(r["pred_boundary_history"], r["true_boundary_history"])]
        mean_be = sum(be) / max(1, len(be))
        print(f"[predsim] [{i+1}/{len(prompts)}] ntok={r['n_tokens']:4d} cyc={r['n_cycles']:4d} "
              f"TPF_verify={r['tpf_verify']:.3f} TPF_all={r['tpf_all_forwards']:.3f} "
              f"mean_n_acc={r['mean_n_acc']:.2f}  mean(pred-true)={mean_be:+.2f}", flush=True)

    total_tok = sum(r["n_tokens"] for r in rows)
    total_cyc = sum(r["n_cycles"] for r in rows)
    print(f"\n[predsim] CORPUS: Σtok={total_tok}  Σcyc={total_cyc}  "
          f"TPF_verify={total_tok/max(1,total_cyc):.4f}  "
          f"TPF_all={total_tok/max(1,2*total_cyc):.4f}", flush=True)
    print(f"[predsim] per-prompt-mean TPF_verify = {sum(r['tpf_verify'] for r in rows)/len(rows):.4f}", flush=True)


if __name__ == "__main__":
    main()
