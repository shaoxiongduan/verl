"""Visualize per-position features vs true boundary for sample cycles.

For each of ~20 sample cycles (varied n_acc), produce one panel:
  - y-axis: feature value (top1_prob, entropy/8, margin/8, P_correct from probe)
  - x-axis: K-block position j
  - vertical line at true n_acc (the boundary)
  - draft tokens annotated along the bottom

Saves a multi-panel PNG.
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="boundary_h_*.npz with hidden + aux features")
    p.add_argument("--probe", required=True, help="trained probe.npz (W, b, mu, sd, with_aux)")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True, help="output PNG")
    p.add_argument("--n_examples", type=int, default=20)
    p.add_argument("--targets", type=str, default="1,2,3,4,5,6,7,8,9,10,12,15,18,22")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"loading {args.data}")
    z = np.load(args.data)
    pz = np.load(args.probe, allow_pickle=True)
    H = z["hidden"]; nacc = z["n_acc"]
    top1 = z["top1_prob"]; ent = z["entropy"]; marg = z["margin_log"]; t5s = z["top5_prob_sum"]
    warm = z["warm_argmax"]; verify = z["verify_argmax"]
    N, K, d = H.shape
    print(f"N={N}  K={K}  d_hidden={d}")

    W = pz["W"].astype(np.float32); b = float(pz["b"])
    mu = pz["mu"].astype(np.float32); sd = pz["sd"].astype(np.float32)
    with_aux = bool(pz["with_aux"])

    # Compute per-position P_correct for ALL cycles (vectorized).
    X_h = H.reshape(N * K, d).astype(np.float32)
    if with_aux:
        aux_feats = [z["top1_prob"].reshape(N*K).astype(np.float32),
                     z["entropy"].reshape(N*K).astype(np.float32),
                     z["margin_log"].reshape(N*K).astype(np.float32),
                     z["top5_prob_sum"].reshape(N*K).astype(np.float32),
                     np.tile(np.arange(K, dtype=np.float32) / K, N)]
        X = np.concatenate([X_h, np.stack(aux_feats, axis=1)], axis=1)
    else:
        X = X_h
    Xs = (X - mu) / sd
    P = (1 / (1 + np.exp(-(Xs @ W + b)))).reshape(N, K)
    print(f"computed P_correct for {N} cycles")

    print(f"loading tokenizer {args.tokenizer}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    # Pick cycles: try to get one per target n_acc value
    targets = [int(x) for x in args.targets.split(",")]
    by_nacc = defaultdict(list)
    for i in range(N):
        by_nacc[int(nacc[i])].append(i)
    picks = []
    rng = np.random.default_rng(args.seed)
    for t in targets:
        if t in by_nacc:
            chosen = rng.choice(by_nacc[t])
            picks.append((t, int(chosen)))
        if len(picks) >= args.n_examples:
            break
    # Fill out if too few
    while len(picks) < args.n_examples:
        t = rng.integers(1, 12)
        if int(t) in by_nacc:
            picks.append((int(t), int(rng.choice(by_nacc[int(t)]))))

    n_panels = len(picks)
    cols = 4
    rows = (n_panels + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 3.6), squeeze=False)
    plt.subplots_adjust(hspace=0.8, wspace=0.25)

    js = np.arange(K)
    for panel_i, (target_n, idx) in enumerate(picks):
        ax = axes[panel_i // cols][panel_i % cols]
        true_b = int(nacc[idx])
        # Plot lines
        ax.plot(js, top1[idx].astype(np.float32), label="top1_p", color="C0", linewidth=1.5)
        ax.plot(js, np.clip(ent[idx].astype(np.float32) / 8.0, 0, 1), label="entropy/8", color="C1", linewidth=1.0, linestyle="--")
        ax.plot(js, np.clip(marg[idx].astype(np.float32) / 8.0, 0, 1), label="margin/8", color="C2", linewidth=1.0, linestyle=":")
        ax.plot(js, P[idx], label="probe P", color="C3", linewidth=2.0)
        ax.axhline(0.5, color="gray", linewidth=0.5, alpha=0.5)
        # True boundary vertical line
        ax.axvline(true_b - 0.5, color="red", linewidth=2.0, alpha=0.7, label=f"true_b={true_b}")
        ax.set_xlim(-0.5, K - 0.5)
        ax.set_ylim(-0.05, 1.1)
        # Title with cycle info
        ax.set_title(f"n_acc={true_b}  (cyc#{idx})", fontsize=10)
        ax.set_xlabel("position j", fontsize=8)
        if panel_i % cols == 0:
            ax.set_ylabel("feature value", fontsize=8)
        # Decode warm tokens at bottom; show first ~12 chars per token
        toks = warm[idx].tolist()
        decoded = []
        for j in range(K):
            s = tok.decode([int(toks[j])])
            s = s.replace("\n", "\\n").replace(" ", "·")
            if len(s) > 8:
                s = s[:7] + "…"
            decoded.append(s)
        for j in range(K):
            color = "darkgreen" if j < true_b else "darkred"
            ax.text(j, -0.18 - (j % 2) * 0.08, decoded[j], rotation=90, fontsize=5, ha="center", va="top", color=color)
        ax.set_xticks(range(0, K, 4))
        ax.tick_params(axis="both", labelsize=7)
        if panel_i == 0:
            ax.legend(fontsize=7, loc="upper right")

    # Hide unused panels
    for k in range(n_panels, rows * cols):
        axes[k // cols][k % cols].axis("off")

    plt.suptitle("Per-position features vs true boundary  (red line = true n_acc; bottom annotations = warm argmax)",
                 fontsize=11)
    plt.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
