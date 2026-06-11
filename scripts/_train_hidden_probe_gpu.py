"""GPU-accelerated per-position probe training.

Supports linear and 2-layer MLP heads on last-layer hidden state (+ optional
aux logit features). Targets: 'accept' or 'before_boundary'. Splits: cycle-level
random OR prompt-level holdout. Saves .npz weights compatible with the existing
sim (W, b, mu, sd, with_aux) for the linear head and (W1, b1, W2, b2, mu, sd,
with_aux, hidden_dim) for the MLP head.
"""
from __future__ import annotations
import argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--head", choices=["linear", "mlp"], default="linear")
    p.add_argument("--hidden", type=int, default=128, help="MLP hidden dim")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--with_aux", action="store_true")
    p.add_argument("--target", choices=["accept", "before_boundary"], default="before_boundary")
    p.add_argument("--pos_weight", type=float, default=1.0)
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--l2", type=float, default=1e-4)
    p.add_argument("--mb", type=int, default=8192)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--holdout_prompts_lt", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--report_every", type=int, default=100)
    return p.parse_args()


def auc(scores, labels):
    pos = scores[labels == 1]; neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    combined = np.concatenate([pos, neg]); order = combined.argsort()
    ranks = np.empty_like(order, dtype=np.float64); ranks[order] = np.arange(1, len(combined)+1)
    return (ranks[:len(pos)].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg))


def boundary_from_p(P_va_np, thr=0.3):
    K = P_va_np.shape[1]
    out = []
    for r in P_va_np:
        below = np.where(r < thr)[0]
        out.append(int(below[0]) if len(below) else K)
    return np.array(out)


def main():
    args = parse_args()
    print(f"loading {args.data}")
    z = np.load(args.data)
    H = z["hidden"]; A = z["accept"]; nacc = z["n_acc"]
    N, K, d = H.shape
    print(f"N={N} K={K} d_hidden={d}")

    X_h = H.reshape(N * K, d).astype(np.float32)
    if args.target == "accept":
        y = A.reshape(N * K).astype(np.float32)
    else:
        Y = (np.arange(K)[None, :] < nacc[:, None]).astype(np.float32)
        y = Y.reshape(N * K)
    print(f"target={args.target} positive_rate={y.mean():.4f}")

    if args.with_aux:
        aux_feats = [z["top1_prob"].reshape(N*K).astype(np.float32),
                     z["entropy"].reshape(N*K).astype(np.float32),
                     z["margin_log"].reshape(N*K).astype(np.float32),
                     z["top5_prob_sum"].reshape(N*K).astype(np.float32),
                     np.tile(np.arange(K, dtype=np.float32) / K, N)]
        X = np.concatenate([X_h, np.stack(aux_feats, axis=1)], axis=1)
    else:
        X = X_h
    d_in = X.shape[1]
    print(f"feature dim = {d_in}")
    mu = X.mean(axis=0); sd = X.std(axis=0) + 1e-6
    Xs = (X - mu) / sd

    rng = np.random.default_rng(args.seed)
    if args.holdout_prompts_lt is not None:
        pi = z["prompt_idx"]
        tr_cyc = np.where(pi >= args.holdout_prompts_lt)[0]
        va_cyc = np.where(pi < args.holdout_prompts_lt)[0]
        print(f"PROMPT split: tr={len(tr_cyc)} va={len(va_cyc)}")
    else:
        perm = rng.permutation(N)
        n_tr = int(N * (1 - args.val_frac))
        tr_cyc = perm[:n_tr]; va_cyc = perm[n_tr:]
        print(f"CYCLE split: tr={len(tr_cyc)} va={len(va_cyc)}")

    def flat_idx(cyc):
        return np.concatenate([np.arange(c*K, (c+1)*K) for c in cyc])
    tr_idx = flat_idx(tr_cyc); va_idx = flat_idx(va_cyc)

    device = args.device
    Xtr = torch.from_numpy(Xs[tr_idx]).to(device)
    ytr = torch.from_numpy(y[tr_idx]).to(device)
    Xva = torch.from_numpy(Xs[va_idx]).to(device)
    yva_np = y[va_idx]
    yva = torch.from_numpy(yva_np).to(device)

    w_tr = torch.where(ytr > 0.5, torch.tensor(args.pos_weight, device=device), torch.tensor(1.0, device=device))

    if args.head == "linear":
        model = nn.Linear(d_in, 1, bias=True).to(device)
    else:
        model = nn.Sequential(
            nn.Linear(d_in, args.hidden), nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden, 1),
        ).to(device)
    print(f"head={args.head} hidden={args.hidden} dropout={args.dropout}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2)
    t0 = time.time()
    best_va_auc = -1.0
    best_state = None
    n_tr_pos = len(tr_idx)
    for it in range(args.iters):
        model.train()
        idx = torch.randint(0, n_tr_pos, (min(args.mb, n_tr_pos),), device=device)
        z_out = model(Xtr[idx]).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(z_out, ytr[idx], weight=w_tr[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if it % args.report_every == 0 or it == args.iters - 1:
            model.eval()
            with torch.no_grad():
                # Full val pass
                z_va = model(Xva).squeeze(-1)
                p_va = torch.sigmoid(z_va).cpu().numpy()
            auc_va = auc(p_va, yva_np.astype(int))
            pred = (p_va > 0.5).astype(int)
            acc_va = (pred == yva_np.astype(int)).mean()
            tp = ((pred == 1) & (yva_np == 1)).sum(); fp = ((pred == 1) & (yva_np == 0)).sum()
            fn = ((pred == 0) & (yva_np == 1)).sum(); tn = ((pred == 0) & (yva_np == 0)).sum()
            # Boundary MAE
            P_va_mat = p_va.reshape(len(va_cyc), K)
            pred_b = boundary_from_p(P_va_mat, thr=0.3)
            true_b = nacc[va_cyc].astype(int)
            mae_b = np.abs(pred_b - true_b).mean()
            signed_b = (pred_b - true_b).mean()
            w1 = (np.abs(pred_b - true_b) <= 1).mean()
            elapsed = time.time() - t0
            print(f"  it={it:>4d}  tr_loss={loss.item():.4f}  AUC={auc_va:.4f}  acc={acc_va:.4f}  FN/FP={fn}/{fp}  MAE_b={mae_b:.3f}  signed_b={signed_b:+.3f}  ±1_b={w1:.3f} ({elapsed:.1f}s)")
            if auc_va > best_va_auc:
                best_va_auc = auc_va
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\nrestored best (val_AUC={best_va_auc:.4f})")

    # Save in the format expected by the sim
    out_dict = {"mu": mu, "sd": sd, "with_aux": args.with_aux}
    if args.head == "linear":
        W = model.weight.detach().cpu().numpy().reshape(-1)
        b = float(model.bias.detach().cpu().numpy())
        out_dict.update({"W": W, "b": b})
    else:
        W1 = model[0].weight.detach().cpu().numpy()  # (h, d_in)
        b1 = model[0].bias.detach().cpu().numpy()
        W2 = model[3].weight.detach().cpu().numpy()  # (1, h)
        b2 = model[3].bias.detach().cpu().numpy()
        out_dict.update({"W1": W1.T, "b1": b1, "W2": W2.T, "b2": b2, "hidden_dim": args.hidden})
    np.savez(args.out, **out_dict)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
