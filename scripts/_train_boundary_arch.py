"""GPU-trained alternative architectures for boundary prediction.

Heads:
  - per_pos_linear  : current baseline; predicts P(j<n_acc) per position from h[j]+aux
  - per_pos_mlp     : MLP-128 per-position; current best
  - global_linear   : single Linear(K*(d_h + 5_aux), K+1) → softmax → boundary
  - global_mlp      : MLP-256 over flatten(K*hidden+5*K) → K+1 softmax
  - attn            : tiny 1-layer transformer over K positions → per-position score then first-crossing
  - attn_global     : same transformer but outputs single softmax(K+1) via CLS-style pooling

All can train with cycle-level or prompt-level holdout split.
Loss: per-position uses BCE on before_boundary label; global uses CE on n_acc class.
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
    p.add_argument("--arch", choices=["per_pos_linear", "per_pos_mlp", "global_linear",
                                      "global_mlp", "attn", "attn_global"], required=True)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--with_aux", action="store_true", default=True)
    p.add_argument("--pos_weight", type=float, default=1.0)
    p.add_argument("--iters", type=int, default=4000)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--l2", type=float, default=1e-4)
    p.add_argument("--mb", type=int, default=512, help="minibatch size in CYCLES (not positions)")
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--holdout_prompts_lt", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--report_every", type=int, default=200)
    return p.parse_args()


def auc(scores, labels):
    pos = scores[labels == 1]; neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    combined = np.concatenate([pos, neg]); order = combined.argsort()
    ranks = np.empty_like(order, dtype=np.float64); ranks[order] = np.arange(1, len(combined)+1)
    return (ranks[:len(pos)].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg))


def first_crossing(P_mat, thr=0.3):
    out = []
    K = P_mat.shape[1]
    for r in P_mat:
        below = np.where(r < thr)[0]
        out.append(int(below[0]) if len(below) else K)
    return np.array(out)


class PerPosLinear(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.lin = nn.Linear(d_in, 1)
    def forward(self, x):  # x: (B, K, d_in)
        return self.lin(x).squeeze(-1)  # (B, K)


class PerPosMLP(nn.Module):
    def __init__(self, d_in, hidden=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)


class GlobalLinear(nn.Module):
    """Flatten K positions → Linear → (K+1) softmax."""
    def __init__(self, d_in, K):
        super().__init__()
        self.K = K
        self.lin = nn.Linear(d_in * K, K + 1)
    def forward(self, x):  # x: (B, K, d_in)
        flat = x.flatten(1)
        return self.lin(flat)  # (B, K+1) logits


class GlobalMLP(nn.Module):
    def __init__(self, d_in, K, hidden=256, dropout=0.1):
        super().__init__()
        self.K = K
        self.net = nn.Sequential(
            nn.Linear(d_in * K, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, K + 1))
    def forward(self, x):
        flat = x.flatten(1)
        return self.net(flat)


class TinyAttn(nn.Module):
    """Tiny transformer over K positions, per-position scoring head."""
    def __init__(self, d_in, K, embed=64, n_heads=4, n_layers=1, dropout=0.1, global_head=False):
        super().__init__()
        self.K = K
        self.global_head = global_head
        self.proj = nn.Linear(d_in, embed)
        self.pos_emb = nn.Embedding(K + 1, embed)  # +1 for optional CLS
        layer = nn.TransformerEncoderLayer(d_model=embed, nhead=n_heads, dim_feedforward=4*embed,
                                           dropout=dropout, batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        if global_head:
            self.head = nn.Linear(embed, K + 1)
            self.cls = nn.Parameter(torch.randn(1, 1, embed) * 0.02)
        else:
            self.head = nn.Linear(embed, 1)

    def forward(self, x):  # x: (B, K, d_in)
        B = x.shape[0]; K = x.shape[1]
        h = self.proj(x)
        pos_ids = torch.arange(K, device=x.device).unsqueeze(0).expand(B, K)
        h = h + self.pos_emb(pos_ids)
        if self.global_head:
            cls = self.cls.expand(B, 1, -1) + self.pos_emb(torch.tensor([K], device=x.device)).unsqueeze(0)
            h = torch.cat([cls, h], dim=1)
            h = self.enc(h)
            return self.head(h[:, 0])  # (B, K+1) logits
        else:
            h = self.enc(h)
            return self.head(h).squeeze(-1)  # (B, K)


def main():
    args = parse_args()
    print(f"loading {args.data}")
    z = np.load(args.data)
    H = z["hidden"]; A = z["accept"]; nacc = z["n_acc"]
    N, K, d = H.shape
    print(f"N={N} K={K} d_hidden={d}")

    aux = np.stack([z["top1_prob"], z["entropy"], z["margin_log"], z["top5_prob_sum"]], axis=-1).astype(np.float32)  # (N, K, 4)
    pos = np.tile(np.arange(K, dtype=np.float32) / K, (N, 1))[..., None]  # (N, K, 1)
    if args.with_aux:
        feats = np.concatenate([H.astype(np.float32), aux, pos], axis=-1)  # (N, K, d+5)
    else:
        feats = H.astype(np.float32)
    d_in = feats.shape[-1]
    print(f"feature dim per pos = {d_in}")

    # Standardize per-feature across (N, K)
    flat = feats.reshape(-1, d_in)
    mu = flat.mean(axis=0); sd = flat.std(axis=0) + 1e-6
    feats = (feats - mu) / sd  # broadcasts over batch dim
    print(f"standardized features")

    # Labels
    Y_pp = (np.arange(K)[None, :] < nacc[:, None]).astype(np.float32)  # (N, K) for per-pos head
    Y_g = nacc.astype(np.int64)  # (N,) for global head

    rng = np.random.default_rng(args.seed)
    if args.holdout_prompts_lt is not None:
        pi = z["prompt_idx"]
        tr_c = np.where(pi >= args.holdout_prompts_lt)[0]
        va_c = np.where(pi < args.holdout_prompts_lt)[0]
        print(f"PROMPT split: tr={len(tr_c)} va={len(va_c)}")
    else:
        perm = rng.permutation(N)
        n_tr = int(N * (1 - args.val_frac))
        tr_c = perm[:n_tr]; va_c = perm[n_tr:]
        print(f"CYCLE split: tr={len(tr_c)} va={len(va_c)}")

    device = args.device
    Xtr = torch.from_numpy(feats[tr_c]).to(device)
    Ytr_pp = torch.from_numpy(Y_pp[tr_c]).to(device)
    Ytr_g = torch.from_numpy(Y_g[tr_c]).to(device)
    Xva = torch.from_numpy(feats[va_c]).to(device)
    yva_pp_np = Y_pp[va_c]
    yva_g_np = Y_g[va_c]
    yva_g = torch.from_numpy(yva_g_np).to(device)

    is_global = args.arch.startswith("global") or args.arch == "attn_global"
    if args.arch == "per_pos_linear":
        model = PerPosLinear(d_in)
    elif args.arch == "per_pos_mlp":
        model = PerPosMLP(d_in, hidden=args.hidden, dropout=args.dropout)
    elif args.arch == "global_linear":
        model = GlobalLinear(d_in, K)
    elif args.arch == "global_mlp":
        model = GlobalMLP(d_in, K, hidden=args.hidden, dropout=args.dropout)
    elif args.arch == "attn":
        model = TinyAttn(d_in, K, embed=args.hidden, n_heads=args.n_heads, n_layers=args.n_layers,
                          dropout=args.dropout, global_head=False)
    elif args.arch == "attn_global":
        model = TinyAttn(d_in, K, embed=args.hidden, n_heads=args.n_heads, n_layers=args.n_layers,
                          dropout=args.dropout, global_head=True)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"arch={args.arch} params={n_params}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2)
    t0 = time.time()
    best_score = -1.0; best_state = None
    n_tr = len(tr_c)

    pw = torch.tensor([args.pos_weight], device=device)
    for it in range(args.iters):
        model.train()
        idx = torch.randint(0, n_tr, (min(args.mb, n_tr),), device=device)
        xb = Xtr[idx]
        out = model(xb)
        if is_global:
            yb = Ytr_g[idx]
            loss = F.cross_entropy(out, yb)
        else:
            yb = Ytr_pp[idx]
            # weighted BCE
            w = torch.where(yb > 0.5, pw, torch.tensor([1.0], device=device))
            loss = F.binary_cross_entropy_with_logits(out, yb, weight=w)
        opt.zero_grad(); loss.backward(); opt.step()

        if it % args.report_every == 0 or it == args.iters - 1:
            model.eval()
            with torch.no_grad():
                out_va = model(Xva)
                if is_global:
                    pred_b = out_va.argmax(dim=1).cpu().numpy()
                    acc = (pred_b == yva_g_np).mean()
                    mae = np.abs(pred_b - yva_g_np).mean()
                    signed = (pred_b - yva_g_np).mean()
                    w1 = (np.abs(pred_b - yva_g_np) <= 1).mean()
                    w2 = (np.abs(pred_b - yva_g_np) <= 2).mean()
                    score = -mae  # smaller MAE → higher score
                    print(f"  it={it:>4d}  loss={loss.item():.4f}  acc={acc:.4f}  MAE_b={mae:.3f}  signed={signed:+.3f}  ±1={w1:.3f}  ±2={w2:.3f}  ({time.time()-t0:.1f}s)")
                else:
                    p_va = torch.sigmoid(out_va).cpu().numpy()  # (Nva, K)
                    auc_v = auc(p_va.reshape(-1), yva_pp_np.reshape(-1).astype(int))
                    pred_b = first_crossing(p_va, thr=0.3)
                    mae = np.abs(pred_b - yva_g_np).mean()
                    signed = (pred_b - yva_g_np).mean()
                    w1 = (np.abs(pred_b - yva_g_np) <= 1).mean()
                    w2 = (np.abs(pred_b - yva_g_np) <= 2).mean()
                    score = -mae
                    print(f"  it={it:>4d}  loss={loss.item():.4f}  AUC={auc_v:.4f}  MAE_b={mae:.3f}  signed={signed:+.3f}  ±1={w1:.3f}  ±2={w2:.3f}  ({time.time()-t0:.1f}s)")
            if score > best_score:
                best_score = score
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\nbest_score={best_score:.3f}")
    torch.save({"state_dict": model.state_dict(),
                "arch": args.arch,
                "d_in": d_in, "K": K,
                "hidden": args.hidden, "n_heads": args.n_heads, "n_layers": args.n_layers,
                "dropout": args.dropout,
                "mu": mu, "sd": sd, "with_aux": args.with_aux},
               args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
