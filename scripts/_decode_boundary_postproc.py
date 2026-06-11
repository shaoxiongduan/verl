"""Post-hoc decoders for per-position probe outputs.

Given a trained per-position probe (linear or MLP), evaluates several decoding
strategies on the val split and reports boundary-MAE:

  - first:    first j with P[j] < threshold (current method)
  - last:     last j with P[j] >= threshold (then b = last + 1)
  - mle_changepoint: argmax_b sum_{j<b} log P[j] + sum_{j>=b} log(1-P[j])
  - expected: round(sum_j P[j])  (expected length under independence)
  - argmax_p_pos: argmax_j P[j] * (some weighting; default just count >0.5)

Loads either:
  - a linear/MLP probe (.npz with W,b,mu,sd,with_aux  OR W1,b1,W2,b2,...)
  - or a torch .pt saved by _train_boundary_arch.py
"""
from __future__ import annotations
import argparse
import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--probe", required=True, help=".npz or .pt probe file")
    p.add_argument("--holdout_prompts_lt", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--thresholds", default="0.2,0.3,0.4,0.5,0.6")
    return p.parse_args()


def load_probe_compute_P(z_data, probe_path):
    """Return P matrix (N, K) for all cycles."""
    H = z_data["hidden"].astype(np.float32)
    N, K, d = H.shape
    aux = np.stack([z_data["top1_prob"], z_data["entropy"], z_data["margin_log"], z_data["top5_prob_sum"]], axis=-1).astype(np.float32)
    pos = np.tile(np.arange(K, dtype=np.float32) / K, (N, 1))[..., None]
    feats3 = np.concatenate([H, aux, pos], axis=-1)  # (N, K, d+5)
    d_in = feats3.shape[-1]

    if probe_path.endswith(".npz"):
        pz = np.load(probe_path, allow_pickle=True)
        mu = pz["mu"].astype(np.float32); sd = pz["sd"].astype(np.float32)
        with_aux = bool(pz["with_aux"])
        if not with_aux:
            feats3 = H
            d_in = d
        feats2 = feats3.reshape(N * K, d_in)
        Xs = (feats2 - mu) / sd
        if "W1" in pz.files:
            # MLP
            W1 = pz["W1"].astype(np.float32); b1 = pz["b1"].astype(np.float32)
            W2 = pz["W2"].astype(np.float32); b2 = pz["b2"].astype(np.float32)
            h = np.maximum(0, Xs @ W1 + b1)
            z = (h @ W2 + b2).flatten()
        else:
            W = pz["W"].astype(np.float32); b = float(pz["b"])
            z = Xs @ W + b
        P = (1 / (1 + np.exp(-z))).reshape(N, K)
        return P
    else:
        # PyTorch .pt
        ckpt = torch.load(probe_path, map_location="cpu", weights_only=False)
        sd_dict = ckpt["state_dict"]
        mu = ckpt["mu"].astype(np.float32); sd = ckpt["sd"].astype(np.float32)
        # Standardize
        feats3 = (feats3 - mu) / sd
        from scripts._train_boundary_arch import (PerPosLinear, PerPosMLP, GlobalLinear, GlobalMLP, TinyAttn)
        arch = ckpt["arch"]
        if arch == "per_pos_linear":
            model = PerPosLinear(d_in)
        elif arch == "per_pos_mlp":
            model = PerPosMLP(d_in, hidden=ckpt["hidden"], dropout=ckpt["dropout"])
        elif arch == "attn":
            model = TinyAttn(d_in, K, embed=ckpt["hidden"], n_heads=ckpt["n_heads"],
                              n_layers=ckpt["n_layers"], dropout=ckpt["dropout"], global_head=False)
        else:
            raise ValueError(f"unsupported per-position arch in pt: {arch}")
        model.load_state_dict(sd_dict)
        model.eval()
        with torch.no_grad():
            x = torch.from_numpy(feats3.astype(np.float32))
            out = model(x)
            P = torch.sigmoid(out).numpy()
        return P


def decode_first(P, thr):
    K = P.shape[1]; out = []
    for r in P:
        below = np.where(r < thr)[0]
        out.append(int(below[0]) if len(below) else K)
    return np.array(out)


def decode_last(P, thr):
    """Last j with P >= thr, then b = last+1."""
    K = P.shape[1]; out = []
    for r in P:
        above = np.where(r >= thr)[0]
        out.append(int(above[-1] + 1) if len(above) else 0)
    return np.array(out)


def decode_mle_changepoint(P):
    """Argmax_b [sum_{j<b} log P[j] + sum_{j>=b} log(1-P[j])]."""
    eps = 1e-6
    K = P.shape[1]
    out = []
    for r in P:
        logp = np.log(np.clip(r, eps, 1-eps))
        log1mp = np.log(np.clip(1 - r, eps, 1-eps))
        # cumulative sums
        cum_logp = np.concatenate([[0], np.cumsum(logp)])
        suffix_log1mp = np.concatenate([np.cumsum(log1mp[::-1])[::-1], [0]])
        # For b in [0, K]:  ll(b) = cum_logp[b] + suffix_log1mp[b]
        ll = cum_logp + suffix_log1mp
        out.append(int(np.argmax(ll)))
    return np.array(out)


def decode_expected(P):
    return np.clip(np.round(P.sum(axis=1)), 0, P.shape[1]).astype(int)


def report(name, pred_b, true_b):
    err = pred_b - true_b
    mae = np.abs(err).mean()
    signed = err.mean()
    w0 = (err == 0).mean()
    w1 = (np.abs(err) <= 1).mean()
    w2 = (np.abs(err) <= 2).mean()
    w3 = (np.abs(err) <= 3).mean()
    print(f"  {name:<30}  MAE={mae:.3f}  signed={signed:+.3f}  exact={w0:.3f}  ±1={w1:.3f}  ±2={w2:.3f}  ±3={w3:.3f}")


def main():
    args = parse_args()
    print(f"loading {args.data}")
    z = np.load(args.data)
    nacc = z["n_acc"]
    print(f"loading probe {args.probe}")
    P = load_probe_compute_P(z, args.probe)

    pi = z["prompt_idx"]
    va = np.where(pi < args.holdout_prompts_lt)[0]
    P_va = P[va]; nacc_va = nacc[va].astype(int)
    print(f"val cycles = {len(va)}; true mean n_acc = {nacc_va.mean():.3f}")

    for thr in [float(x) for x in args.thresholds.split(",")]:
        pred = decode_first(P_va, thr)
        report(f"first(thr={thr})", pred, nacc_va)

    for thr in [0.3, 0.5]:
        pred = decode_last(P_va, thr)
        report(f"last(thr={thr})", pred, nacc_va)

    pred = decode_mle_changepoint(P_va)
    report("mle_changepoint", pred, nacc_va)

    pred = decode_expected(P_va)
    report("expected(sum P, round)", pred, nacc_va)


if __name__ == "__main__":
    main()
