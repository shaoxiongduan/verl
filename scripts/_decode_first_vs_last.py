"""Apply first-crossing and last-crossing decoders to a trained probe + dataset.

Reports under-pred / over-pred / MAE on the prompt-holdout val cycles for each
decoder.

first-crossing rule  : b = first j where P[j] < threshold (current, conservative)
last-crossing rule   : b = 1 + last j where P[j] >= threshold (lenient)
"""
from __future__ import annotations
import argparse
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--probe", required=True)
    p.add_argument("--holdout_prompts_lt", type=int, default=16)
    p.add_argument("--thresholds", default="0.2,0.3,0.4,0.5,0.6")
    return p.parse_args()


def compute_P(z_data, probe_path):
    H = z_data["hidden"].astype(np.float32)
    N, K, d = H.shape
    aux = np.stack([z_data["top1_prob"], z_data["entropy"], z_data["margin_log"], z_data["top5_prob_sum"]], axis=-1).astype(np.float32)
    pos = np.tile(np.arange(K, dtype=np.float32) / K, (N, 1))[..., None]
    feats3 = np.concatenate([H, aux, pos], axis=-1)
    d_in = feats3.shape[-1]
    pz = np.load(probe_path, allow_pickle=True)
    mu = pz["mu"].astype(np.float32); sd = pz["sd"].astype(np.float32)
    with_aux = bool(pz["with_aux"])
    if not with_aux:
        feats3 = H; d_in = d
    feats2 = feats3.reshape(N * K, d_in)
    Xs = (feats2 - mu) / sd
    if "W1" in pz.files:
        W1 = pz["W1"].astype(np.float32); b1 = pz["b1"].astype(np.float32)
        W2 = pz["W2"].astype(np.float32); b2 = pz["b2"].astype(np.float32)
        h = np.maximum(0, Xs @ W1 + b1)
        z = (h @ W2 + b2).flatten()
    else:
        W = pz["W"].astype(np.float32); b = float(pz["b"])
        z = Xs @ W + b
    P = (1 / (1 + np.exp(-z))).reshape(N, K)
    return P


def decode_first(P, thr):
    K = P.shape[1]; out = []
    for r in P:
        below = np.where(r < thr)[0]
        out.append(int(below[0]) if len(below) else K)
    return np.array(out)


def decode_last(P, thr):
    """b = 1 + last index j where P[j] >= thr. If none, b=0."""
    K = P.shape[1]; out = []
    for r in P:
        above = np.where(r >= thr)[0]
        out.append(int(above[-1] + 1) if len(above) else 0)
    return np.array(out)


def report(name, pred_b, true_b):
    err = pred_b - true_b
    under = (pred_b < true_b)
    n_under = under.sum()
    n = len(pred_b)
    shortfall_under = (true_b - pred_b)[under]
    catastrophic = under & (pred_b <= true_b / 2)
    print(f"  {name}")
    print(f"    P(under)={n_under/n:.3f}  P(match)={(pred_b==true_b).mean():.3f}  P(over)={(pred_b>true_b).mean():.3f}")
    print(f"    MAE={np.abs(err).mean():.3f}  signed={err.mean():+.3f}  std={err.std():.3f}")
    print(f"    catastrophic_under (pred<=true/2) = {catastrophic.sum()}  ({catastrophic.sum()/n*100:.2f}%)")
    if n_under > 0:
        print(f"    among under-preds: shortfall mean={shortfall_under.mean():.2f}, median={int(np.median(shortfall_under))}")
        print(f"      shortfall hist: =1: {(shortfall_under==1).sum()}  =2: {(shortfall_under==2).sum()}  =3: {(shortfall_under==3).sum()}  =4: {(shortfall_under==4).sum()}  5+: {(shortfall_under>=5).sum()}")


def main():
    args = parse_args()
    z = np.load(args.data)
    P = compute_P(z, args.probe)
    pi = z["prompt_idx"]
    va = np.where(pi < args.holdout_prompts_lt)[0]
    P_va = P[va]; nacc_va = z["n_acc"][va].astype(int)
    print(f"val cycles = {len(va)}, true mean n_acc = {nacc_va.mean():.3f}")
    for thr in [float(x) for x in args.thresholds.split(",")]:
        print(f"\n=== threshold {thr} ===")
        pred_first = decode_first(P_va, thr)
        pred_last = decode_last(P_va, thr)
        report(f"first-crossing(thr={thr})", pred_first, nacc_va)
        report(f"last-crossing(thr={thr})", pred_last, nacc_va)
        # Also try last + offset
        K = P_va.shape[1]
        for off in [1, 2, 3, 5]:
            pred_last_off = np.minimum(pred_last + off, K)
            report(f"last+offset={off}(thr={thr})", pred_last_off, nacc_va)


if __name__ == "__main__":
    main()
