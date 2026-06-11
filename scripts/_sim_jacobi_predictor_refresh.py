"""Standard Jacobi block decode with predictor-refresh between iters.

Protocol — 1 forward per iter, NO verify forward:
  while not done:
      logits = forward([committed | draft])          # 1 forward
      cur_argmax = argmax(logits)
      # Jacobi acceptance: prefix where cur_argmax == draft (= this iter's input)
      n_acc = longest j s.t. cur_argmax[0..j-1] == draft[0..j-1]
      commit cur_argmax[0..n_acc-1]                  # standard JF block-decode
      # Shift K-window by n_acc; new positions [K-n_acc..K) start fresh random
      shifted = cur_argmax[n_acc..K) + [random]*n_acc

      # PREDICTOR REFRESH: clean shifted draft using logit features at src=j+n_acc
      p_correct[j] = sigmoid(w · standardized features at original position j+n_acc)
      new_draft[j] = shifted[j]   if p_correct[j] > threshold   else random

      draft = new_draft

TPF = total_committed_tokens / n_forwards.  No second forward per cycle.

Compare against:
  - none mode: don't refresh, always pass shifted draft → standard JF block-decode
  - oracle mode: use TRUE convergence-check derived boundary → upper bound
"""
from __future__ import annotations
import argparse, json, math, random
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


LOGREG5_WEIGHTS = {
    "math_k3": {
        "w": [1.0651507, 0.54727219, 0.86257177, -0.12074035, 0.41574884],
        "b": 0.20852359,
    },
    "base": {
        "w": [0.60511182, -0.16665991, 0.63486387, 0.00439597, 0.50162427],
        "b": -0.23659919,
    },
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--model_key", default=None, choices=["math_k3", "base"],
                   help="Required only for logreg5/logit_p refresh (selects LOGREG5_WEIGHTS).")
    p.add_argument("--prompts_jsonl", required=True)
    p.add_argument("--feat_jsonl", default=None,
                   help="Required only for logreg5/logit_p refresh (feature standardization).")
    p.add_argument("--out_jsonl", required=True)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--max_new", type=int, default=512)
    p.add_argument("--max_iters", type=int, default=512)
    p.add_argument("--vocab_size", type=int, default=152064)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--refresh", choices=["none", "logit_p", "logreg5", "fixed", "hidden_probe", "global_boundary"], default="logreg5",
                   help="Refresh policy. 'fixed' uses --fixed_i. 'hidden_probe' uses --probe_npz per-position. 'global_boundary' uses --probe_npz (K+1)-way classifier.")
    p.add_argument("--fixed_i", type=int, default=None,
                   help="For --refresh fixed: keep shifted[0..fixed_i), reinit shifted[fixed_i..] to random.")
    p.add_argument("--probe_npz", type=str, default=None,
                   help="For --refresh hidden_probe: path to trained probe .npz (W, b, mu, sd).")
    p.add_argument("--leak", choices=["boundary", "per_position"], default="boundary")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--always_keep_pos0", action="store_true",
                   help="Force keep position 0 of the shifted window regardless of predictor (AR floor guard)")
    p.add_argument("--boundary_offset", type=int, default=0,
                   help="Add this constant to the predicted boundary (positive = bias overreport).")
    p.add_argument("--refresh_to", choices=["random", "mask"], default="random",
                   help="Token used for reinit (initial draft, shifted-in tail, refreshed positions). "
                        "'mask' = constant --mask_id token; for mask-tail-trained drafters (idea A).")
    p.add_argument("--mask_id", type=int, default=151643,
                   help="Mask token id for --refresh_to mask (default Qwen pad <|endoftext|>).")
    p.add_argument("--mask_stall_limit", type=int, default=8,
                   help="In mask mode: stop after this many consecutive zero-commit iters with "
                        "argmax[0]==mask (model declares even the next token undecided).")
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


def predict(args, top1, ent, marg, t5s, K, weights, mu, sd):
    if args.refresh == "logit_p":
        return top1.copy()
    pos = np.arange(K, dtype=np.float64) / K
    X = np.stack([top1, ent, marg, t5s, pos], axis=1)
    Xs = (X - mu) / sd
    z = Xs @ np.array(weights["w"]) + weights["b"]
    return 1.0 / (1.0 + np.exp(-z))


def keep_mask(p_correct, mode, thr):
    K = len(p_correct)
    if mode == "per_position":
        return p_correct > thr
    below = np.where(p_correct < thr)[0]
    b = int(below[0]) if len(below) else K
    m = np.zeros(K, dtype=bool); m[:b] = True
    return m


@torch.no_grad()
def simulate(model, tok, prompt_ids, args, rng, weights, mu, sd, probe_weights=None):
    device = model.device
    K = args.K
    eos_id = 151645; pad_id = 151643
    mask_mode = args.refresh_to == "mask"
    mask_id = args.mask_id
    # In mask mode the mask token (= pad) is "undecided", never a stop signal;
    # it is also never committed (see acceptance below), so drop it from stops.
    stop_ids = {eos_id} if mask_mode else {eos_id, pad_id}

    def fresh():
        return mask_id if mask_mode else rng.randrange(args.vocab_size)

    committed = list(prompt_ids)
    draft = [fresh() for _ in range(K)]
    n_forwards = 0
    total_tok = 0
    n_acc_history = []
    pred_b_history = []
    iters_per_commit_run = 0
    iters_no_commit = []
    mask_frac_history = []
    first_mask_pos_history = []
    mask_stall_run = 0
    stopped_by_mask_stall = False

    while total_tok < args.max_new and n_forwards < args.max_iters:
        L = len(committed)
        inp = torch.tensor([committed + draft], dtype=torch.long, device=device)
        # Need hidden states for probe refresh mode
        need_hidden = (args.refresh == "hidden_probe")
        out = model(input_ids=inp, output_hidden_states=need_hidden)
        logits_K = out.logits[0, L - 1 : L - 1 + K, :]
        cur_argmax = logits_K.argmax(dim=-1).cpu().tolist()
        hidden_K = None
        if need_hidden:
            hidden_K = out.hidden_states[-1][0, L - 1 : L - 1 + K, :].float().cpu().numpy()
        n_forwards += 1
        iters_per_commit_run += 1

        # Track where the model itself emits mask (drafter's declared boundary)
        if mask_id in cur_argmax:
            first_mask = cur_argmax.index(mask_id)
        else:
            first_mask = K
        mask_frac_history.append(sum(1 for t in cur_argmax if t == mask_id) / K)
        first_mask_pos_history.append(first_mask)

        # Jacobi convergence: cur_argmax matches draft (this iter's input) on prefix.
        # Mask tokens are "undecided" — a mask==mask match is convergence-on-undecided
        # and must never be committed; stop the scan there.
        n_acc = 0
        for j in range(K):
            if cur_argmax[j] == draft[j] and not (mask_mode and cur_argmax[j] == mask_id):
                n_acc += 1
            else:
                break

        # Stall guard: model insists even the NEXT token is mask → nothing will
        # ever commit. After mask_stall_limit consecutive such iters, stop.
        if mask_mode:
            if n_acc == 0 and cur_argmax[0] == mask_id:
                mask_stall_run += 1
                if mask_stall_run >= args.mask_stall_limit:
                    stopped_by_mask_stall = True
                    break
            else:
                mask_stall_run = 0

        if n_acc > 0:
            # Truncate the commit at the FIRST stop token: a stop accepted
            # mid-block must end generation there, not run past it.
            hit_stop = False
            n_commit = n_acc
            for j in range(n_acc):
                if int(cur_argmax[j]) in stop_ids:
                    n_commit = j + 1
                    hit_stop = True
                    break
            for j in range(n_commit):
                committed.append(int(cur_argmax[j]))
            total_tok += n_commit
            n_acc_history.append(n_commit)
            iters_no_commit.append(iters_per_commit_run)
            iters_per_commit_run = 0
            if hit_stop:
                break
            if total_tok >= args.max_new:
                break

        # Build shifted base for next iter: positions [n_acc, K) of cur_argmax → [0, K-n_acc)
        shifted = list(cur_argmax[n_acc:K])
        # Compute predictor over original positions [n_acc, K)
        top1, ent, marg, t5s = per_position_features_np(logits_K)
        # Slice feature arrays to [n_acc, K)
        n_keep = K - n_acc
        if args.refresh == "none":
            keep = np.ones(n_keep, dtype=bool)
        elif args.refresh == "fixed":
            fi = max(0, min(args.fixed_i, n_keep))
            keep = np.zeros(n_keep, dtype=bool); keep[:fi] = True
        elif args.refresh == "hidden_probe":
            # Per-position feature: [hidden(d), top1_p, entropy, margin_log, top5_prob_sum, pos/K]
            h_slice = hidden_K[n_acc:K]  # (n_keep, d) fp32
            _t1, _et, _mg, _t5 = per_position_features_np(logits_K)
            aux = np.stack([
                _t1[n_acc:K], _et[n_acc:K], _mg[n_acc:K], _t5[n_acc:K],
                np.arange(n_acc, K, dtype=np.float64) / K
            ], axis=1).astype(np.float32)
            if probe_weights["with_aux"]:
                X = np.concatenate([h_slice, aux], axis=1).astype(np.float32)
            else:
                X = h_slice.astype(np.float32)
            Xs = (X - probe_weights["mu"]) / probe_weights["sd"]
            if "W1" in probe_weights:  # MLP
                h_pre = Xs @ probe_weights["W1"] + probe_weights["b1"]
                h_act = np.maximum(0, h_pre)
                z = (h_act @ probe_weights["W2"] + probe_weights["b2"]).flatten()
            else:
                z = Xs @ probe_weights["W"] + probe_weights["b"]
            pc = 1.0 / (1.0 + np.exp(-z))
            keep = keep_mask(pc, args.leak, args.threshold)
        elif args.refresh == "global_boundary":
            # Use whole K-block hidden states (NOT just shifted) since predictor was trained on
            # original cycle's K-block. Predicts true n_acc directly via softmax(K+1) argmax.
            _t1, _et, _mg, _t5 = per_position_features_np(logits_K)
            X = hidden_K.reshape(K * hidden_K.shape[1]).astype(np.float32)  # flatten K hidden states
            if probe_weights["with_aux"]:
                pos_idx = (np.arange(K, dtype=np.float32) / K).reshape(-1, 1)
                aux_block = np.stack([_t1, _et, _mg, _t5], axis=-1).astype(np.float32)  # (K, 4)
                aux_block = np.concatenate([aux_block, pos_idx], axis=1)  # (K, 5)
                X = np.concatenate([X, aux_block.reshape(-1)])  # (K*d + K*5,)
            Xs = (X - probe_weights["mu"]) / probe_weights["sd"]
            logits_b = Xs @ probe_weights["W"] + probe_weights["b"]  # (K+1,)
            pred_b = int(np.argmax(logits_b))
            # Translate predicted true n_acc to "keep how many of the shifted window"
            # shifted_window starts at original position n_acc; predicted boundary is at original position pred_b
            # In the shifted window, the position corresponding to original boundary pred_b is (pred_b - n_acc)
            keep_count = max(0, pred_b - n_acc)
            keep = np.zeros(n_keep, dtype=bool); keep[:keep_count] = True
        else:
            t1s = top1[n_acc:K]; ets = ent[n_acc:K]; mgs = marg[n_acc:K]; tss = t5s[n_acc:K]
            pos_orig = np.arange(n_acc, K, dtype=np.float64) / K
            if args.refresh == "logit_p":
                pc = t1s.copy()
            else:
                Xs = (np.stack([t1s, ets, mgs, tss, pos_orig], axis=1) - mu) / sd
                z = Xs @ np.array(weights["w"]) + weights["b"]
                pc = 1.0 / (1.0 + np.exp(-z))
            keep = keep_mask(pc, args.leak, args.threshold)

        # Post-process keep mask for ALL refresh modes that produced a mask
        if args.refresh != "none" and len(keep) > 0:
            if args.boundary_offset != 0 and args.leak == "boundary":
                current_b = int(keep.sum())
                new_b = max(0, min(n_keep, current_b + args.boundary_offset))
                keep = np.zeros(n_keep, dtype=bool); keep[:new_b] = True
            if args.always_keep_pos0:
                keep[0] = True  # AR floor

        # Apply mask: keep shifted[j] if keep[j], else reinit (random or mask token)
        new_draft = [0] * K
        for j in range(n_keep):
            if keep[j]:
                new_draft[j] = shifted[j]
            else:
                new_draft[j] = fresh()
        for j in range(n_keep, K):
            new_draft[j] = fresh()

        # Record predicted boundary (= count of True before first False)
        if args.refresh == "none":
            pred_b_history.append(n_keep)
        else:
            below_idx = np.where(~keep)[0]
            pred_b = int(below_idx[0]) if len(below_idx) else n_keep
            pred_b_history.append(pred_b)

        draft = new_draft

    return {
        "n_tokens": total_tok,
        "n_forwards": n_forwards,
        "tpf": total_tok / max(1, n_forwards),
        "n_commits": len(n_acc_history),
        "mean_n_acc_per_commit": (sum(n_acc_history) / max(1, len(n_acc_history))),
        "mean_iters_between_commit": (sum(iters_no_commit) / max(1, len(iters_no_commit))),
        "n_acc_history": n_acc_history,
        "pred_b_history": pred_b_history,
        "refresh": args.refresh,
        "leak_mode": args.leak,
        "threshold": args.threshold,
        "refresh_to": args.refresh_to,
        "mask_frac_history": mask_frac_history,
        "first_mask_pos_history": first_mask_pos_history,
        "stopped_by_mask_stall": stopped_by_mask_stall,
    }


def main():
    args = parse_args()
    print(f"[jsim] loading {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    prompts = [json.loads(l) for l in open(args.prompts_jsonl)]
    rng = random.Random(args.seed)

    need_logreg = args.refresh in ("logit_p", "logreg5")
    weights = mu = sd = None
    if need_logreg:
        assert args.model_key, "--model_key required for logit_p/logreg5 refresh"
        assert args.feat_jsonl, "--feat_jsonl required for logit_p/logreg5 refresh"
        weights = LOGREG5_WEIGHTS[args.model_key]
        print(f"[jsim] computing feature mu/sd from {args.feat_jsonl}", flush=True)
        mu, sd = compute_mu_sd(args.feat_jsonl)
    print(f"[jsim] refresh={args.refresh}  refresh_to={args.refresh_to}  leak={args.leak}  thr={args.threshold}  K={args.K}  max_new={args.max_new}", flush=True)

    probe_weights = None
    if args.refresh in ("hidden_probe", "global_boundary"):
        assert args.probe_npz, "--probe_npz required for hidden-state predictor"
        z = np.load(args.probe_npz, allow_pickle=True)
        if args.refresh == "hidden_probe":
            probe_weights = {
                "mu": z["mu"].astype(np.float32),
                "sd": z["sd"].astype(np.float32),
                "with_aux": bool(z["with_aux"]),
                "mode": "per_position",
            }
            if "W1" in z.files:  # MLP
                probe_weights["W1"] = z["W1"].astype(np.float32)
                probe_weights["b1"] = z["b1"].astype(np.float32)
                probe_weights["W2"] = z["W2"].astype(np.float32)
                probe_weights["b2"] = z["b2"].astype(np.float32)
            else:  # linear
                probe_weights["W"] = z["W"].astype(np.float32)
                probe_weights["b"] = float(z["b"])
        else:
            # global_boundary: W is (d_feat, K+1)
            probe_weights = {
                "W": z["W"].astype(np.float32),
                "b": z["b"].astype(np.float32),
                "mu": z["mu"].astype(np.float32),
                "sd": z["sd"].astype(np.float32),
                "with_aux": bool(z["with_aux"]),
                "mode": "global_boundary",
                "K": int(z["K"]) if "K" in z.files else args.K,
            }
        kind = "MLP" if "W1" in probe_weights else "linear"
        shape = probe_weights.get("W1", probe_weights.get("W")).shape
        print(f"[jsim] loaded probe {args.probe_npz}  kind={kind}  shape={shape}  mode={probe_weights['mode']}  with_aux={probe_weights['with_aux']}", flush=True)

    out_fp = open(args.out_jsonl, "w")
    rows = []
    for i, p in enumerate(prompts):
        chat = [{"role": "user", "content": p["input"]}]
        text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        pid = tok(text, return_tensors="pt").input_ids[0].tolist()
        r = simulate(model, tok, pid, args, rng, weights, mu, sd, probe_weights=probe_weights)
        r["batch_idx"] = i
        rows.append(r)
        out_fp.write(json.dumps(r) + "\n")
        out_fp.flush()
        print(f"[jsim] [{i+1}/{len(prompts)}] ntok={r['n_tokens']:4d} fwd={r['n_forwards']:4d} TPF={r['tpf']:.3f} "
              f"commits={r['n_commits']:3d} iters/commit={r['mean_iters_between_commit']:.2f} mean_nacc={r['mean_n_acc_per_commit']:.2f}",
              flush=True)

    total_tok = sum(r["n_tokens"] for r in rows)
    total_fwd = sum(r["n_forwards"] for r in rows)
    print(f"\n[jsim] CORPUS: Σtok={total_tok}  Σfwd={total_fwd}  TPF={total_tok/max(1,total_fwd):.4f}", flush=True)
    print(f"[jsim] per-prompt-mean TPF = {sum(r['tpf'] for r in rows)/len(rows):.4f}", flush=True)


if __name__ == "__main__":
    main()
