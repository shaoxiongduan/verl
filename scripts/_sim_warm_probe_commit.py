"""Warm-restart with PROBE as the verifier (no verify forward).

Protocol — 1 forward per cycle:
  1. mixed_draft = K random tokens (warm_restart style)
  2. forward(committed + mixed_draft) → cur_argmax + hidden_states + logits
  3. probe(features) → predict boundary b (with optional offset bias)
  4. commit cur_argmax[0..b]  (committing probe's claimed-correct prefix)
  5. Loop

TPF = (committed_tokens) / (n_forwards).  No verify forward.

Counts ALL forwards. Compared against:
  - shift-leak oracle TPF_verify = 5.94 (math_k3) but that was per-verify-fwd in 2-fwd cycles
  - vLLM standard JF TPF = 3.31 (math_k3) / 3.83 (base) per forward
"""
from __future__ import annotations
import argparse, json, random
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompts_jsonl", required=True)
    p.add_argument("--out_jsonl", required=True)
    p.add_argument("--probe_npz", required=True)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--max_new", type=int, default=512)
    p.add_argument("--max_iters", type=int, default=256)
    p.add_argument("--vocab_size", type=int, default=152064)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--boundary_offset", type=int, default=0,
                   help="add to predicted boundary (positive=overcommit, negative=undercommit safer)")
    p.add_argument("--always_keep_pos0", action="store_true",
                   help="ensure at least 1 commit per iter (AR floor)")
    return p.parse_args()


def per_position_features_np(logits_K: torch.Tensor):
    fp32 = logits_K.float()
    probs = F.softmax(fp32, dim=-1)
    top5_vals, _ = probs.topk(5, dim=-1)
    top1 = top5_vals[:, 0].cpu().numpy()
    top5_sum = top5_vals.sum(dim=-1).cpu().numpy()
    entropy = -(probs * (probs.clamp_min(1e-12).log())).sum(dim=-1).cpu().numpy()
    top2_logits, _ = fp32.topk(2, dim=-1)
    margin = (top2_logits[:, 0] - top2_logits[:, 1]).cpu().numpy()
    return top1, entropy, margin, top5_sum


def load_probe(path):
    z = np.load(path, allow_pickle=True)
    w = {"mu": z["mu"].astype(np.float32), "sd": z["sd"].astype(np.float32),
         "with_aux": bool(z["with_aux"])}
    if "W1" in z.files:
        w["W1"] = z["W1"].astype(np.float32); w["b1"] = z["b1"].astype(np.float32)
        w["W2"] = z["W2"].astype(np.float32); w["b2"] = z["b2"].astype(np.float32)
        w["kind"] = "mlp"
    else:
        w["W"] = z["W"].astype(np.float32); w["b"] = float(z["b"])
        w["kind"] = "linear"
    return w


def predict_boundary(hidden_K, top1, ent, marg, t5s, K, probe):
    if probe["with_aux"]:
        aux = np.stack([top1, ent, marg, t5s, np.arange(K, dtype=np.float32) / K], axis=1).astype(np.float32)
        X = np.concatenate([hidden_K.astype(np.float32), aux], axis=1)
    else:
        X = hidden_K.astype(np.float32)
    Xs = (X - probe["mu"]) / probe["sd"]
    if probe["kind"] == "mlp":
        h = np.maximum(0, Xs @ probe["W1"] + probe["b1"])
        z = (h @ probe["W2"] + probe["b2"]).flatten()
    else:
        z = Xs @ probe["W"] + probe["b"]
    pc = 1.0 / (1.0 + np.exp(-z))
    return pc


@torch.no_grad()
def simulate(model, tok, prompt_ids, args, rng, probe):
    device = model.device
    K = args.K
    eos_id = 151645; pad_id = 151643
    stop_ids = {eos_id, pad_id}
    committed = list(prompt_ids)
    total_tok = 0
    n_forwards = 0
    pred_b_history = []

    while total_tok < args.max_new and n_forwards < args.max_iters:
        L = len(committed)
        mixed_draft = [rng.randrange(args.vocab_size) for _ in range(K)]
        inp = torch.tensor([committed + mixed_draft], dtype=torch.long, device=device)
        out = model(input_ids=inp, output_hidden_states=True)
        logits_K = out.logits[0, L - 1 : L - 1 + K, :]
        cur_argmax = logits_K.argmax(dim=-1).cpu().tolist()
        hidden_K = out.hidden_states[-1][0, L - 1 : L - 1 + K, :].float().cpu().numpy()
        n_forwards += 1

        top1, ent, marg, t5s = per_position_features_np(logits_K)
        pc = predict_boundary(hidden_K, top1, ent, marg, t5s, K, probe)

        # boundary = first j where P < threshold
        below = np.where(pc < args.threshold)[0]
        b = int(below[0]) if len(below) else K
        b = max(0, min(K, b + args.boundary_offset))
        if args.always_keep_pos0:
            b = max(1, b)
        pred_b_history.append(b)

        # commit cur_argmax[0..b]
        for j in range(b):
            committed.append(int(cur_argmax[j]))
        total_tok += b
        if b > 0 and committed[-1] in stop_ids:
            break

    return {
        "n_tokens": total_tok,
        "n_forwards": n_forwards,
        "tpf": total_tok / max(1, n_forwards),
        "pred_b_history": pred_b_history,
        "mean_pred_b": float(np.mean(pred_b_history)) if pred_b_history else 0.0,
    }


def main():
    args = parse_args()
    print(f"[wpc] loading {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    prompts = [json.loads(l) for l in open(args.prompts_jsonl)]
    rng = random.Random(args.seed)
    probe = load_probe(args.probe_npz)
    print(f"[wpc] probe kind={probe['kind']}  with_aux={probe['with_aux']}  thr={args.threshold}  offset={args.boundary_offset}  always_keep_pos0={args.always_keep_pos0}", flush=True)

    out_fp = open(args.out_jsonl, "w")
    rows = []
    for i, p in enumerate(prompts):
        chat = [{"role": "user", "content": p["input"]}]
        text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        pid = tok(text, return_tensors="pt").input_ids[0].tolist()
        r = simulate(model, tok, pid, args, rng, probe)
        r["batch_idx"] = i
        rows.append(r)
        out_fp.write(json.dumps(r) + "\n")
        out_fp.flush()
        print(f"[wpc] [{i+1}/{len(prompts)}] ntok={r['n_tokens']:4d} fwd={r['n_forwards']:4d} TPF={r['tpf']:.3f} mean_pred_b={r['mean_pred_b']:.2f}", flush=True)

    total_tok = sum(r["n_tokens"] for r in rows)
    total_fwd = sum(r["n_forwards"] for r in rows)
    print(f"\n[wpc] CORPUS: Σtok={total_tok}  Σfwd={total_fwd}  TPF={total_tok/max(1,total_fwd):.4f}", flush=True)
    print(f"[wpc] per-prompt-mean TPF = {sum(r['tpf'] for r in rows)/len(rows):.4f}", flush=True)


if __name__ == "__main__":
    main()
