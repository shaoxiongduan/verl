"""Warm-restart sim (2-forward per cycle) with MLP-probe-driven leakage.

Per cycle:
  1. mixed_draft = leak_part + (K-len(leak)) random
  2. warm forward(committed + mixed_draft) → warm_argmax + hidden_states
  3. verify forward(committed + warm_argmax) → verify_argmax
  4. n_acc = standard spec-decode prefix match (warm vs verify), commit n_acc+1
  5. Apply probe to (hidden, logits) → predicted boundary pred_b
  6. next cycle's leak_part = warm[pred_b..pred_b+leak_amount] (those positions probe-rejected)
     Actually simpler: predicted-correct positions stay (we don't leak them — they go into
     the next cycle as the front of the new draft, anchored to new K-window position 0).

Logs per cycle:
  - true_n_acc (from verify), pred_b (from probe), err = pred_b - n_acc
  - tokens committed = n_acc + 1
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
    p.add_argument("--max_cycles", type=int, default=128)
    p.add_argument("--vocab_size", type=int, default=152064)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--boundary_offset", type=int, default=0)
    p.add_argument("--boundary_rule", choices=["first", "last"], default="first")
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


def predict_pc(hidden_K, top1, ent, marg, t5s, K, probe):
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
    return 1.0 / (1.0 + np.exp(-z))


@torch.no_grad()
def simulate(model, tok, prompt_ids, args, rng, probe):
    device = model.device
    K = args.K
    eos_id = 151645; pad_id = 151643
    stop_ids = {eos_id, pad_id}
    committed = list(prompt_ids)
    total_tok = 0
    n_cycles = 0
    pred_b_hist = []
    true_b_hist = []
    leak_part = []

    while total_tok < args.max_new and n_cycles < args.max_cycles:
        L = len(committed)
        # Build mixed_draft = leak_part + random
        leak_len = len(leak_part)
        mixed_draft = list(leak_part) + [rng.randrange(args.vocab_size) for _ in range(K - leak_len)]

        # warm forward
        inp1 = torch.tensor([committed + mixed_draft], dtype=torch.long, device=device)
        out1 = model(input_ids=inp1, output_hidden_states=True)
        warm_logits = out1.logits[0, L - 1 : L - 1 + K, :]
        warm_argmax = warm_logits.argmax(dim=-1).cpu().tolist()
        hidden_K = out1.hidden_states[-1][0, L - 1 : L - 1 + K, :].float().cpu().numpy()

        # verify forward
        inp2 = torch.tensor([committed + warm_argmax], dtype=torch.long, device=device)
        verify_argmax = model(input_ids=inp2).logits[0, L - 1 : L - 1 + K, :].argmax(dim=-1).cpu().tolist()

        # standard spec-decode acceptance
        n_acc = 0
        for j in range(K):
            if warm_argmax[j] == verify_argmax[j]:
                n_acc += 1
            else:
                break
        for j in range(n_acc):
            committed.append(int(warm_argmax[j]))
        if n_acc < K:
            committed.append(int(verify_argmax[n_acc]))
            total_tok += n_acc + 1
        else:
            total_tok += K
        n_cycles += 1
        true_b_hist.append(n_acc)

        # Apply probe to warm output → predicted boundary
        top1, ent, marg, t5s = per_position_features_np(warm_logits)
        pc = predict_pc(hidden_K, top1, ent, marg, t5s, K, probe)
        if args.boundary_rule == "first":
            below = np.where(pc < args.threshold)[0]
            pred_b = int(below[0]) if len(below) else K
        else:
            above = np.where(pc >= args.threshold)[0]
            pred_b = int(above[-1] + 1) if len(above) else 0
        pred_b = max(0, min(K, pred_b + args.boundary_offset))
        pred_b_hist.append(pred_b)

        if committed[-1] in stop_ids:
            break

        # leak_part for next cycle = warm[pred_b : pred_b + shift_amount]
        # i.e. positions predictor THOUGHT were correct but were beyond commit
        # actually simpler: leak nothing if pred_b <= n_acc+1 (no overcommit)
        # leak warm[n_acc+1 : pred_b] as wrong-leaked into next cycle
        if pred_b > n_acc + 1 and (n_acc + 1) < K:
            leak_part = list(warm_argmax[n_acc + 1 : min(pred_b, K)])
        else:
            leak_part = []

    return {
        "n_tokens": total_tok,
        "n_cycles": n_cycles,
        "tpf_verify": total_tok / max(1, n_cycles),
        "tpf_all": total_tok / max(1, 2 * n_cycles),
        "pred_b_history": pred_b_hist,
        "true_b_history": true_b_hist,
    }


def main():
    args = parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    prompts = [json.loads(l) for l in open(args.prompts_jsonl)]
    rng = random.Random(args.seed)
    probe = load_probe(args.probe_npz)
    print(f"[wmp] probe={probe['kind']} thr={args.threshold} offset={args.boundary_offset}", flush=True)

    out_fp = open(args.out_jsonl, "w")
    rows = []
    all_errs = []
    for i, p in enumerate(prompts):
        chat = [{"role": "user", "content": p["input"]}]
        text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        pid = tok(text, return_tensors="pt").input_ids[0].tolist()
        r = simulate(model, tok, pid, args, rng, probe)
        r["batch_idx"] = i
        rows.append(r)
        out_fp.write(json.dumps(r) + "\n")
        out_fp.flush()
        errs = np.array([pb - tb for pb, tb in zip(r["pred_b_history"], r["true_b_history"])])
        all_errs.extend(errs.tolist())
        print(f"[wmp] [{i+1}/{len(prompts)}] ntok={r['n_tokens']:4d} cyc={r['n_cycles']:4d} "
              f"TPF_v={r['tpf_verify']:.3f} TPF_a={r['tpf_all']:.3f} mean_pred_b={np.mean(r['pred_b_history']):.2f} mean_true_b={np.mean(r['true_b_history']):.2f} mean_err={errs.mean():+.2f}",
              flush=True)

    total_tok = sum(r["n_tokens"] for r in rows)
    total_cyc = sum(r["n_cycles"] for r in rows)
    print(f"\n[wmp] CORPUS: Σtok={total_tok}  Σcyc={total_cyc}  TPF_verify={total_tok/max(1,total_cyc):.4f}  TPF_all={total_tok/max(1,2*total_cyc):.4f}", flush=True)
    print(f"[wmp] per-prompt-mean TPF_verify = {sum(r['tpf_verify'] for r in rows)/len(rows):.4f}", flush=True)

    # Error histogram
    arr = np.array(all_errs)
    print(f"\n[wmp] err distribution over {len(arr)} cycles:")
    print(f"  mean = {arr.mean():+.3f}  std = {arr.std():.3f}  MAE = {np.abs(arr).mean():.3f}")
    print(f"  P(err < 0) = {(arr < 0).mean():.3f}  P(err = 0) = {(arr == 0).mean():.3f}  P(err > 0) = {(arr > 0).mean():.3f}")
    print(f"  P(err in [-5,-1]) = {((arr>=-5) & (arr<=-1)).mean():.3f}")
    print(f"  P(err in [0,5]) = {((arr>=0) & (arr<=5)).mean():.3f}")
    print(f"  P(err > 5) = {(arr > 5).mean():.3f}")
    bins = [-32, -10, -5, -2, -1, 0, 1, 3, 6, 11, 33]
    hist, _ = np.histogram(arr, bins=bins)
    for i in range(len(bins) - 1):
        print(f"  err in [{bins[i]:>+3d},{bins[i+1]:>+3d}): {hist[i]:>5d}  ({hist[i]/len(arr)*100:.1f}%)")


if __name__ == "__main__":
    main()
