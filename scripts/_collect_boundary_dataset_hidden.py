"""Collect boundary dataset INCLUDING last-layer hidden states per K-position.

Saves two files:
  <out>.jsonl   — same cycle-level metadata as _collect_boundary_dataset.py
                  (top1_prob, entropy, margin_log, top5*, accept labels, etc.)
  <out>.npz     — keys:
                  hidden:  fp16 array shape (N_cycles, K, d_hidden)
                  accept:  int8 array shape (N_cycles, K)
                  n_acc:   int16 array shape (N_cycles,)
                  prompt_idx: int16 array (N_cycles,)
                  cycle_idx:  int16 array (N_cycles,)
                  top1_prob, entropy, margin_log, top5_prob_sum: fp16 (N, K)
                  warm_argmax: int32 (N, K)
                  verify_argmax: int32 (N, K)
"""
from __future__ import annotations
import argparse, json, math, random
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompts_jsonl", required=True)
    p.add_argument("--out_prefix", required=True)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--max_new", type=int, default=512)
    p.add_argument("--max_cycles", type=int, default=128)
    p.add_argument("--vocab_size", type=int, default=152064)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def per_position_features(logits_K: torch.Tensor):
    K, V = logits_K.shape
    fp32 = logits_K.float()
    probs = F.softmax(fp32, dim=-1)
    top5_vals, top5_idx = probs.topk(5, dim=-1)
    top1_prob = top5_vals[:, 0]
    top5_sum = top5_vals.sum(dim=-1)
    entropy = -(probs * (probs.clamp_min(1e-12).log())).sum(dim=-1)
    top2_logits, _ = fp32.topk(2, dim=-1)
    margin_log = top2_logits[:, 0] - top2_logits[:, 1]
    return {
        "top1_prob": top1_prob.cpu().numpy().astype(np.float16),
        "top5_prob_sum": top5_sum.cpu().numpy().astype(np.float16),
        "entropy": entropy.cpu().numpy().astype(np.float16),
        "margin_log": margin_log.cpu().numpy().astype(np.float16),
        "top1_token": top5_idx[:, 0].cpu().numpy().astype(np.int32),
    }


@torch.no_grad()
def collect_for_prompt(model, prompt_ids, args, rng, prompt_idx, store):
    device = model.device
    K = args.K
    eos_id = 151645; pad_id = 151643; stop_ids = {eos_id, pad_id}
    committed = list(prompt_ids)
    total_tok = 0
    n_cycles = 0

    while total_tok < args.max_new and n_cycles < args.max_cycles:
        L = len(committed)
        mixed_draft = [rng.randrange(args.vocab_size) for _ in range(K)]

        # Warm forward, also return hidden states
        inp1 = torch.tensor([committed + mixed_draft], dtype=torch.long, device=device)
        out1 = model(input_ids=inp1, output_hidden_states=True)
        warm_logits = out1.logits[0, L - 1 : L - 1 + K, :]
        warm_argmax = warm_logits.argmax(dim=-1).cpu().tolist()
        hidden_last = out1.hidden_states[-1][0, L - 1 : L - 1 + K, :].cpu().to(torch.float16).numpy()
        feats = per_position_features(warm_logits)

        # Verify forward (no hidden states needed)
        inp2 = torch.tensor([committed + warm_argmax], dtype=torch.long, device=device)
        verify_argmax = model(input_ids=inp2).logits[0, L - 1 : L - 1 + K, :].argmax(dim=-1).cpu().tolist()

        accept = np.array([int(warm_argmax[j] == verify_argmax[j]) for j in range(K)], dtype=np.int8)
        n_acc = 0
        for j in range(K):
            if accept[j]:
                n_acc += 1
            else:
                break

        store["hidden"].append(hidden_last)
        store["accept"].append(accept)
        store["n_acc"].append(n_acc)
        store["prompt_idx"].append(prompt_idx)
        store["cycle_idx"].append(n_cycles)
        for k in ("top1_prob", "entropy", "margin_log", "top5_prob_sum"):
            store[k].append(feats[k])
        store["warm_argmax"].append(np.array(warm_argmax, dtype=np.int32))
        store["verify_argmax"].append(np.array(verify_argmax, dtype=np.int32))

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

    return n_cycles, total_tok


def main():
    args = parse_args()
    print(f"[col] loading {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    prompts = [json.loads(l) for l in open(args.prompts_jsonl)]
    rng = random.Random(args.seed)
    print(f"[col] K={args.K} max_new={args.max_new} n_prompts={len(prompts)} hidden_dim={model.config.hidden_size}", flush=True)

    store = {k: [] for k in ("hidden", "accept", "n_acc", "prompt_idx", "cycle_idx",
                              "top1_prob", "entropy", "margin_log", "top5_prob_sum",
                              "warm_argmax", "verify_argmax")}
    total_cycles = 0; total_tokens = 0
    for i, p in enumerate(prompts):
        chat = [{"role": "user", "content": p["input"]}]
        text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        pid = tok(text, return_tensors="pt").input_ids[0].tolist()
        nc, nt = collect_for_prompt(model, pid, args, rng, i, store)
        total_cycles += nc; total_tokens += nt
        print(f"[col] [{i+1}/{len(prompts)}] cycles={nc} tokens={nt} total_cycles={total_cycles}", flush=True)

    # Convert and save
    out_npz = args.out_prefix + ".npz"
    arrays = {}
    for k in ("hidden",):
        arrays[k] = np.stack(store[k], axis=0)  # (N, K, d)
    for k in ("accept", "top1_prob", "entropy", "margin_log", "top5_prob_sum",
              "warm_argmax", "verify_argmax"):
        arrays[k] = np.stack(store[k], axis=0)  # (N, K)
    arrays["n_acc"] = np.array(store["n_acc"], dtype=np.int16)
    arrays["prompt_idx"] = np.array(store["prompt_idx"], dtype=np.int16)
    arrays["cycle_idx"] = np.array(store["cycle_idx"], dtype=np.int16)
    np.savez_compressed(out_npz, **arrays)
    print(f"\n[col] saved {out_npz}  shapes: hidden={arrays['hidden'].shape}  accept={arrays['accept'].shape}", flush=True)
    print(f"[col] DONE: {total_cycles} cycles dumped, {total_tokens} tokens generated", flush=True)


if __name__ == "__main__":
    main()
