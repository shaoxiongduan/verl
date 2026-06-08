#!/usr/bin/env python3
"""
Benchmark Decode-Learning's windowed batched chunked decoder on math val prompts.

Measures: per-batch aggregate TPF, wall clock, tokens/sec at BS=1,4,8,16,32.
Compares to JF reference TPF baseline (from scripts/jf_per_iter_probe.py).

Env:
  MODEL_PATH   k3 step 300 ckpt (default below)
  DATA_PATH    math val parquet (default below)
  NUM_PROMPTS  total prompts to bench (default 32)
  BLOCK_LEN    jacobi block length (default 32)
  MAX_NEW      max new tokens (default 512)
  BATCH_SIZES  comma list (default "1,4,8,16,32")
  OUT_JSON     output path
"""
from __future__ import annotations
import os, sys, json, time, re, statistics
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, "/mnt/weka/home/hao.zhang/shao/Decode-Learning")
from nanovllm import LLM, SamplingParams

MODEL_PATH = os.environ.get("MODEL_PATH", "/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/math_k3_ds_step_300")
DATA_PATH = os.environ.get("DATA_PATH", "/mnt/weka/home/hao.zhang/shao/verl/data/openmathinstruct2/val.parquet")
NUM_PROMPTS = int(os.environ.get("NUM_PROMPTS", "32"))
BLOCK_LEN = int(os.environ.get("BLOCK_LEN", "32"))
MAX_NEW = int(os.environ.get("MAX_NEW", "512"))
BATCH_SIZES = [int(x) for x in os.environ.get("BATCH_SIZES", "1,4,8,16,32").split(",")]
OUT_JSON = os.environ.get("OUT_JSON", "/mnt/weka/home/hao.zhang/shao/verl/scripts/dl_windowed_batched_bench.json")


def load_prompts(tokenizer, n: int):
    table = pq.read_table(DATA_PATH).to_pylist()
    out = []
    for r in table[:n]:
        msgs = r["prompt"]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        # Schema flexibility: openmath has r["reward_model"]["ground_truth"];
        # eval_prompts_tpf has r["expected_answer"].
        if "reward_model" in r and r["reward_model"] is not None:
            gt = r["reward_model"]["ground_truth"]
        else:
            gt = r.get("expected_answer", "")
        out.append({"text": text, "gt": gt})
    return out


_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
def extract_boxed(s: str):
    if not s:
        return None
    m = _BOXED_RE.findall(s)
    return m[-1].strip() if m else None


def grade(pred: str, gt: str) -> bool:
    p = extract_boxed(pred)
    if p is None:
        return False
    # normalize whitespace and strip $...$
    p_n = p.replace(" ", "").replace("\\,", "").replace("$", "")
    g_n = str(gt).replace(" ", "").replace("\\,", "").replace("$", "")
    if p_n == g_n:
        return True
    try:
        return float(p_n) == float(g_n)
    except Exception:
        return False


def reset_stats(dec):
    if dec is None:
        return
    for k in ("accept_len_trajectory", "accepted_tokens_per_iter",
              "draft_per_iter", "greedy_per_iter",
              "tokens_per_call", "iterations_per_call", "tokens_per_iteration"):
        if k in dec.stats:
            dec.stats[k] = []


def run_batch(llm, prompts, batch):
    """Run one batched generate call. Return (tpf_agg, total_tokens, total_iters, walltime, outputs)."""
    # Decoder is lazily created on first generate; if it exists, reset stats so this batch is isolated.
    dec = getattr(llm.model_runner, "jacobi_decoder", None)
    reset_stats(dec)

    sp = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_NEW,
        decode_strategy="jacobi",
        jacobi_block_len=BLOCK_LEN,
        jacobi_max_iterations=512,
    )
    texts = [p["text"] for p in prompts]

    t0 = time.time()
    outs = llm.generate(texts, sp, use_tqdm=False)
    dt = time.time() - t0

    # Re-fetch after generate in case it was just created.
    dec = llm.model_runner.jacobi_decoder
    total_tokens = sum(dec.stats["tokens_per_call"])
    total_iters = sum(dec.stats["iterations_per_call"])
    tpf = total_tokens / max(1, total_iters)

    return tpf, total_tokens, total_iters, dt, outs


def main():
    print(f"Loading {MODEL_PATH}...", flush=True)
    t0 = time.time()
    llm = LLM(MODEL_PATH, enforce_eager=os.environ.get("ENFORCE_EAGER","0")=="1",
              max_model_len=4096, tensor_parallel_size=1)
    print(f"Loaded in {time.time() - t0:.1f}s", flush=True)

    prompts = load_prompts(llm.tokenizer, NUM_PROMPTS)
    print(f"Loaded {len(prompts)} math val prompts", flush=True)

    results = {"model_path": MODEL_PATH, "block_len": BLOCK_LEN, "max_new": MAX_NEW,
               "num_prompts": NUM_PROMPTS, "by_bs": {}}

    for bs in BATCH_SIZES:
        if bs > len(prompts):
            print(f"\n[skip BS={bs}: only {len(prompts)} prompts]", flush=True)
            continue

        n_rounds = len(prompts) // bs
        print(f"\n=== BS={bs} ({n_rounds} batches) ===", flush=True)

        round_results = []
        all_outputs = []
        for r_idx in range(n_rounds):
            sub = prompts[r_idx * bs : (r_idx + 1) * bs]
            tpf, ntok, niter, dt, outs = run_batch(llm, sub, batch=bs)
            tps = ntok / max(1e-6, dt)
            print(f"  round {r_idx}: TPF={tpf:.3f}  tokens={ntok}  iters={niter}  dt={dt:.2f}s  tok/s={tps:.1f}", flush=True)
            round_results.append({"tpf": tpf, "tokens": ntok, "iters": niter, "wall_s": dt, "tps": tps})
            # capture outputs for grading
            for p, o in zip(sub, outs):
                gen = o["text"] if isinstance(o, dict) else str(o)
                all_outputs.append({"gt": p["gt"], "gen": gen, "correct": grade(gen, p["gt"])})

        mean_tpf = statistics.mean(r["tpf"] for r in round_results)
        mean_tps = statistics.mean(r["tps"] for r in round_results)
        total_wall = sum(r["wall_s"] for r in round_results)
        n_correct = sum(1 for o in all_outputs if o["correct"])
        acc = n_correct / max(1, len(all_outputs))
        print(f"  -> mean TPF={mean_tpf:.3f}  mean tok/s={mean_tps:.1f}  total_wall={total_wall:.1f}s  acc={acc:.3f} ({n_correct}/{len(all_outputs)})", flush=True)

        results["by_bs"][str(bs)] = {
            "rounds": round_results,
            "mean_tpf": mean_tpf,
            "mean_tps": mean_tps,
            "total_wall": total_wall,
            "n_correct": n_correct,
            "n_total": len(all_outputs),
            "acc": acc,
        }

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"{'BS':>4} {'TPF':>7} {'tok/s':>9} {'wall(s)':>9} {'acc':>7}")
    for bs in BATCH_SIZES:
        k = str(bs)
        if k not in results["by_bs"]:
            continue
        r = results["by_bs"][k]
        print(f"{bs:>4} {r['mean_tpf']:>7.3f} {r['mean_tps']:>9.1f} {r['total_wall']:>9.1f} {r['acc']:>7.3f}")
    print(f"\nJF reference (single-seq, math k3 step 300): TPF ≈ 3.35")
    print(f"\nJSON: {OUT_JSON}")

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
