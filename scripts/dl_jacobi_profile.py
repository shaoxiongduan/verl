#!/usr/bin/env python3
"""
Profile DL windowed batched Jacobi to localize the BS-scaling gap vs vLLM.

For each BS, monkey-patches `_forward_batched` to record:
  - per-call wall time (forward kernel + dispatch)
  - per-call B (active concurrent seqs in this forward)

Then computes:
  - mean active concurrency vs BS (how many seqs are alive on average per iter)
  - total forward time vs total wall (overhead share)
  - per-token forward time scaling with B

Hypothesis: at large BS the gap comes from (a) static batching (active << BS as
short seqs finish), and (b) per-forward kernel growth with B*L tokens.
"""
from __future__ import annotations
import os, sys, json, time, re, statistics
from pathlib import Path
import pyarrow.parquet as pq

sys.path.insert(0, "/mnt/weka/home/hao.zhang/shao/Decode-Learning")
import torch
from nanovllm import LLM, SamplingParams

MODEL_PATH = os.environ.get("MODEL_PATH", "/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/math_k3_ds_step_300")
DATA_PATH = os.environ.get("DATA_PATH", "/mnt/weka/home/hao.zhang/shao/verl/data/openmathinstruct2/val.parquet")
NUM_PROMPTS = int(os.environ.get("NUM_PROMPTS", "32"))
BLOCK_LEN = int(os.environ.get("BLOCK_LEN", "32"))
MAX_NEW = int(os.environ.get("MAX_NEW", "512"))
BATCH_SIZES = [int(x) for x in os.environ.get("BATCH_SIZES", "1,4,8,16,32").split(",")]
OUT_JSON = os.environ.get("OUT_JSON", "/mnt/weka/home/hao.zhang/shao/verl/scripts/dl_jacobi_profile.json")


def load_prompts(tokenizer, n: int):
    rows = pq.read_table(DATA_PATH).to_pylist()[:n]
    out = []
    for r in rows:
        text = tokenizer.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True)
        out.append(text)
    return out


def install_hooks(decoder):
    """Wrap _forward_batched to record (B, wall_ms)."""
    decoder._prof_calls = []
    orig = decoder._forward_batched

    def wrapped(sub_seqs, draft_batch):
        B, L = int(draft_batch.size(0)), int(draft_batch.size(1))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = orig(sub_seqs, draft_batch)
        torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1e3
        decoder._prof_calls.append((B, L, dt_ms))
        return out

    decoder._forward_batched = wrapped


def reset_prof(decoder):
    decoder._prof_calls = []
    for k in ("tokens_per_call", "iterations_per_call"):
        if k in decoder.stats:
            decoder.stats[k] = []


def main():
    print(f"Loading {MODEL_PATH} (graphs ON)...", flush=True)
    t0 = time.time()
    llm = LLM(MODEL_PATH, enforce_eager=False, max_model_len=4096, tensor_parallel_size=1)
    print(f"Loaded in {time.time() - t0:.1f}s", flush=True)

    prompts = load_prompts(llm.tokenizer, NUM_PROMPTS)
    print(f"Loaded {len(prompts)} prompts", flush=True)

    # Warmup + ensure decoder exists by running a tiny batch.
    sp = SamplingParams(temperature=0.0, max_tokens=64,
                        decode_strategy="jacobi", jacobi_block_len=BLOCK_LEN,
                        jacobi_max_iterations=512)
    _ = llm.generate(prompts[:2], sp, use_tqdm=False)
    dec = llm.model_runner.jacobi_decoder
    install_hooks(dec)
    print("Hook installed on _forward_batched\n", flush=True)

    sp_full = SamplingParams(temperature=0.0, max_tokens=MAX_NEW,
                             decode_strategy="jacobi", jacobi_block_len=BLOCK_LEN,
                             jacobi_max_iterations=512)

    results = {"by_bs": {}}

    for bs in BATCH_SIZES:
        if bs > len(prompts):
            continue
        sub = prompts[:bs]
        reset_prof(dec)

        torch.cuda.synchronize()
        t_wall = time.perf_counter()
        _ = llm.generate(sub, sp_full, use_tqdm=False)
        torch.cuda.synchronize()
        wall_s = time.perf_counter() - t_wall

        calls = list(dec._prof_calls)
        n_calls = len(calls)
        total_fwd_ms = sum(c[2] for c in calls)
        total_tokens = sum(dec.stats["tokens_per_call"])

        # group by B
        by_B = {}
        for B, L, dt in calls:
            by_B.setdefault(B, []).append(dt)

        # iter-by-iter active count over the run (in call order)
        avg_active = statistics.mean(c[0] for c in calls) if calls else 0
        max_active = max((c[0] for c in calls), default=0)

        # forward cost vs B (one repr per B): median dt per token (B*L)
        cost_by_B = {}
        for B, dts in sorted(by_B.items()):
            n = len(dts)
            med_dt = statistics.median(dts)
            cost_by_B[B] = {"n_calls": n, "median_dt_ms": med_dt}

        overhead_ms = wall_s * 1e3 - total_fwd_ms
        overhead_share = overhead_ms / (wall_s * 1e3) if wall_s > 0 else 0
        fwd_tps = total_tokens / (total_fwd_ms / 1e3) if total_fwd_ms > 0 else 0
        wall_tps = total_tokens / wall_s if wall_s > 0 else 0

        print(f"=== BS={bs} ===")
        print(f"  wall={wall_s*1e3:.0f}ms  fwd_sum={total_fwd_ms:.0f}ms ({100-100*overhead_share:.1f}% of wall)  overhead={overhead_ms:.0f}ms")
        print(f"  calls={n_calls}  tokens={total_tokens}  wall_tps={wall_tps:.0f}  fwd_tps={fwd_tps:.0f}")
        print(f"  avg active seqs / iter = {avg_active:.2f} / {bs}    max={max_active}")
        print(f"  per-B kernel cost (median ms): " + ", ".join(f"B={B}:{c['median_dt_ms']:.2f}ms (n={c['n_calls']})" for B,c in cost_by_B.items()))
        print()
        results["by_bs"][str(bs)] = {
            "wall_ms": wall_s * 1e3, "fwd_ms": total_fwd_ms, "overhead_ms": overhead_ms,
            "overhead_share": overhead_share, "calls": n_calls, "tokens": total_tokens,
            "wall_tps": wall_tps, "fwd_tps": fwd_tps,
            "avg_active": avg_active, "max_active": max_active,
            "cost_by_B": cost_by_B,
        }

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON: {OUT_JSON}")


if __name__ == "__main__":
    main()
