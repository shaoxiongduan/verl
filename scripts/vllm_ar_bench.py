#!/usr/bin/env python3
"""
vLLM AR-decoding benchmark — same model, same prompts, same max_tokens as
dl_windowed_batched_bench.py so we can directly compare wall clock + tokens/sec.

vLLM does NOT do Jacobi speculative decoding here, so per-prompt TPF is N/A.
We compare tok/s and accuracy across batch sizes.
"""
from __future__ import annotations
import os, sys, json, time, re, statistics
import pyarrow.parquet as pq

from vllm import LLM, SamplingParams

MODEL_PATH = os.environ.get("MODEL_PATH", "/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/math_k3_ds_step_300")
DATA_PATH = os.environ.get("DATA_PATH", "/mnt/weka/home/hao.zhang/shao/verl/data/openmathinstruct2/val.parquet")
NUM_PROMPTS = int(os.environ.get("NUM_PROMPTS", "32"))
MAX_NEW = int(os.environ.get("MAX_NEW", "512"))
BATCH_SIZES = [int(x) for x in os.environ.get("BATCH_SIZES", "1,4,8,16,32").split(",")]
OUT_JSON = os.environ.get("OUT_JSON", "/mnt/weka/home/hao.zhang/shao/verl/scripts/vllm_ar_bench.json")


def load_prompts(tokenizer, n: int):
    rows = pq.read_table(DATA_PATH).to_pylist()[:n]
    out = []
    for r in rows:
        msgs = r["prompt"]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        out.append({"text": text, "gt": r["reward_model"]["ground_truth"]})
    return out


_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
def extract_boxed(s):
    if not s: return None
    m = _BOXED_RE.findall(s)
    return m[-1].strip() if m else None

def grade(pred, gt):
    p = extract_boxed(pred)
    if p is None: return False
    p_n = p.replace(" ", "").replace("\\,", "").replace("$", "")
    g_n = str(gt).replace(" ", "").replace("\\,", "").replace("$", "")
    if p_n == g_n: return True
    try: return float(p_n) == float(g_n)
    except Exception: return False


def main():
    print(f"Loading {MODEL_PATH} in vLLM ...", flush=True)
    t0 = time.time()
    llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        enforce_eager=False,
    )
    print(f"Loaded in {time.time() - t0:.1f}s", flush=True)

    tok = llm.get_tokenizer()
    prompts = load_prompts(tok, NUM_PROMPTS)
    print(f"Loaded {len(prompts)} prompts", flush=True)

    sp = SamplingParams(temperature=0.0, max_tokens=MAX_NEW)

    results = {"model_path": MODEL_PATH, "max_new": MAX_NEW, "num_prompts": NUM_PROMPTS, "by_bs": {}}

    for bs in BATCH_SIZES:
        if bs > len(prompts):
            print(f"\n[skip BS={bs}]", flush=True); continue

        n_rounds = len(prompts) // bs
        print(f"\n=== BS={bs} ({n_rounds} batches) ===", flush=True)

        round_results = []
        all_outputs = []
        for r_idx in range(n_rounds):
            sub = prompts[r_idx * bs : (r_idx + 1) * bs]
            texts = [p["text"] for p in sub]
            t0 = time.time()
            outs = llm.generate(texts, sp, use_tqdm=False)
            dt = time.time() - t0
            n_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
            tps = n_tokens / max(1e-6, dt)
            print(f"  round {r_idx}: tokens={n_tokens}  dt={dt:.2f}s  tok/s={tps:.1f}", flush=True)
            round_results.append({"tokens": n_tokens, "wall_s": dt, "tps": tps})
            for p, o in zip(sub, outs):
                gen = o.outputs[0].text
                all_outputs.append({"gt": p["gt"], "gen": gen, "correct": grade(gen, p["gt"])})

        mean_tps = statistics.mean(r["tps"] for r in round_results)
        total_wall = sum(r["wall_s"] for r in round_results)
        n_correct = sum(1 for o in all_outputs if o["correct"])
        acc = n_correct / max(1, len(all_outputs))
        print(f"  -> mean tok/s={mean_tps:.1f}  total_wall={total_wall:.1f}s  acc={acc:.3f} ({n_correct}/{len(all_outputs)})", flush=True)
        results["by_bs"][str(bs)] = {
            "rounds": round_results, "mean_tps": mean_tps,
            "total_wall": total_wall, "n_correct": n_correct,
            "n_total": len(all_outputs), "acc": acc,
        }

    print(f"\n{'='*50}")
    print(f"SUMMARY (vLLM, greedy AR)")
    print(f"{'='*50}")
    print(f"{'BS':>4} {'tok/s':>9} {'wall(s)':>9} {'acc':>7}")
    for bs in BATCH_SIZES:
        k = str(bs)
        if k not in results["by_bs"]: continue
        r = results["by_bs"][k]
        print(f"{bs:>4} {r['mean_tps']:>9.1f} {r['total_wall']:>9.1f} {r['acc']:>7.3f}")
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON: {OUT_JSON}")


if __name__ == "__main__":
    main()
