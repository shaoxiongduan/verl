"""Smoke test for the drop-in vllm_jacobi_patch module.

Demonstrates the intended caller flow:
  1. import vllm_jacobi_patch + call enable_jacobi_spec_decode(K, traj_path)
  2. import vllm normally + use as usual with speculative_config method="ngram"
  3. after generation, call aggregate_trajectories() for TPF stats
"""
from __future__ import annotations
import sys, os, time, re
import pyarrow.parquet as pq

# 1) Install the patches BEFORE importing vllm.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vllm_jacobi_patch
TRAJ = "/mnt/weka/home/hao.zhang/shao/verl/scripts/traj_smoke.jsonl"
vllm_jacobi_patch.enable_jacobi_spec_decode(K=32, traj_path=TRAJ)

# 2) Use vLLM normally.
from vllm import LLM, SamplingParams

MODEL = "/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/math_k3_ds_step_300"
DATA = "/mnt/weka/home/hao.zhang/shao/verl/data/openmathinstruct2/val.parquet"
NUM = int(os.environ.get("NUM_PROMPTS", "8"))
MAX_NEW = int(os.environ.get("MAX_NEW", "256"))


def grade(p, g):
    m = re.findall(r"\\boxed\{([^{}]*)\}", p or "")
    if not m: return False
    pn = m[-1].strip().replace(" ", "").replace("\\,", "").replace("$", "")
    gn = str(g).replace(" ", "").replace("\\,", "").replace("$", "")
    if pn == gn: return True
    try: return float(pn) == float(gn)
    except: return False


def main():
    llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.6, enforce_eager=False,
              speculative_config={"method": "ngram", "num_speculative_tokens": 32,
                                  "prompt_lookup_min": 2, "prompt_lookup_max": 4})
    tok = llm.get_tokenizer()
    rows = pq.read_table(DATA).to_pylist()[:NUM]
    prompts = [tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True) for r in rows]
    gts = [r["reward_model"]["ground_truth"] for r in rows]

    sp = SamplingParams(temperature=0.0, max_tokens=MAX_NEW)
    t0 = time.time()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    dt = time.time() - t0
    n_tok = sum(len(o.outputs[0].token_ids) for o in outs)
    n_ok = sum(1 for o, gt in zip(outs, gts) if grade(o.outputs[0].text, gt))

    stats = vllm_jacobi_patch.aggregate_trajectories()

    print(f"\n[SMOKE] BS={NUM} K=32 MAX_NEW={MAX_NEW}")
    print(f"  wall={dt:.2f}s  tokens={n_tok}  tok/s={n_tok/dt:.0f}  acc={n_ok}/{NUM}")
    print(f"  spec_iters={stats['n_iters']}  agg_tpf={stats['agg_tpf']:.2f}  "
          f"avg_per_req={sum(stats['per_req_tpf'])/max(1,len(stats['per_req_tpf'])):.2f}")
    print(f"  trajectory files: {stats['n_files']}  total spec tokens accepted: {stats['spec_tok_total']}")
    print(f"  TRAJ at: {vllm_jacobi_patch.get_trajectory_path()}.<PID>")


if __name__ == "__main__":
    main()
