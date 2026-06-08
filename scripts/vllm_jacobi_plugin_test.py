"""Verify the jacobi vLLM plugin auto-installs WITHOUT any code changes.

The caller does NOT call enable_jacobi_spec_decode(). Just vanilla vLLM use.
The plugin entry-point fires when vLLM loads, both in the parent and the
subprocess (EngineCore), thanks to vllm.plugins.load_general_plugins().
"""
from __future__ import annotations
import os, time, re
import pyarrow.parquet as pq

# No import of vllm_jacobi_patch. No enable_jacobi_spec_decode() call.
from vllm import LLM, SamplingParams

MODEL = "/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/math_k3_ds_step_300"
DATA = "/mnt/weka/home/hao.zhang/shao/verl/data/openmathinstruct2/val.parquet"
NUM = int(os.environ.get("NUM_PROMPTS", "4"))
MAX_NEW = int(os.environ.get("MAX_NEW", "128"))


def grade(p, g):
    m = re.findall(r"\\boxed\{([^{}]*)\}", p or "")
    if not m: return False
    pn = m[-1].strip().replace(" ", "").replace("\\,", "").replace("$", "")
    gn = str(g).replace(" ", "").replace("\\,", "").replace("$", "")
    if pn == gn: return True
    try: return float(pn) == float(gn)
    except: return False


def main():
    K = int(os.environ.get("JACOBI_K", "32"))
    llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.6, enforce_eager=False,
              speculative_config={"method": "ngram", "num_speculative_tokens": K,
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

    print(f"\n[PLUGIN-TEST] BS={NUM} MAX_NEW={MAX_NEW}")
    print(f"  wall={dt:.2f}s  tokens={n_tok}  tok/s={n_tok/dt:.0f}  acc={n_ok}/{NUM}")
    print(f"  (Plugin should have auto-installed Jacobi; expected speed > 800 tok/s @ BS=4 K=32)")


if __name__ == "__main__":
    main()
