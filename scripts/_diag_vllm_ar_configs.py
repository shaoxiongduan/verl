"""Try multiple vLLM AR configs at BS=1 to find what gets us to roofline.

Theoretical: 7B bf16 = 14 GB; H200 BW = 4.8 TB/s → 2.92 ms/token = 343 tok/s upper bound.
Observed default: 178 tok/s (= 5.6 ms/token). 2x overhead — Python/scheduler.

Each test loads its own LLM, generates 8 prompts × 256 tokens at BS=1, reports tok/s.
"""
from __future__ import annotations
import os, sys, time, json, gc, subprocess


CASE = os.environ.get("AR_CASE", "default")
print(f"\n========== CASE={CASE} ==========\n", flush=True)


def get_kwargs():
    kw = dict(
        model=os.environ["MODEL"],
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=float(os.environ.get("GMU", "0.6")),
        enforce_eager=os.environ.get("EAGER", "0") == "1",
        max_num_seqs=1,
    )
    if os.environ.get("ATTN_BACKEND"):
        os.environ["VLLM_ATTENTION_BACKEND"] = os.environ["ATTN_BACKEND"]
    if os.environ.get("CHUNKED", "1") == "0":
        kw["enable_chunked_prefill"] = False
    return kw


def main():
    from vllm import LLM, SamplingParams

    prompts_raw = [json.loads(l) for l in open(os.environ["PROMPTS"])][:8]
    kw = get_kwargs()
    print(f"[case={CASE}] kwargs={kw}  ATTN={os.environ.get('VLLM_ATTENTION_BACKEND')}", flush=True)
    t0 = time.time()
    llm = LLM(**kw)
    print(f"[case={CASE}] load_s={time.time()-t0:.1f}", flush=True)
    tok = llm.get_tokenizer()
    chat_texts = [
        tok.apply_chat_template([{"role":"user","content":p["input"]}],
                                 tokenize=False, add_generation_prompt=True)
        for p in prompts_raw
    ]
    # Warmup
    sp = SamplingParams(temperature=0.0, max_tokens=32)
    _ = llm.generate(chat_texts[:1], sp, use_tqdm=False)
    # Bench
    sp = SamplingParams(temperature=0.0, max_tokens=256)
    t0 = time.time()
    outs = llm.generate(chat_texts, sp, use_tqdm=False)
    dt = time.time() - t0
    n_tok = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"[case={CASE}] BS=1 wall={dt:.2f}s tokens={n_tok} TPS={n_tok/dt:.1f} ms_tok={1000*dt/n_tok:.2f}", flush=True)


if __name__ == "__main__":
    main()
