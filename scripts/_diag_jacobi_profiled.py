"""Run instrumented Jacobi spec-decode and dump per-segment timing."""
from __future__ import annotations
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Must run before any vllm import in this process (and __main__ is re-imported in worker).
import vllm_jacobi_patch_profiled as patch
patch.enable_jacobi_spec_decode(K=int(os.environ.get("JACOBI_K", "32")))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompts_jsonl", required=True)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--max_num_seqs", type=int, default=1)
    p.add_argument("--n_prompts", type=int, default=8)
    p.add_argument("--gpu_mem_util", type=float, default=0.6)
    p.add_argument("--enforce_eager", action="store_true")
    args = p.parse_args()

    from vllm import LLM, SamplingParams

    prompts_raw = [json.loads(l) for l in open(args.prompts_jsonl)][:args.n_prompts]

    K = int(os.environ.get("JACOBI_K", "32"))
    kw = dict(
        model=args.model, dtype="bfloat16",
        max_model_len=4096, gpu_memory_utilization=args.gpu_mem_util,
        enforce_eager=args.enforce_eager,
        max_num_seqs=args.max_num_seqs,
        speculative_config={
            "method": "ngram", "num_speculative_tokens": K,
            "prompt_lookup_min": 2, "prompt_lookup_max": 4,
        },
    )
    t0 = time.time()
    llm = LLM(**kw)
    print(f"[diag] load_s={time.time()-t0:.1f}", flush=True)

    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    chat_texts = [
        tok.apply_chat_template([{"role": "user", "content": p["input"]}],
                                tokenize=False, add_generation_prompt=True)
        for p in prompts_raw
    ]
    # Warmup
    _ = llm.generate(chat_texts[:1], SamplingParams(max_tokens=16), use_tqdm=False)
    # Reset our profiling buckets so warmup iters don't pollute
    patch.reset_metrics()
    t0 = time.time()
    outs = llm.generate(chat_texts, sp, use_tqdm=False)
    dt = time.time() - t0
    n_tok = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"[diag] wall={dt:.2f}s tokens={n_tok} TPS={n_tok/dt:.1f} ms/tok={1000*dt/n_tok:.2f}", flush=True)

    # The profiling buckets live in the *worker* process for vLLM v1.
    # We can't read them from the parent. Instead, dump them from the worker via env-set side effect.
    # Easiest path: write to a file inside the worker. Use atexit in patch module.
    # ... but for the simple case (we may already be in the worker in v0/eager fallback),
    # try to print from this process too.
    rep = patch.report_metrics()
    print(f"[diag-metrics-parent] {json.dumps(rep)}", flush=True)


if __name__ == "__main__":
    main()
