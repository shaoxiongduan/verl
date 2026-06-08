"""Diagnose vLLM Jacobi slowness at small BS.

Compare:
  A) vLLM AR (no spec)
  B) vLLM ngram spec-decode (no monkey-patch)  -- vanilla SpecDecode baseline
  C) vLLM Jacobi via our monkey-patch
  D) vLLM Jacobi via plugin (via env var, same logical path as in verl rollout)

Same model, prompts, K, max_new, max_num_seqs. Reports tok/s + ms/forward.
"""
from __future__ import annotations
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODE = os.environ.get("VLLM_DIAG_MODE", "ar")  # ar | ngram | jacobi
print(f"[diag] MODE={MODE}", flush=True)

if MODE == "jacobi":
    import vllm_jacobi_patch  # noqa
    vllm_jacobi_patch.enable_jacobi_spec_decode(K=int(os.environ.get("JACOBI_K", "32")), traj_path=None)


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
    )
    if MODE in ("jacobi", "ngram"):
        kw["speculative_config"] = {
            "method": "ngram",
            "num_speculative_tokens": K,
            "prompt_lookup_min": 2,
            "prompt_lookup_max": 4,
        }

    t0 = time.time()
    llm = LLM(**kw)
    print(f"[diag] load_s={time.time()-t0:.1f}", flush=True)

    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    chat_texts = [
        tok.apply_chat_template(
            [{"role": "user", "content": p["input"]}],
            tokenize=False, add_generation_prompt=True,
        )
        for p in prompts_raw
    ]
    # Warm-up: 1 short req
    _ = llm.generate(chat_texts[:1], SamplingParams(max_tokens=32), use_tqdm=False)
    # Real bench
    t0 = time.time()
    outs = llm.generate(chat_texts, sp, use_tqdm=False)
    dt = time.time() - t0
    n_tok = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"[diag] mode={MODE} max_num_seqs={args.max_num_seqs} enforce_eager={args.enforce_eager}", flush=True)
    print(f"[diag] n_prompts={args.n_prompts} wall={dt:.2f}s tokens={n_tok} TPS={n_tok/dt:.1f}", flush=True)
    # Per-prompt avg tokens
    per_prompt = [len(o.outputs[0].token_ids) for o in outs]
    avg_tok = sum(per_prompt)/len(per_prompt)
    print(f"[diag] avg_tokens_per_prompt={avg_tok:.1f}  ms_per_token={1000*dt/n_tok:.2f}", flush=True)


if __name__ == "__main__":
    main()
