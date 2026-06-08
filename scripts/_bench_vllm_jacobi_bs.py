"""Proper vLLM Jacobi BS sweep — cap concurrent reqs via max_num_seqs."""
from __future__ import annotations
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vllm_jacobi_patch  # noqa

_K = int(os.environ.get("JACOBI_K", "32"))
vllm_jacobi_patch.enable_jacobi_spec_decode(K=_K, traj_path=None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompts_jsonl", required=True)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--max_num_seqs", type=int, required=True,
                   help="vLLM max concurrent reqs (= effective batch size)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--gpu_mem_util", type=float, default=0.6)
    args = p.parse_args()

    from vllm import LLM, SamplingParams

    prompts_raw = [json.loads(l) for l in open(args.prompts_jsonl)]
    print(f"Loaded {len(prompts_raw)} prompts | max_num_seqs={args.max_num_seqs}", flush=True)

    llm = LLM(
        model=args.model, dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=args.gpu_mem_util,
        enforce_eager=False,
        max_num_seqs=args.max_num_seqs,
        speculative_config={
            "method": "ngram",
            "num_speculative_tokens": _K,
            "prompt_lookup_min": 2,
            "prompt_lookup_max": 4,
        },
    )
    tokenizer = llm.get_tokenizer()
    sp = SamplingParams(temperature=args.temperature, max_tokens=args.max_new_tokens)
    chat_texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": pdata["input"]}],
            tokenize=False, add_generation_prompt=True,
        )
        for pdata in prompts_raw
    ]
    # warmup with 1 short req so first batch isn't penalized by JIT
    _ = llm.generate(chat_texts[:1], SamplingParams(max_tokens=32), use_tqdm=False)
    t0 = time.time()
    outs = llm.generate(chat_texts, sp, use_tqdm=False)
    dt = time.time() - t0
    n_tok = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"\nBS_cap={args.max_num_seqs}  wall={dt:.2f}s  tokens={n_tok}  TPS={n_tok/dt:.1f}", flush=True)


if __name__ == "__main__":
    main()
