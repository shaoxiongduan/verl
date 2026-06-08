"""Single-BS Jacobi/AR bench (fresh process per call to avoid patch state leak).
Prints exactly one CSV row to stdout: mode,bs,wall_s,output_tokens,tps"""
from __future__ import annotations
import argparse, json, os, sys, time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument("--prompts_jsonl", required=True)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--bs", type=int, required=True)
    p.add_argument("--mode", required=True, choices=["jacobi", "ar"])
    p.add_argument("--gpu_mem_util", type=float, default=0.85)
    args = p.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    if args.mode == "jacobi":
        import vllm_jacobi_patch
        vllm_jacobi_patch.enable_jacobi_spec_decode(K=args.K)

    from vllm import LLM, SamplingParams

    llm_kwargs = dict(
        model=args.target, dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=args.gpu_mem_util,
        enforce_eager=False,
        max_num_seqs=args.bs,
    )
    if args.mode == "jacobi":
        llm_kwargs["speculative_config"] = {
            "method": "ngram",
            "num_speculative_tokens": args.K,
            "prompt_lookup_min": 2,
            "prompt_lookup_max": 4,
        }
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    prompts_raw = [json.loads(l) for l in open(args.prompts_jsonl)]
    chat_texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p["input"]}],
            tokenize=False, add_generation_prompt=True,
        )
        for p in prompts_raw
    ]
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    _ = llm.generate(chat_texts[:1], SamplingParams(max_tokens=32), use_tqdm=False)
    t0 = time.time()
    outs = llm.generate(chat_texts, sp, use_tqdm=False)
    dt = time.time() - t0
    n_tok = sum(len(o.outputs[0].token_ids) for o in outs)
    tps = n_tok / dt
    print(f"CSVROW,{args.mode},{args.bs},{dt:.3f},{n_tok},{tps:.2f}", flush=True)


if __name__ == "__main__":
    main()
