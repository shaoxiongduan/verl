"""Drive vllm_jacobi_patch_v2 (FIX A + B prototype) at BS=1 K=32."""
from __future__ import annotations
import argparse, json, os, sys, time

os.environ["VLLM_PLUGINS"] = ""
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vllm_jacobi_patch_v2 as patch
patch.enable()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompts_jsonl", required=True)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--max_num_seqs", type=int, default=1)
    p.add_argument("--n_prompts", type=int, default=8)
    p.add_argument("--gpu_mem_util", type=float, default=0.6)
    args = p.parse_args()

    from vllm import LLM, SamplingParams
    prompts_raw = [json.loads(l) for l in open(args.prompts_jsonl)][:args.n_prompts]
    K = int(os.environ.get("JACOBI_K", "32"))
    cap = sorted({1, 2, 4, 8, 16, 32, K+1, 2*(K+1), 4*(K+1)})
    comp = {
        "level": 3, "use_inductor": True,
        "cudagraph_mode": 1, "use_cudagraph": True,
        "cudagraph_capture_sizes": cap,
    }
    kw = dict(model=args.model, dtype="bfloat16",
              max_model_len=4096, gpu_memory_utilization=args.gpu_mem_util,
              enforce_eager=False, max_num_seqs=args.max_num_seqs,
              compilation_config=comp,
              speculative_config={
                  "method": "ngram", "num_speculative_tokens": K,
                  "prompt_lookup_min": 2, "prompt_lookup_max": 4,
              })
    t0 = time.time()
    llm = LLM(**kw)
    print(f"[diag] mode=jacobi_v2 cap={cap} load_s={time.time()-t0:.1f}", flush=True)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    chat_texts = [
        tok.apply_chat_template([{"role": "user", "content": p["input"]}],
                                tokenize=False, add_generation_prompt=True)
        for p in prompts_raw
    ]
    _ = llm.generate(chat_texts[:1], SamplingParams(max_tokens=16), use_tqdm=False)
    t0 = time.time()
    outs = llm.generate(chat_texts, sp, use_tqdm=False)
    dt = time.time() - t0
    n_tok = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"[diag] wall={dt:.2f}s tok={n_tok} TPS={n_tok/dt:.1f} ms/tok={1000*dt/n_tok:.2f}",
          flush=True)
    # Spot check: print a small portion of one output to verify correctness vs baseline
    print(f"[diag] sample_out_0_first40={list(outs[0].outputs[0].token_ids[:40])}", flush=True)


if __name__ == "__main__":
    main()
