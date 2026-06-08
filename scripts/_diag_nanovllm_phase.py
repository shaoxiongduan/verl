"""Per-phase timing for nanovllm AR and Jacobi at BS=1 K=32 on JF Math 7B.

Apples-to-apples with vLLM phase profile:
  same model (JF Math 7B bf16), same prompts (eval_passk/eval_prompts_tpf.jsonl),
  same K=32, max_num_seqs=1, max_tokens=128.

nanovllm exposes a built-in profiler activated by PROFILE=1 env var (set BEFORE
this module imports nanovllm). It auto-inserts cuda-sync brackets at:
  jacobi.draft_build, jacobi.forward, jacobi.lm_head, jacobi.verify,
  jacobi.commit, jacobi.trim, jacobi.next_draft, etc.
For AR we add a wrap around run_model() to bucket as "ar.forward".

Outputs the profiler's full breakdown.
"""
from __future__ import annotations
import os, sys, time, json

# Activate PROFILE BEFORE importing nanovllm
os.environ["PROFILE"] = "1"

sys.path.insert(0, "/mnt/weka/home/hao.zhang/shao/Decode-Learning")

import argparse
import torch
from nanovllm import LLM, SamplingParams
from nanovllm.engine.model_runner import get_profiler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompts_jsonl", required=True)
    p.add_argument("--mode", choices=["ar", "jacobi"], default="jacobi")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--max_num_seqs", type=int, default=1)
    p.add_argument("--n_prompts", type=int, default=8)
    p.add_argument("--block_len", type=int, default=32)
    p.add_argument("--warmup", action="store_true")
    args = p.parse_args()

    print(f"[diag] mode={args.mode} loading model {args.model}", flush=True)
    t0 = time.time()
    # max_num_seqs=1 breaks nanovllm cudagraph capture; default works fine. BS=1
    # in steady state is enforced by feeding prompts sequentially below.
    llm = LLM(args.model, enforce_eager=False, max_model_len=4096,
              tensor_parallel_size=1)
    print(f"[diag] load_s={time.time()-t0:.1f}", flush=True)

    # Build prompts (token IDs avoid chat-template differences)
    tok = llm.tokenizer
    prompts_raw = [json.loads(l) for l in open(args.prompts_jsonl)][:args.n_prompts]
    chat_texts = [
        tok.apply_chat_template([{"role": "user", "content": p["input"]}],
                                tokenize=False, add_generation_prompt=True)
        for p in prompts_raw
    ]

    if args.mode == "jacobi":
        sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens,
                            decode_strategy="jacobi", jacobi_block_len=args.block_len,
                            jacobi_max_iterations=512, ignore_eos=False)
    else:
        sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens,
                            decode_strategy="autoregressive", ignore_eos=False)

    # Add an "ar.forward" tracker around run_model() for AR mode.
    mr = llm.model_runner
    if args.mode == "ar":
        prof = get_profiler()
        orig_run = mr.run_model
        def w_run(input_ids, positions, is_prefill):
            torch.cuda.synchronize(); t = time.perf_counter()
            out = orig_run(input_ids, positions, is_prefill)
            torch.cuda.synchronize();
            dt = (time.perf_counter() - t) * 1e3
            name = "ar.prefill" if is_prefill else "ar.forward"
            prof.timings[name] = prof.timings.get(name, 0.0) + dt
            prof.counts[name] = prof.counts.get(name, 0) + 1
            if not is_prefill:
                prof.add_tokens(1); prof.add_iteration()
            return out
        mr.run_model = w_run

    # 1 warmup
    _ = llm.generate([chat_texts[0]], SamplingParams(max_tokens=16,
        decode_strategy=("jacobi" if args.mode == "jacobi" else "autoregressive"),
        temperature=0.0, jacobi_block_len=args.block_len), use_tqdm=False)

    # Reset profiler buckets after warmup
    prof = get_profiler()
    prof.reset()
    prof.enabled = True

    t0 = time.time()
    # Feed prompts ONE AT A TIME to enforce BS=1 in steady state.
    all_outs = []
    for ct in chat_texts:
        outs = llm.generate([ct], sp, use_tqdm=False)
        all_outs.extend(outs)
    dt = time.time() - t0
    n_tok = sum(len(o["token_ids"]) for o in all_outs)
    print(f"[diag] mode={args.mode} wall={dt:.3f}s tok={n_tok} TPS={n_tok/dt:.1f} "
          f"ms/tok={1000*dt/n_tok:.3f}", flush=True)
    print(f"[diag] sample_out_0_first40={all_outs[0]['token_ids'][:40]}", flush=True)

    prof.report()


if __name__ == "__main__":
    main()
