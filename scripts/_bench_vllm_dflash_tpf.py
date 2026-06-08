"""vLLM DFlash TPF/TPS bench on the same eval_prompts_tpf.jsonl prompts as
the JF Jacobi vLLM bench. Mirrors _bench_vllm_jacobi_bs.py args."""
from __future__ import annotations
import argparse, json, os, sys, time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="Qwen/Qwen3-8B")
    p.add_argument("--drafter", default="z-lab/Qwen3-8B-DFlash-b16")
    p.add_argument("--prompts_jsonl", required=True)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--max_num_seqs", type=int, required=True)
    p.add_argument("--num_speculative_tokens", type=int, default=15)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--gpu_mem_util", type=float, default=0.85)
    p.add_argument("--attention_backend", default="flash_attn")
    p.add_argument("--no_spec", action="store_true",
                   help="Skip DFlash — run plain AR baseline.")
    args = p.parse_args()

    # Accumulate SpecDecodingStats observations into a global.
    spec_acc = {"num_drafts": 0, "num_draft_tokens": 0, "num_accepted_tokens": 0}
    from vllm.v1.spec_decode.metrics import SpecDecodingStats
    _orig = SpecDecodingStats.observe_draft
    def _patched(self, num_draft_tokens, num_accepted_tokens):
        _orig(self, num_draft_tokens, num_accepted_tokens)
        spec_acc["num_drafts"] += 1
        spec_acc["num_draft_tokens"] += num_draft_tokens
        spec_acc["num_accepted_tokens"] += num_accepted_tokens
    SpecDecodingStats.observe_draft = _patched

    from vllm import LLM, SamplingParams

    prompts_raw = [json.loads(l) for l in open(args.prompts_jsonl)]
    print(f"target={args.target}  drafter={args.drafter}", flush=True)
    print(f"loaded {len(prompts_raw)} prompts | max_num_seqs={args.max_num_seqs} "
          f"| K={args.num_speculative_tokens} | spec={not args.no_spec}",
          flush=True)

    llm_kwargs = dict(
        model=args.target, dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=args.gpu_mem_util,
        enforce_eager=False,
        max_num_seqs=args.max_num_seqs,
    )
    if not args.no_spec:
        llm_kwargs["speculative_config"] = {
            "method": "dflash",
            "model": args.drafter,
            "num_speculative_tokens": args.num_speculative_tokens,
        }
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", args.attention_backend.upper())

    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    sp = SamplingParams(temperature=args.temperature, max_tokens=args.max_new_tokens)
    chat_texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": pdata["input"]}],
            tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        for pdata in prompts_raw
    ]
    # warmup
    _ = llm.generate(chat_texts[:1], SamplingParams(max_tokens=32), use_tqdm=False)
    # reset counters after warmup
    spec_acc["num_drafts"] = 0
    spec_acc["num_draft_tokens"] = 0
    spec_acc["num_accepted_tokens"] = 0

    t0 = time.time()
    outs = llm.generate(chat_texts, sp, use_tqdm=False)
    dt = time.time() - t0
    n_tok = sum(len(o.outputs[0].token_ids) for o in outs)

    print(f"\n==== RESULT  BS_cap={args.max_num_seqs}  spec={not args.no_spec} ====",
          flush=True)
    print(f"  wall={dt:.2f}s  output_tokens={n_tok}  TPS={n_tok/dt:.1f}", flush=True)
    if not args.no_spec and spec_acc["num_drafts"] > 0:
        nd = spec_acc["num_drafts"]
        nat = spec_acc["num_accepted_tokens"]
        ndt = spec_acc["num_draft_tokens"]
        accept_rate = nat / ndt if ndt else 0.0
        # TPF = output_tokens / num_target_forwards; in v1 spec decode the
        # target runs once per draft iter, so num_target_forwards == num_drafts.
        tpf = n_tok / nd
        # mean_accept = avg accepted spec tokens per draft iter (excludes bonus)
        mean_accept = nat / nd
        print(f"  num_drafts={nd}  num_draft_tok={ndt}  num_accepted={nat}",
              flush=True)
        print(f"  mean_accept_per_draft={mean_accept:.3f}  accept_rate={accept_rate:.3f}",
              flush=True)
        print(f"  TPF (tok/target_forward) = {tpf:.3f}", flush=True)


if __name__ == "__main__":
    main()
