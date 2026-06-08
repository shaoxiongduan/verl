"""Post-fix bench: TPF + TPS for trained ckpts with the FIXED vLLM Jacobi engine.

Sets `max_num_seqs` to truly cap concurrent BS. Writes per-prompt jsonl + summary line.
"""
from __future__ import annotations
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vllm_jacobi_patch  # noqa

_K = int(os.environ.get("JACOBI_K", "32"))
_TRAJ_PATH = os.environ.get("VLLM_TPF_TRAJ_PATH", "/tmp/vllm_postfix_traj.jsonl")
vllm_jacobi_patch.enable_jacobi_spec_decode(K=_K, traj_path=_TRAJ_PATH)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompts_jsonl", required=True)
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--max_num_seqs", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--gpu_mem_util", type=float, default=0.6)
    args = p.parse_args()

    from vllm import LLM, SamplingParams

    prompts_raw = [json.loads(l) for l in open(args.prompts_jsonl)]

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
    chat_texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p["input"]}],
            tokenize=False, add_generation_prompt=True,
        )
        for p in prompts_raw
    ]
    # Warmup
    _ = llm.generate(chat_texts[:1], SamplingParams(max_tokens=32), use_tqdm=False)

    sp = SamplingParams(temperature=args.temperature, max_tokens=args.max_new_tokens)
    t0 = time.time()
    outs = llm.generate(chat_texts, sp, use_tqdm=False)
    dt = time.time() - t0
    n_tok = sum(len(o.outputs[0].token_ids) for o in outs)

    stats = vllm_jacobi_patch.aggregate_trajectories()
    per_req = stats.get("per_req_tpf", []) if isinstance(stats, dict) else []

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    with open(args.output_jsonl, "w") as fp:
        for j, (pdata, out) in enumerate(zip(prompts_raw, outs, strict=False)):
            row = {
                "task_id": pdata.get("id"),
                "prompt": pdata["input"],
                "completion": out.outputs[0].text,
                "num_tokens": len(out.outputs[0].token_ids),
                "tpf": per_req[j] if j < len(per_req) else None,
                "batch_idx": j,
            }
            fp.write(json.dumps(row) + "\n")

    valid_tpf = [t for t in per_req if t is not None]
    mean_tpf = sum(valid_tpf) / len(valid_tpf) if valid_tpf else float("nan")
    tps = n_tok / dt if dt > 0 else 0
    print(f"\n[POSTFIX] model={os.path.basename(args.model)} T={args.temperature} BS_cap={args.max_num_seqs}", flush=True)
    print(f"[POSTFIX] wall={dt:.2f}s tokens={n_tok} TPS={tps:.1f} per_req_TPF_mean={mean_tpf:.3f} n={len(valid_tpf)}", flush=True)


if __name__ == "__main__":
    main()
