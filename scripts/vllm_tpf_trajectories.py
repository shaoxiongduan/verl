"""vLLM TPF benchmark counterpart of tpf_trajectories.py — runs the same
prompts under the vllm_jacobi_patch (ngram-slot hijack into JacobiProposer)
and reports per-prompt TPF in the same JSONL schema, so JF nanovllm and
vLLM Jacobi numbers can be compared on identical models / prompts / K.

Usage:
  K=32 TRAJ_PATH=/tmp/vt.jsonl python3 scripts/vllm_tpf_trajectories.py \
    --model <hf-dir> --prompts_jsonl <jsonl> --output_jsonl <jsonl> \
    --jacobi_block_len 32 --max_new_tokens 1024 \
    --temperature 0.6 --batch_size 16

CRITICAL: enable_jacobi_spec_decode MUST run at module level, NOT inside
main(). vLLM's EngineCore subprocess uses `spawn` and re-imports __main__;
the patch only takes effect there if it ran during that re-import.
"""
from __future__ import annotations
import argparse, json, os, sys, time

# IMPORTANT: install patches at module level so child EngineCore picks them up.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vllm_jacobi_patch  # noqa: E402

# Read K and traj path from env so they're available at module import time
# (argparse can't run yet — would break subprocess re-import).
_K = int(os.environ.get("JACOBI_K", "32"))
_TRAJ_PATH = os.environ.get("VLLM_TPF_TRAJ_PATH", "/tmp/vllm_tpf_traj.jsonl")
vllm_jacobi_patch.enable_jacobi_spec_decode(K=_K, traj_path=_TRAJ_PATH)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompts_jsonl", required=True)
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--jacobi_block_len", type=int, default=_K,
                   help="K = num speculative tokens. Must match JACOBI_K env.")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--gpu_mem_util", type=float, default=0.6)
    args = p.parse_args()

    assert args.jacobi_block_len == _K, (
        f"--jacobi_block_len ({args.jacobi_block_len}) must match JACOBI_K env "
        f"({_K}); patch already installed.")

    from vllm import LLM, SamplingParams  # noqa: E402

    prompts_raw = []
    with open(args.prompts_jsonl, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            prompts_raw.append(json.loads(line))
    print(f"Loaded {len(prompts_raw)} prompts", flush=True)

    print(f"Loading model: {args.model}", flush=True)
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=args.gpu_mem_util,
        enforce_eager=False,
        speculative_config={
            "method": "ngram",
            "num_speculative_tokens": args.jacobi_block_len,
            "prompt_lookup_min": 2,
            "prompt_lookup_max": 4,
        },
    )
    tokenizer = llm.get_tokenizer()

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    out_fp = open(args.output_jsonl, "w")

    # Pass stop_token_ids so vLLM truncates the spec-decode commit batch at the
    # first EOS occurrence (not just the last token of the batch). Without this,
    # when EOS lands mid-K=32 commit, the post-EOS positions get committed as
    # junk and the model loops past the natural end → inflated TPF + length cap.
    _stop_ids = []
    if tokenizer.eos_token_id is not None:
        _stop_ids.append(int(tokenizer.eos_token_id))
    # Qwen2.5 has both <|im_end|>=151645 (chat eos) and <|endoftext|>=151643 (pad).
    # Add both for safety; vLLM dedupes.
    for _sid in (151645, 151643):
        if _sid not in _stop_ids:
            _stop_ids.append(_sid)
    sp = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        stop_token_ids=_stop_ids,
    )
    print(f"[bench] using stop_token_ids={_stop_ids}", flush=True)

    # Send ALL prompts in one big batch so per_req_tpf[i] aligns with prompt i.
    # vLLM internally batches/schedules; this just hands the engine all reqs.
    chat_texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": pdata["input"]}],
            tokenize=False, add_generation_prompt=True,
        )
        for pdata in prompts_raw
    ]
    t0 = time.time()
    outs = llm.generate(chat_texts, sp, use_tqdm=True)
    dt = time.time() - t0
    print(f"\ngen_wall={dt:.1f}s for {len(prompts_raw)} prompts", flush=True)

    # Aggregate trajectories.
    stats = vllm_jacobi_patch.aggregate_trajectories()
    per_req = stats.get("per_req_tpf", []) if isinstance(stats, dict) else []

    # Write per-prompt row.
    total = 0
    for j, (pdata, out) in enumerate(zip(prompts_raw, outs, strict=False)):
        ntok = len(out.outputs[0].token_ids)
        completion = out.outputs[0].text
        per_req_tpf = per_req[j] if j < len(per_req) else None
        row = {
            "task_id": pdata.get("id"),
            "prompt": pdata["input"],
            "completion": completion,
            "num_tokens": ntok,
            "tpf": per_req_tpf,  # may be None if patch didn't fire
            "batch_idx": j,
        }
        out_fp.write(json.dumps(row) + "\n")
        total += 1

    out_fp.flush()
    out_fp.close()

    # Summary.
    print(f"\n=== vLLM TPF stats ({args.model}) ===")
    agg_tpf = stats.get("agg_tpf") if isinstance(stats, dict) else None
    if isinstance(agg_tpf, (int, float)):
        print(f"  agg_tpf:       {agg_tpf:.3f}")
    else:
        print(f"  agg_tpf:       {agg_tpf} (NO TRAJECTORY CAPTURED — patch likely didn't fire in EngineCore)")
    if per_req:
        mean_per_req = sum(per_req) / len(per_req)
        print(f"  per_req TPF mean: {mean_per_req:.3f} (n={len(per_req)})")
        print(f"  per_req TPF min:  {min(per_req):.2f}")
        print(f"  per_req TPF max:  {max(per_req):.2f}")
    print(f"  spec_iters: {stats.get('n_iters')}")
    print(f"  spec_tok_total: {stats.get('spec_tok_total')}")
    print(f"  trajectory files: {stats.get('n_files')}")

    print(f"\nWrote {total} rows -> {args.output_jsonl}", flush=True)


if __name__ == "__main__":
    main()
