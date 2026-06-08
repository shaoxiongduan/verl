"""Capture per-prompt trajectory JSONLs for trajectory analysis.

Strategy: keep the patch's traj file open across prompts, then split by
byte-offset checkpoints around each generate() call.
"""
from __future__ import annotations
import argparse, json, os, sys, time, glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vllm_jacobi_patch  # noqa

_K = int(os.environ.get("JACOBI_K", "32"))
# IMPORTANT: install patches at module level so EngineCore subprocess
# (uses 'spawn' which re-imports __main__) picks them up.
_TRAJ_PATH = os.environ.get("VLLM_TRAJ_CAPTURE_PATH",
                            f"/tmp/_traj_capture_{os.environ.get('TRAJ_LABEL','default')}.jsonl")
vllm_jacobi_patch.enable_jacobi_spec_decode(K=_K, traj_path=_TRAJ_PATH)


def file_sizes(pattern):
    """Map of {path: bytes} for all files matching pattern."""
    return {p: os.path.getsize(p) for p in glob.glob(pattern) if os.path.isfile(p)}


def read_new_records(pattern, since):
    """Read JSONL records appended after `since` byte-offsets per file."""
    out = []
    for p in glob.glob(pattern):
        if not os.path.isfile(p):
            continue
        start = since.get(p, 0)
        try:
            with open(p, "rb") as fp:
                fp.seek(start)
                data = fp.read()
        except OSError:
            continue
        for line in data.split(b"\n"):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompts_jsonl", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--n_prompts", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--label", required=True)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    prompts_raw = [json.loads(l) for l in open(args.prompts_jsonl)][:args.n_prompts]

    # patch already installed at module level via _TRAJ_PATH env
    base_traj = _TRAJ_PATH
    # Clean stale per-PID files from prior runs (file handles in current process untouched)
    for tp in glob.glob(base_traj + ".*"):
        try: os.remove(tp)
        except OSError: pass
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model, dtype="bfloat16",
        max_model_len=4096, gpu_memory_utilization=0.6,
        enforce_eager=False, max_num_seqs=1,
        speculative_config={
            "method": "ngram", "num_speculative_tokens": _K,
            "prompt_lookup_min": 2, "prompt_lookup_max": 4,
        },
    )
    tokenizer = llm.get_tokenizer()
    sp = SamplingParams(temperature=args.temperature, max_tokens=args.max_new_tokens)

    # Warm-up (this will write its own iters; checkpoint AFTER to exclude them)
    _ = llm.generate(["warmup"], SamplingParams(max_tokens=8), use_tqdm=False)
    # Checkpoint file sizes after warmup so each prompt's iters are the delta.
    offsets = file_sizes(base_traj + ".*")

    for i, pdata in enumerate(prompts_raw):
        chat_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": pdata["input"]}],
            tokenize=False, add_generation_prompt=True,
        )
        outs = llm.generate([chat_text], sp, use_tqdm=False)
        completion = outs[0].outputs[0].text
        n_tok = len(outs[0].outputs[0].token_ids)

        # Read records appended since last checkpoint
        new_iters = read_new_records(base_traj + ".*", offsets)
        # Update offsets
        offsets = file_sizes(base_traj + ".*")
        # Filter: keep only single-req records (warmup uses bigger batches)
        iters = [r for r in new_iters if r.get("num_draft") and len(r["num_draft"]) == 1]

        per_prompt_out = os.path.join(args.out_dir, f"p{i:02d}__{args.label}.json")
        with open(per_prompt_out, "w") as fp:
            json.dump({
                "prompt_idx": i,
                "prompt": pdata["input"],
                "expected": pdata.get("expected_answer"),
                "completion": completion,
                "n_tokens": n_tok,
                "n_iters": len(iters),
                "tpf": n_tok / max(1, len(iters)),
                "model_label": args.label,
                "K": _K,
                "iters": iters,
            }, fp)
        print(f"[traj] {args.label} p{i:02d}: tok={n_tok} iters={len(iters)} tpf={n_tok/max(1,len(iters)):.3f}", flush=True)


if __name__ == "__main__":
    main()
