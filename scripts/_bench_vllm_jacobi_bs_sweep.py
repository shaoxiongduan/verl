"""BS sweep for vLLM Jacobi spec-decode (JF Math 7B) vs vLLM AR (same model).
Mirrors _bench_vllm_dflash_bs_sweep.py so output CSVs are mergeable for plotting."""
from __future__ import annotations
import argparse, csv, gc, json, os, sys, time

# Patch must be installed BEFORE importing vllm.LLM. Path-fixed below.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vllm_jacobi_patch  # noqa


def run_one(mode: str, bs: int, args) -> dict:
    from vllm import LLM, SamplingParams
    import torch

    llm_kwargs = dict(
        model=args.target, dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=args.gpu_mem_util,
        enforce_eager=False,
        max_num_seqs=bs,
    )
    if mode == "jacobi":
        llm_kwargs["speculative_config"] = {
            "method": "ngram",
            "num_speculative_tokens": args.K,
            "prompt_lookup_min": 2,
            "prompt_lookup_max": 4,  # ignored by JacobiProposer but required by schema
        }
    print(f"\n>>> mode={mode}  BS={bs}  K={args.K} ...", flush=True)
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
    print(f"<<< mode={mode}  BS={bs}  wall={dt:.2f}s  out={n_tok}  TPS={tps:.1f}",
          flush=True)
    row = dict(mode=mode, bs=bs, wall_s=round(dt, 3),
               output_tokens=n_tok, tps=round(tps, 2))
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True,
                   help="JF Math 7B local snapshot dir")
    p.add_argument("--prompts_jsonl", required=True)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--K", type=int, default=32, help="Jacobi block length")
    p.add_argument("--bs_list", default="1,2,4,8,16,32,64")
    p.add_argument("--modes", default="jacobi,ar")
    p.add_argument("--gpu_mem_util", type=float, default=0.85)
    p.add_argument("--output_csv", required=True)
    args = p.parse_args()

    # Enable jacobi patch globally — it's a no-op when mode=ar (no spec config).
    vllm_jacobi_patch.enable_jacobi_spec_decode(K=args.K)

    bs_list = [int(x) for x in args.bs_list.split(",")]
    modes = args.modes.split(",")
    rows = []
    for mode in modes:
        for bs in bs_list:
            rows.append(run_one(mode, bs, args))

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mode", "bs", "wall_s",
                                          "output_tokens", "tps"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nCSV -> {args.output_csv}", flush=True)
    print("\n=== SUMMARY ===")
    for r in rows:
        print(f"  {r['mode']:>7s}  BS={r['bs']:>3d}  TPS={r['tps']:>8.1f}  "
              f"wall={r['wall_s']:>6.1f}s  out={r['output_tokens']}")


if __name__ == "__main__":
    main()
