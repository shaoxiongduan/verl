"""Vanilla AR (transformers .generate) on the same eval_prompts_tpf.jsonl."""
from __future__ import annotations
import argparse, json, time
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="Qwen/Qwen3-8B")
    p.add_argument("--prompts_jsonl", required=True)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--attn_impl", default="flash_attention_2")
    args = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device("cuda:0")
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

    print(f"target={args.target}  attn={args.attn_impl}", flush=True)
    target = AutoModelForCausalLM.from_pretrained(
        args.target, attn_implementation=args.attn_impl, dtype=torch.bfloat16,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.target)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    prompts_raw = [json.loads(l) for l in open(args.prompts_jsonl)]
    print(f"loaded {len(prompts_raw)} prompts", flush=True)

    # warmup
    w = tokenizer.apply_chat_template(
        [{"role":"user","content":"Hi"}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    w_ids = tokenizer.encode(w, return_tensors="pt").to(device)
    _ = target.generate(w_ids, max_new_tokens=16, do_sample=False)

    total_out = 0
    t0 = time.time()
    rows = []
    for idx, pdata in enumerate(prompts_raw):
        text = tokenizer.apply_chat_template(
            [{"role":"user","content":pdata["input"]}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        ids = tokenizer.encode(text, return_tensors="pt").to(device)
        n_in = ids.shape[1]
        t = time.time()
        out = target.generate(
            ids, max_new_tokens=args.max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        dt = time.time() - t
        n_new = int(out.shape[1] - n_in)
        total_out += n_new
        tps = n_new / max(dt, 1e-6)
        rows.append(dict(idx=idx, n_in=n_in, n_new=n_new, wall_s=dt, tps=tps))
        print(f"[{idx+1:2d}/{len(prompts_raw)}] n_new={n_new:4d} wall={dt:.2f}s tps={tps:.1f}",
              flush=True)
    t_total = time.time() - t0
    print()
    print(f"==== AR AGGREGATE  prompts={len(rows)} ====")
    print(f"  total_output_tokens = {total_out}")
    print(f"  wall = {t_total:.1f}s   overall TPS = {total_out / t_total:.1f}")
    print(f"  per-prompt TPS mean = {np.mean([r['tps'] for r in rows]):.1f} "
          f"± {np.std([r['tps'] for r in rows]):.1f}")


if __name__ == "__main__":
    main()
