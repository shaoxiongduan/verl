"""Transformers-backend DFlash TPF bench on the same eval_prompts_tpf.jsonl
prompts as the JF Jacobi TPF bench. Engine-agnostic TPF = mean(accept) + 1."""
from __future__ import annotations
import argparse, json, os, time
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="Qwen/Qwen3-8B")
    p.add_argument("--drafter", default="z-lab/Qwen3-8B-DFlash-b16")
    p.add_argument("--prompts_jsonl", required=True)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--block_size", type=int, default=None,
                   help="Override drafter block_size. Default: drafter.block_size (16 for b16).")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--attn_impl", default="sdpa",
                   help="sdpa | flash_attention_2 | eager")
    p.add_argument("--output_jsonl", default=None)
    args = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from dflash.model import DFlashDraftModel, dflash_generate

    device = torch.device("cuda:0")
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

    print(f"target={args.target}  drafter={args.drafter}  attn={args.attn_impl}",
          flush=True)
    target = AutoModelForCausalLM.from_pretrained(
        args.target, attn_implementation=args.attn_impl, dtype=torch.bfloat16,
    ).to(device).eval()
    drafter = DFlashDraftModel.from_pretrained(
        args.drafter, attn_implementation=args.attn_impl, dtype=torch.bfloat16,
    ).to(device).eval()
    block_size = args.block_size if args.block_size is not None else drafter.block_size
    print(f"block_size={block_size}  mask_token_id={drafter.mask_token_id}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.target)
    prompts_raw = [json.loads(l) for l in open(args.prompts_jsonl)]
    print(f"loaded {len(prompts_raw)} prompts", flush=True)

    # warmup (1 short prompt)
    warm_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hi"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    warm_ids = tokenizer.encode(warm_text, return_tensors="pt").to(device)
    _ = dflash_generate(drafter, target=target, input_ids=warm_ids,
                        max_new_tokens=16, stop_token_ids=[tokenizer.eos_token_id],
                        temperature=args.temperature, block_size=block_size,
                        return_stats=False)

    all_rows = []
    total_drafts = 0
    total_accepted = 0
    total_output_tokens = 0
    t_total0 = time.time()

    for idx, pdata in enumerate(prompts_raw):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": pdata["input"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
        t0 = time.time()
        out = dflash_generate(
            drafter, target=target, input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            stop_token_ids=[tokenizer.eos_token_id],
            temperature=args.temperature, block_size=block_size,
            return_stats=True,
        )
        dt = time.time() - t0
        accs = list(out.acceptance_lengths)
        n_drafts = len(accs)
        n_accept = int(sum(accs))
        n_out = int(out.num_output_tokens)
        # TPF: tokens generated per target forward.
        # Each iter: 1 target verify forward → (accept + 1) tokens.
        # So TPF = n_out / n_drafts = mean(accept) + 1.
        tpf = n_out / max(n_drafts, 1)
        total_drafts += n_drafts
        total_accepted += n_accept
        total_output_tokens += n_out
        row = dict(
            idx=idx, n_input=int(out.num_input_tokens), n_out=n_out,
            n_drafts=n_drafts, n_accept=n_accept,
            mean_accept=float(np.mean(accs)) if accs else 0.0,
            tpf=tpf, wall_s=dt,
        )
        all_rows.append(row)
        print(f"[{idx+1:2d}/{len(prompts_raw)}] n_out={n_out:4d} drafts={n_drafts:4d} "
              f"accepted={n_accept:5d} mean_accept={row['mean_accept']:.2f} "
              f"TPF={tpf:.3f} wall={dt:.2f}s", flush=True)

    t_total = time.time() - t_total0

    overall_tpf = total_output_tokens / max(total_drafts, 1)
    overall_mean_accept = total_accepted / max(total_drafts, 1)
    overall_tps = total_output_tokens / max(t_total, 1e-6)
    per_prompt_tpf_mean = float(np.mean([r["tpf"] for r in all_rows]))
    per_prompt_tpf_std = float(np.std([r["tpf"] for r in all_rows]))

    print()
    print(f"==== AGGREGATE  prompts={len(all_rows)}  block_size={block_size} ====")
    print(f"  total_output_tokens = {total_output_tokens}")
    print(f"  total_drafts        = {total_drafts}")
    print(f"  total_accepted_spec = {total_accepted}")
    print(f"  overall TPF (out_tok/draft) = {overall_tpf:.3f}")
    print(f"  overall mean_accept         = {overall_mean_accept:.3f}")
    print(f"  per-prompt TPF mean = {per_prompt_tpf_mean:.3f} ± {per_prompt_tpf_std:.3f}")
    print(f"  wall = {t_total:.1f}s   TPS = {overall_tps:.1f}")

    if args.output_jsonl:
        os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)
        with open(args.output_jsonl, "w") as f:
            for r in all_rows:
                f.write(json.dumps(r) + "\n")
        print(f"per-prompt rows -> {args.output_jsonl}")


if __name__ == "__main__":
    main()
