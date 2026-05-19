"""Run the JacobiForcing paper's custom `jacobi_forward_greedy` decoder on a
given Qwen2-arch HF checkpoint and score against HumanEval.

Adapted from
  research/JacobiForcing/JacobiForcing/jacobi_forcing_inference_humaneval.py

Differences from the original:
  - --model and --tokenizer CLI args (defaults match the paper)
  - imports from `_improved.py` (the `_continuous_drafting` filename in the
    original script refers to a file that's not in the public repo; the
    `_improved` variant exposes the same `jacobi_forward_greedy`)
  - reads our parquet (data/humaneval/val.parquet), scores via reward_code_assert
  - emits eval_passk/jacobi_greedy/<tag>.json
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
from tqdm import tqdm

# Make the JF modeling module importable.
JF_ROOT = Path(__file__).resolve().parents[1] / "research" / "JacobiForcing"
sys.path.insert(0, str(JF_ROOT))
from modeling.cllm2_qwen2_modeling_kv_terminate_on_eos_improved_continuous_drafting import (  # noqa: E402
    jacobi_forward_greedy,
)

# Make our reward fn importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reward_code_assert import _run_one  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument(
        "--tokenizer",
        default=(
            "/mnt/weka/home/hao.zhang/.cache/huggingface/hub/models--Qwen--"
            "Qwen2.5-Coder-7B-Instruct/snapshots/c03e6d358207e414f1eca0bb1891e29f1db0e242"
        ),
    )
    p.add_argument("--tag", required=True)
    p.add_argument("--val_file", default="data/humaneval/val.parquet")
    p.add_argument("--out_dir", default="eval_passk/jacobi_greedy")
    p.add_argument("--n_token_seq_len", type=int, default=64)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--max_calls", type=int, default=1024)
    p.add_argument("--timeout_s", type=int, default=15)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--limit", type=int, default=0, help="Cap on number of problems")
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    from transformers import AutoTokenizer, Qwen2ForCausalLM

    Qwen2ForCausalLM.jacobi_forward_greedy = jacobi_forward_greedy

    print(f"tokenizer: {args.tokenizer}\nmodel:     {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    # Use SDPA rather than flash_attention_2 — the flash-attn wheel ABI
    # depends on the exact torch/cuda combo used to build the wheel, and our
    # isolated .venv_jf installs torch+cu126 which doesn't pair with the
    # PyPI flash-attn 2.8.3 wheel. SDPA gives the same greedy outputs.
    model = Qwen2ForCausalLM.from_pretrained(
        args.model,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    eos_id = tokenizer.eos_token_id
    alt_eos_id = 151645

    t = pq.read_table(args.val_file).to_pydict()
    n = len(t["prompt"])
    if args.limit > 0:
        n = min(n, args.limit)

    n_token_seq_len = args.n_token_seq_len
    passes = 0
    results = []
    t0 = time.time()

    for idx in tqdm(range(n)):
        text = tokenizer.apply_chat_template(t["prompt"][idx], tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        input_ids_in = model_inputs["input_ids"]
        attention_mask = torch.full_like(input_ids_in, 1, device=model.device)

        prev_len = input_ids_in.shape[1]
        prompt_len = prev_len
        total_new_tokens = 0
        calls = 0
        prefill_phase = True
        generated_ids = input_ids_in
        prefill_drafted_n_gram = None
        first_correct_token = None
        past_key_values = None
        stop_reason = None

        while True:
            generated_part = generated_ids[0, prompt_len:]
            hit_eos = False
            if eos_id is not None:
                hit_eos = (generated_part == eos_id).any().item()
            if not hit_eos:
                hit_eos = (generated_part == alt_eos_id).any().item()
            if hit_eos:
                stop_reason = "eos"
                break
            if total_new_tokens >= args.max_new_tokens:
                stop_reason = "max_new_tokens"
                break
            if calls >= args.max_calls:
                stop_reason = "max_calls"
                break

            if prefill_phase:
                q_sampled = [
                    torch.tensor([random.choice(generated_ids[0].tolist())], dtype=torch.long, device=model.device).unsqueeze(0)
                    for _ in range(n_token_seq_len)
                ]
                prefill_draft_token_ids = torch.cat(q_sampled, dim=1)
                prefill_input_ids = torch.cat((input_ids_in, prefill_draft_token_ids), dim=-1)
                past_key_values, first_correct_token, prefill_drafted_n_gram, _ = model.jacobi_forward_greedy(
                    input_ids=prefill_input_ids,
                    attention_mask=attention_mask,
                    past_key_values=None,
                    use_cache=True,
                    prefill_phase=True,
                    n_token_seq_len=n_token_seq_len,
                    tokenizer=tokenizer,
                    eos_token_id=eos_id,
                )
                prefill_phase = False
                generated_ids = input_ids_in
                input_ids = None
            else:
                if calls == 1:
                    input_ids = prefill_drafted_n_gram
                else:
                    q_sampled = [
                        torch.tensor([random.choice(generated_ids[0].tolist())], dtype=torch.long, device=model.device).unsqueeze(0)
                        for _ in range(n_token_seq_len - 1)
                    ]
                    q_sampled = torch.cat(q_sampled, dim=1)
                    input_ids = torch.cat((first_correct_token.view(1, -1), q_sampled), dim=-1)
                past_key_values, first_correct_token, accepted_n_gram, _ = model.jacobi_forward_greedy(
                    input_ids=input_ids,
                    attention_mask=None,
                    past_key_values=past_key_values,
                    use_cache=True,
                    prefill_phase=False,
                    n_token_seq_len=n_token_seq_len,
                    tokenizer=tokenizer,
                    eos_token_id=eos_id,
                )
                generated_ids = torch.cat((generated_ids, accepted_n_gram), dim=-1)

            calls += 1
            added = generated_ids.shape[1] - prev_len
            if added > 0:
                total_new_tokens += added
            prev_len = generated_ids.shape[1]

        gen_str = tokenizer.decode(generated_ids[0, prompt_len:], skip_special_tokens=False)
        gt = t["reward_model"][idx]["ground_truth"]
        acc = _run_one((gen_str, gt, args.timeout_s, None))
        passes += int(acc)
        results.append({"task_id": t["extra_info"][idx]["task_id"], "acc": acc, "stop": stop_reason})

    elapsed = time.time() - t0
    print(f"\n=== {args.tag} ===  pass@1 = {passes}/{n} = {passes/n:.4f}  ({elapsed:.1f}s)", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.tag}.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "model": args.model,
                "tokenizer": args.tokenizer,
                "tag": args.tag,
                "n": n,
                "pass": passes,
                "pass_rate": passes / n,
                "elapsed_sec": elapsed,
                "per_task": results,
            },
            f,
            indent=2,
        )
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
