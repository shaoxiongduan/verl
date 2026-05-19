"""Stress-test the consistency forward at near-production scale to surface
shape/index bugs that the smoke test misses.
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/mnt/weka/home/hao.zhang/shao/verl/scripts")
from consistency.loss import compute_consistency_loss

JF = ("/mnt/weka/home/hao.zhang/.cache/huggingface/hub/"
      "models--JacobiForcing--JacobiForcing_Coder_7B_v1/snapshots/"
      "81815b050f535c622153b5f6df38efc71326f938")


def main():
    os.environ["CONSISTENCY_DEBUG"] = "1"
    tok = AutoTokenizer.from_pretrained(JF, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    pad_id = int(tok.pad_token_id)

    model = AutoModelForCausalLM.from_pretrained(
        JF, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="flash_attention_2",
    )
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)

    # 4 prompts each with ~1k token response → block_size=32, T=32 → Lmax ~3k
    prompts = ["Write add."] * 4
    responses = ["def add(a, b):\n    return a + b\n" * 30] * 4   # ~250 tokens each

    prompt_ids = [
        torch.tensor(tok.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=True, add_generation_prompt=True),
            dtype=torch.long)
        for p in prompts
    ]
    response_ids = [
        torch.tensor(tok(r, add_special_tokens=False).input_ids, dtype=torch.long)
        for r in responses
    ]
    print("prompt_lens:", [p.numel() for p in prompt_ids], flush=True)
    print("response_lens:", [r.numel() for r in response_ids], flush=True)

    loss, metrics = compute_consistency_loss(
        model=model,
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        block_size=32,
        pad_id=pad_id,
        max_pairs=16,
        seed=0,
    )
    print(f"loss={loss.item()}  metrics={metrics}", flush=True)
    loss.backward()
    print("backward OK", flush=True)


if __name__ == "__main__":
    main()
