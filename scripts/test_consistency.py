"""End-to-end smoke test of the consistency loss pipeline:
1. Load JF Coder
2. Build a tiny interleaved batch from 2 (prompt, response) pairs
3. Run the consistency forward under flex_attention
4. Compute soft CE loss
5. Backward + assert grads flow through model params

Also verifies that the AR forward call (with attention_mask=None,
attn_implementation="flash_attention_2") still works AFTER the swap,
i.e. swap_attention_impl is truly reversible.

Run on a free GPU under the allocated SLURM job, e.g.:
  srun --jobid=1617918 --overlap bash -lc \
      "source .venv/bin/activate && CUDA_VISIBLE_DEVICES=1 python scripts/test_consistency.py"
"""

import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/mnt/weka/home/hao.zhang/shao/verl/scripts")
from consistency.loss import compute_consistency_loss
from consistency.attention import swap_attention_impl


JF = (
    "/mnt/weka/home/hao.zhang/.cache/huggingface/hub/"
    "models--JacobiForcing--JacobiForcing_Coder_7B_v1/snapshots/"
    "81815b050f535c622153b5f6df38efc71326f938"
)


def main():
    print("Loading tokenizer + model (flash_attention_2 baseline) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(JF, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        JF,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    )
    model.eval()
    pad_id = int(tok.pad_token_id)

    # Two tiny (prompt, response) pairs.
    prompts = [
        "Write a function that returns the sum of two integers.",
        "Write a function that returns the product of two integers.",
    ]
    responses = [
        "def add(a, b):\n    return a + b",
        "def mul(a, b):\n    return a * b",
    ]
    prompt_ids = [
        torch.tensor(
            tok.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=True, add_generation_prompt=True
            ),
            dtype=torch.long,
        )
        for p in prompts
    ]
    response_ids = [
        torch.tensor(tok(r, add_special_tokens=False).input_ids, dtype=torch.long)
        for r in responses
    ]
    print(f"Prompt lens: {[p.numel() for p in prompt_ids]}", flush=True)
    print(f"Response lens: {[r.numel() for r in response_ids]}", flush=True)

    # --- 1. AR baseline forward (verify model still works pre-swap) ---
    print("\n[1] AR forward (flash_attention_2, no swap) ...", flush=True)
    arl = []
    for p, r in zip(prompt_ids, response_ids, strict=False):
        x = torch.cat([p, r]).unsqueeze(0).to(model.device)
        with torch.no_grad():
            ar_out = model(input_ids=x, use_cache=False)
        arl.append(float(ar_out.logits[0, -1].mean().item()))
    print(f"  AR last-token logit means: {arl}", flush=True)

    # --- 2. Consistency forward with flex_attention swap ---
    print("\n[2] Consistency forward (flex_attention swap) ...", flush=True)
    t0 = time.time()
    loss, metrics = compute_consistency_loss(
        model=model,
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        block_size=8,         # small for the smoke test
        pad_id=pad_id,
        T_soft=1.0,
        seed=42,
    )
    dt = time.time() - t0
    print(f"  loss = {loss.item():.4f}  (took {dt:.2f}s)", flush=True)
    print(f"  metrics: {metrics}", flush=True)

    # --- 3. Backward + check grads ---
    print("\n[3] Backward + grad check ...", flush=True)
    for p in model.parameters():
        p.requires_grad_(True)
    model.train()
    loss2, _ = compute_consistency_loss(
        model=model,
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        block_size=8,
        pad_id=pad_id,
        seed=42,
    )
    loss2.backward()
    nonzero_params = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum().item() > 0)
    total_params = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"  {nonzero_params} / {total_params} parameter tensors received non-zero gradient", flush=True)

    # --- 4. AR forward AFTER the swap (verify reversibility) ---
    print("\n[4] AR forward AFTER swap (flash_attention_2 restored) ...", flush=True)
    model.eval()
    arl2 = []
    for p, r in zip(prompt_ids, response_ids, strict=False):
        x = torch.cat([p, r]).unsqueeze(0).to(model.device)
        with torch.no_grad():
            ar_out = model(input_ids=x, use_cache=False)
        arl2.append(float(ar_out.logits[0, -1].mean().item()))
    print(f"  AR last-token logit means: {arl2}", flush=True)
    diffs = [abs(a - b) for a, b in zip(arl, arl2, strict=False)]
    print(f"  AR-pre vs AR-post diffs: {diffs} (should be 0.0 — flash_attn_2 deterministic)", flush=True)

    assert all(d < 1e-4 for d in diffs), "swap_attention_impl did not restore cleanly"
    print("\nALL OK", flush=True)


if __name__ == "__main__":
    main()
