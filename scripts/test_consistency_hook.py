"""Simulate verl's `forward_step` call to the consistency hook with a
realistic fake micro_batch (prompts, responses, attention_mask). Verifies:
- env-var activation path
- _extract_prompts_responses correctness with verl-style padded tensors
- maybe_add_consistency_loss returns a tensor with grad attached
"""

import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/mnt/weka/home/hao.zhang/shao/verl/scripts")
from consistency.verl_hook import maybe_add_consistency_loss

JF = (
    "/mnt/weka/home/hao.zhang/.cache/huggingface/hub/"
    "models--JacobiForcing--JacobiForcing_Coder_7B_v1/snapshots/"
    "81815b050f535c622153b5f6df38efc71326f938"
)


class FakeEngine:
    def __init__(self, module):
        self.module = module
        self._global_step = 0
        # No config — forces env-var path.


def main():
    os.environ["CONSISTENCY_ENABLE"] = "1"
    os.environ["CONSISTENCY_WEIGHT"] = "0.01"
    os.environ["CONSISTENCY_BLOCK_SIZE"] = "8"     # small for smoke test
    os.environ["CONSISTENCY_FRACTION"] = "1.0"
    os.environ["CONSISTENCY_PAD_ID"] = "151643"
    os.environ["CONSISTENCY_T_SOFT"] = "1.0"

    print("Loading model ...", flush=True)
    tok = AutoTokenizer.from_pretrained(JF, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    pad_id = int(tok.pad_token_id)

    model = AutoModelForCausalLM.from_pretrained(
        JF,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    )
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)

    # Build verl-style micro_batch: prompts (B, max_prompt_len) LEFT-padded,
    # responses (B, max_resp_len) RIGHT-padded, attention_mask (B, P+R).
    prompts_raw = [
        torch.tensor(tok.apply_chat_template(
            [{"role": "user", "content": "Write add."}],
            tokenize=True, add_generation_prompt=True), dtype=torch.long),
        torch.tensor(tok.apply_chat_template(
            [{"role": "user", "content": "Write mul."}],
            tokenize=True, add_generation_prompt=True), dtype=torch.long),
    ]
    responses_raw = [
        torch.tensor(tok("def add(a,b):\n    return a+b", add_special_tokens=False).input_ids, dtype=torch.long),
        torch.tensor(tok("def mul(a,b):\n    return a*b", add_special_tokens=False).input_ids, dtype=torch.long),
    ]
    Lp = max(p.numel() for p in prompts_raw)
    Lr = max(r.numel() for r in responses_raw)
    B = len(prompts_raw)
    prompts = torch.full((B, Lp), pad_id, dtype=torch.long)
    responses = torch.full((B, Lr), pad_id, dtype=torch.long)
    attn = torch.zeros((B, Lp + Lr), dtype=torch.long)
    for i, (p, r) in enumerate(zip(prompts_raw, responses_raw, strict=False)):
        # left-pad prompts (verl convention)
        prompts[i, Lp - p.numel():] = p
        responses[i, :r.numel()] = r
        attn[i, Lp - p.numel():Lp] = 1
        attn[i, Lp:Lp + r.numel()] = 1

    micro_batch = {
        "prompts": prompts,
        "responses": responses,
        "attention_mask": attn,
    }

    engine = FakeEngine(module=model)
    base_loss = torch.tensor(2.0, device=model.device, requires_grad=True)
    metrics = {}

    print("Calling maybe_add_consistency_loss with CONSISTENCY_ENABLE=1 ...", flush=True)
    new_loss = maybe_add_consistency_loss(engine, micro_batch, base_loss, metrics)
    print(f"  Loss: {base_loss.item():.4f} -> {new_loss.item():.4f}", flush=True)
    print(f"  Metrics: {metrics}", flush=True)
    assert new_loss.item() != base_loss.item(), "consistency loss did not change total loss"
    assert metrics.get("actor/cons_n_pos", 0) > 0, "no positions selected"

    print("Backward ...", flush=True)
    new_loss.backward()
    nonzero = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    total = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"  {nonzero}/{total} parameter tensors with non-zero gradient", flush=True)

    # Disable via env; verify no-op.
    os.environ["CONSISTENCY_ENABLE"] = "0"
    metrics2 = {}
    base_loss2 = torch.tensor(2.0, device=model.device, requires_grad=True)
    no_op = maybe_add_consistency_loss(engine, micro_batch, base_loss2, metrics2)
    assert no_op is base_loss2, "disabled hook should be a no-op"
    assert metrics2 == {}, f"disabled hook should leave metrics empty, got {metrics2}"
    print("Disabled-path no-op: OK", flush=True)
    print("\nALL OK", flush=True)


if __name__ == "__main__":
    main()
