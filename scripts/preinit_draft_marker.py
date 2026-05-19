"""Pre-initialize the draft-marker embedding row in a Qwen2.5-Coder ckpt.

Qwen2.5 stores reserved-but-unused token rows (ids 151665..152063) at literal
zero. The consistency-loss draft-marker injection adds `embed.weight[151665]`
on top of noisy positions — but with a zero row it contributes nothing, so
the marker has to be slowly grown by gradient before it conveys any signal.

This script copies a model dir to a new path, with row 151665 of the embedding
table replaced by small-norm random values. Optimizer state is independent
(verl will create a fresh optimizer on resume_mode=auto since the EXP_NAME is
new), so we don't need to touch anything else.

Usage:
  python scripts/preinit_draft_marker.py \
    --src <hf-ckpt-path> \
    --dst <new-hf-ckpt-path> \
    --marker_id 151665 \
    --std_match auto    # match std of named special tokens
"""
from __future__ import annotations
import argparse
import os
import shutil
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Source HF ckpt path.")
    p.add_argument("--dst", required=True, help="Destination HF ckpt path.")
    p.add_argument("--marker_id", type=int, default=151665)
    p.add_argument("--std_match", default="auto",
                   help="'auto' = use std of named special tokens; or a float.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--also_tie_lm_head", action="store_true",
                   help="If lm_head is untied, also zero out the lm_head row for marker_id "
                        "(so the model never gets pressured to *predict* the marker).")
    args = p.parse_args()

    if os.path.exists(args.dst):
        print(f"DST already exists: {args.dst}. Refusing to overwrite.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model from {args.src} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.src, torch_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(args.src, use_fast=True)
    embed = model.get_input_embeddings()
    H = embed.weight.shape[-1]
    print(f"Embed shape: {tuple(embed.weight.shape)}  hidden_size={H}", flush=True)

    cur_norm = float(embed.weight[args.marker_id].float().norm().item())
    print(f"Current marker row[{args.marker_id}] norm = {cur_norm:.6f}", flush=True)
    if cur_norm > 1e-3:
        print(f"  WARNING: row is already non-zero. Continuing anyway.", flush=True)

    # Match std of named special tokens (id range 151643..151664), so the marker
    # has a realistic init magnitude rather than overwhelming or under-impacting
    # the existing token-embedding norms.
    if args.std_match == "auto":
        special_rows = embed.weight[151643:151665].float()
        std = special_rows.std().item()
    else:
        std = float(args.std_match)
    print(f"Init std = {std:.6f}  (auto = std of named special tokens)", flush=True)

    gen = torch.Generator().manual_seed(args.seed)
    init_vec = torch.randn(H, generator=gen) * std
    new_norm = float(init_vec.norm().item())
    print(f"New marker row norm = {new_norm:.6f}", flush=True)

    with torch.no_grad():
        embed.weight.data[args.marker_id] = init_vec.to(embed.weight.dtype)

    # If lm_head is untied (i.e., not sharing weights with embed), match the
    # lm_head row to either zero (default) or the init vec.
    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None and lm_head.weight.data_ptr() != embed.weight.data_ptr():
        print("lm_head is UNTIED from embed; setting its row[marker_id] = 0", flush=True)
        with torch.no_grad():
            lm_head.weight.data[args.marker_id].zero_()
    else:
        print("lm_head shares weights with embed (tied); already in sync.", flush=True)

    os.makedirs(args.dst, exist_ok=True)
    print(f"Saving to {args.dst} ...", flush=True)
    model.save_pretrained(args.dst, safe_serialization=True)
    tok.save_pretrained(args.dst)
    # Copy chat_template.jinja if present (some HF versions don't save it via tokenizer)
    src_chat = os.path.join(args.src, "chat_template.jinja")
    if os.path.exists(src_chat):
        shutil.copy(src_chat, args.dst)
    print(f"\nDone. Marker row {args.marker_id} initialized with std={std:.4f}, norm={new_norm:.4f}.", flush=True)


if __name__ == "__main__":
    main()
