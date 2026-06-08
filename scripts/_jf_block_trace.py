"""Trace block-decode Jacobi inner-iter accept counts on k3.

Uses JacobiForcing's jacobi_forward_greedy semantics but with a per-inner-iter
trace hook. Goal: verify the claim that k3 has ~6 mean accepted-tokens at the
first inner iter of each fresh K-block (block_size=32 typical).

Env:
  TPF_MODEL_PATH   (required)
  TPF_DATA_PATH    (default OMI val)
  NUM_PROMPTS      (default 8)
  N_TOKEN_SEQ_LEN  (default 32 to match k3's block size)
  MAX_NEW_TOKENS   (default 512)
  TRACE_OUT_JSON   (required)
  DRAFT_INIT       (default prompt_sample; mask or uniform also supported)
"""
from __future__ import annotations
import json, os, random, sys, time
from pathlib import Path
from typing import Optional

root = Path("/mnt/weka/home/hao.zhang/shao/JacobiForcing")
sys.path.insert(0, str(root))

import torch
import pandas as pd
from transformers import Qwen2ForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

# Importing this monkey-patches DynamicCache.delete_false_key_value
from modeling.cllm2_qwen2_modeling_kv_terminate_on_eos_improved import jacobi_forward_greedy  # noqa

MODEL_PATH = os.environ["TPF_MODEL_PATH"]
DATA_PATH = os.environ.get("TPF_DATA_PATH", "/mnt/weka/home/hao.zhang/shao/verl/data/openmathinstruct2/val.parquet")
NUM_PROMPTS = int(os.environ.get("NUM_PROMPTS", "8"))
N = int(os.environ.get("N_TOKEN_SEQ_LEN", "32"))
MAX_NEW = int(os.environ.get("MAX_NEW_TOKENS", "512"))
TRACE_OUT = os.environ["TRACE_OUT_JSON"]
DRAFT_INIT = os.environ.get("DRAFT_INIT", "prompt_sample")
MASK_TOKEN_ID = int(os.environ.get("MASK_TOKEN_ID", "151643"))
MAX_BLOCKS = int(os.environ.get("MAX_BLOCKS", "256"))

print(f"=== JF block-decode TRACE recorder ===")
for k in ("MODEL_PATH","DATA_PATH","NUM_PROMPTS","N","MAX_NEW","TRACE_OUT","DRAFT_INIT"):
    print(f"  {k}={eval(k) if k!='N' else N}")


def jacobi_greedy_traced(model, input_ids, past_key_values, n_token_seq_len, eos_id, prefill_phase, prefill_drafted_n_gram=None):
    """Greedy block-Jacobi with per-inner-iter trace. Returns (past_kv, first_correct_token,
    accepted_n_gram, iter_count, inner_iters_trace).

    inner_iters_trace: list of dicts, one per inner iter, with input_draft / greedy_tokens / num_accepted.
    """
    if prefill_phase:
        attention_mask = torch.ones_like(input_ids)
        inputs_embeds = model.model.embed_tokens(input_ids)
        if past_key_values is None:
            past_key_values = DynamicCache()
        past_seen = past_key_values.get_seq_length()
        cache_pos = torch.arange(past_seen, past_seen + inputs_embeds.shape[1], device=input_ids.device)
        position_ids = cache_pos.unsqueeze(0)
        mask_kwargs = dict(config=model.config, input_embeds=inputs_embeds, attention_mask=attention_mask, cache_position=cache_pos, past_key_values=past_key_values)
        causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
        if model.model.has_sliding_layers:
            causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
        hidden = inputs_embeds
        pos_embeds = model.model.rotary_emb(hidden, position_ids)
        for layer in model.model.layers:
            hidden = layer(hidden, attention_mask=causal_mask_mapping[layer.attention_type], position_ids=position_ids, past_key_value=past_key_values, use_cache=True, cache_position=cache_pos, position_embeddings=pos_embeds)[0]
        hidden = model.model.norm(hidden)
        logits = model.lm_head(hidden).float()
        prefill_drafted = torch.argmax(logits[:, -n_token_seq_len-1:-1, :], dim=-1)
        first_correct = prefill_drafted[0]
        past_key_values.delete_false_key_value(n_token_seq_len)
        return past_key_values, first_correct, prefill_drafted, 0, []

    # ---- generation ----
    out = input_ids
    accepted_n_gram = out.clone()
    total_accepted = 0
    itr = 0
    inner_trace = []
    while total_accepted < n_token_seq_len:
        itr += 1
        input_snapshot = out.clone().detach().tolist()[0]
        inputs_embeds = model.model.embed_tokens(out)
        attention_mask = torch.ones_like(out, device=input_ids.device)
        past_seen = past_key_values.get_seq_length()
        cache_pos = torch.arange(past_seen, past_seen + out.shape[1], device=inputs_embeds.device)
        position_ids = cache_pos.unsqueeze(0)
        mask_kwargs = dict(config=model.config, input_embeds=inputs_embeds, attention_mask=attention_mask, cache_position=cache_pos, past_key_values=past_key_values)
        causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
        if model.model.has_sliding_layers:
            causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
        hidden = inputs_embeds
        pos_embeds = model.model.rotary_emb(hidden, position_ids)
        for layer in model.model.layers[: model.config.num_hidden_layers]:
            hidden = layer(hidden, attention_mask=causal_mask_mapping[layer.attention_type], position_ids=position_ids, past_key_value=past_key_values, use_cache=True, cache_position=cache_pos, position_embeddings=pos_embeds)[0]
        hidden = model.model.norm(hidden)
        logits = model.lm_head(hidden).float()
        greedy = torch.argmax(logits[:, :-1, :], dim=-1)  # predicts position i+1 for each i
        # entropy of model's predictions at each position
        logp = torch.log_softmax(logits[:, :-1, :], dim=-1)
        ent = -(logp.exp() * logp).sum(dim=-1)
        mismatch = (out[:, 1:] != greedy)
        accepted = (mismatch.cumsum(dim=-1) == 0).sum(dim=-1) + 1
        num_accepted_raw = int(accepted[0])
        num_accepted = num_accepted_raw
        if eos_id is not None:
            in_prefix = (out[0, :num_accepted_raw] == eos_id)
            if in_prefix.any():
                first_eos = int(torch.nonzero(in_prefix)[0])
                num_accepted = first_eos + 1
        L = out.shape[1]
        # record THIS inner iter
        inner_trace.append({
            "iter": itr,
            "input_draft": input_snapshot,
            "greedy_pred": greedy[0].tolist(),
            "entropy": ent[0].tolist(),
            "num_accepted_raw": num_accepted_raw,
            "num_accepted": num_accepted,
            "L": L,
        })
        if num_accepted > 0:
            accepted_n_gram[:, total_accepted:total_accepted+num_accepted] = out[:, :num_accepted].clone()
        total_accepted += num_accepted
        # EOS early-exit
        if eos_id is not None and (out[0, :num_accepted] == eos_id).any():
            cur = past_key_values.get_seq_length(); to_del = max(0, cur - total_accepted)
            if to_del > 0: past_key_values.delete_false_key_value(to_del)
            return past_key_values, torch.full((1,1), eos_id, device=out.device, dtype=out.dtype), accepted_n_gram[:, :total_accepted], itr, inner_trace
        has_rejected = num_accepted_raw < L
        if has_rejected:
            past_key_values.delete_false_key_value(L - num_accepted_raw)
            next_token = torch.argmax(logits[:, num_accepted_raw-1, :], dim=-1, keepdim=True)
            if eos_id is not None and next_token.item() == eos_id:
                accepted_n_gram[:, total_accepted:total_accepted+1] = next_token
                total_accepted += 1
                cur = past_key_values.get_seq_length(); to_del = max(0, cur - total_accepted)
                if to_del > 0: past_key_values.delete_false_key_value(to_del)
                return past_key_values, next_token, accepted_n_gram[:, :total_accepted], itr, inner_trace
            out = next_token
            q_probs_rem = logits[:, num_accepted_raw:-1, :]
            if q_probs_rem.shape[1] > 0:
                q_sampled = torch.argmax(q_probs_rem, dim=-1)
                out = torch.cat((out, q_sampled), dim=-1)
        else:
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            accepted_n_gram[:, total_accepted:total_accepted+1] = next_token
            total_accepted += 1
            if eos_id is not None and next_token.item() == eos_id:
                cur = past_key_values.get_seq_length(); to_del = max(0, cur - total_accepted)
                if to_del > 0: past_key_values.delete_false_key_value(to_del)
                return past_key_values, next_token, accepted_n_gram[:, :total_accepted], itr, inner_trace
    return past_key_values, next_token, accepted_n_gram[:, :total_accepted], itr, inner_trace


def main():
    df = pd.read_parquet(DATA_PATH)
    records = df.to_dict(orient="records")[:NUM_PROMPTS]
    print(f"Loading model from {MODEL_PATH} ...")
    t0 = time.perf_counter()
    model = Qwen2ForCausalLM.from_pretrained(MODEL_PATH, device_map="cuda", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model.eval()
    print(f"Loaded in {time.perf_counter() - t0:.1f}s")
    eos_id = tokenizer.eos_token_id
    alt_eos = 151645

    all_results = []
    random.seed(0)
    for idx, row in enumerate(records):
        messages = list(row["prompt"])
        chat = [{"role": str(m["role"]), "content": str(m["content"])} for m in messages]
        text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inp = tokenizer([text], return_tensors="pt").to(model.device)
        input_ids = inp["input_ids"]
        prompt_len = input_ids.shape[1]

        total_new = 0; calls = 0; past_kv = None
        first_correct = None; prefill_drafted = None; generated = input_ids
        blocks = []
        prev_len = prompt_len
        stop = None

        while True:
            gen_part = generated[0, prompt_len:]
            if (gen_part == eos_id).any() or (gen_part == alt_eos).any():
                stop = "eos"; break
            if total_new >= MAX_NEW:
                stop = "max_new"; break
            if calls >= MAX_BLOCKS:
                stop = "max_blocks"; break

            if calls == 0:
                # prefill: build noisy draft from prompt tail
                q = []
                for _ in range(N):
                    if DRAFT_INIT == "uniform":
                        tok = random.randrange(tokenizer.vocab_size)
                    elif DRAFT_INIT == "mask":
                        tok = MASK_TOKEN_ID
                    else:
                        tok = random.choice(generated[0].tolist())
                    q.append(torch.tensor([tok], dtype=torch.long, device=model.device).unsqueeze(0))
                draft = torch.cat(q, dim=1)
                full_input = torch.cat((input_ids, draft), dim=-1)
                past_kv, first_correct, prefill_drafted, _, _ = jacobi_greedy_traced(
                    model, full_input, None, N, eos_id, prefill_phase=True)
                generated = input_ids
                calls += 1
                continue

            if calls == 1:
                in_ids = prefill_drafted
            else:
                q = []
                for _ in range(N - 1):
                    if DRAFT_INIT == "uniform":
                        tok = random.randrange(tokenizer.vocab_size)
                    elif DRAFT_INIT == "mask":
                        tok = MASK_TOKEN_ID
                    else:
                        tok = random.choice(generated[0].tolist())
                    q.append(torch.tensor([tok], dtype=torch.long, device=model.device).unsqueeze(0))
                q_t = torch.cat(q, dim=1)
                in_ids = torch.cat((first_correct.view(1, -1), q_t), dim=-1)

            past_kv, first_correct, accepted, iter_used, inner = jacobi_greedy_traced(
                model, in_ids, past_kv, N, eos_id, prefill_phase=False)
            generated = torch.cat((generated, accepted), dim=-1)
            calls += 1
            added = generated.shape[1] - prev_len
            if added > 0: total_new += added
            prev_len = generated.shape[1]
            # Record block summary + inner trace
            blocks.append({
                "block_idx": calls - 2,  # 0-based skipping prefill call
                "input_first_inner": inner[0]["input_draft"] if inner else None,
                "n_committed": int(accepted.shape[1]),
                "iter_count": iter_used,
                "first_iter_accepted_raw": inner[0]["num_accepted_raw"] if inner else None,
                "first_iter_accepted": inner[0]["num_accepted"] if inner else None,
                "inner": inner,
            })

        total_new -= 1  # subtract prefill bonus
        all_results.append({
            "prompt_idx": idx,
            "prompt": text[:300],
            "total_new_tokens": total_new,
            "n_blocks": len(blocks),
            "stop_reason": stop,
            "blocks": blocks,
            "tpf_overall": total_new / max(1, sum(b["iter_count"] for b in blocks)),
            "first_iter_accepts": [b["first_iter_accepted"] for b in blocks if b.get("first_iter_accepted") is not None],
        })
        print(f"  [{idx+1}/{len(records)}] n_blocks={len(blocks)} new_toks={total_new} "
              f"mean_first_iter_acc={(sum(all_results[-1]['first_iter_accepts'])/max(1,len(all_results[-1]['first_iter_accepts']))):.3f}")

    out_p = Path(TRACE_OUT); out_p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(all_results, out_p.open("w"))
    # Aggregate
    all_first = [v for r in all_results for v in r["first_iter_accepts"]]
    import statistics
    if all_first:
        print(f"\nMean first-iter accepted across all blocks (N={len(all_first)}): {statistics.mean(all_first):.3f}")
        print(f"Median first-iter accepted: {statistics.median(all_first):.3f}")
    print(f"Wrote {TRACE_OUT}")


if __name__ == "__main__":
    main()
