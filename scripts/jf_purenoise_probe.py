"""
Probe: how many tokens does iter 2 accept when the window is initialized to
PURE NOISE (no first_correct seed)? Compare to cllm-style init
([first_correct | random]).

User hypothesis: cons-trained models (k3 especially) only saw pure-noise blocks
during training, so they handle the iter-1-after-pure-noise regime best. The
iter-2 accept count under pure-noise init should be MUCH higher than under
first_correct+random init (which seeds a clean prefix at position 0,
contaminating the cascade with OOD partial-clean context).

For each prompt, we measure both setups separately, running just 2 iters in
each setup so the comparison is clean.

Setup A — pure noise init:
  window_0 = [random_32]
  forward → preds_0 (length 32, preds[i] is model's prediction for window pos i)
  cascade update: window_1 = preds_0  (just take the predictions as the next draft)
  forward window_1 → preds_1
  accept_iter2 = (# leading positions where window_1[i] == preds_1[i-1] for i>=1) + 1

Setup B — first_correct + random init (cllm convention):
  Forward prompt to get first_correct = argmax at last prompt token.
  window_0 = [first_correct | random_31]
  forward → preds_0
  accept_iter1 = leading matches + 1
  refresh from rejected tail; next_token from first mismatch.
  window_1 = [next_token | refresh] (shrinks to length 32-accept_iter1)
  forward window_1 → preds_1
  accept_iter2 = leading matches + 1

We report (accept_iter1, accept_iter2) for both setups.

Env: TPF_MODEL_PATH, TPF_DATA_PATH, NUM_PROMPTS, N_TOKEN_SEQ_LEN, OUT_JSON, LABEL.
"""
import os, sys, json, random
from pathlib import Path
sys.path.append(str(Path("/mnt/weka/home/hao.zhang/shao/JacobiForcing").resolve()))
sys.path.append(str(Path("/mnt/weka/home/hao.zhang/shao/JacobiForcing/JacobiForcing").resolve()))

import torch, pandas as pd
from transformers import Qwen2ForCausalLM, AutoTokenizer
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.cache_utils import DynamicCache

def _del(self, n):
    for i in range(len(self.key_cache)):
        self.key_cache[i] = self.key_cache[i][..., :-n, :]
        self.value_cache[i] = self.value_cache[i][..., :-n, :]
DynamicCache.delete_false_key_value = _del

MODEL_PATH = os.environ["TPF_MODEL_PATH"]
DATA_PATH = os.environ["TPF_DATA_PATH"]
NUM_PROMPTS = int(os.environ.get("NUM_PROMPTS", "20"))
N_BLOCK = int(os.environ.get("N_TOKEN_SEQ_LEN", "32"))
OUT_JSON = os.environ["OUT_JSON"]
LABEL = os.environ.get("LABEL", Path(MODEL_PATH).name)

print(f"[{LABEL}] loading  N_BLOCK={N_BLOCK}")
model = Qwen2ForCausalLM.from_pretrained(
    MODEL_PATH, device_map="cuda", torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model.eval()

def forward_with_kv(window, past_kv, P_offset):
    """Forward `window` with KV cache; window placed at positions starting at P_offset."""
    device = window.device
    inputs_embeds = model.model.embed_tokens(window)
    attn = torch.ones_like(window, device=device)
    past_seen = past_kv.get_seq_length()
    cache_pos = torch.arange(past_seen, past_seen + window.shape[1], device=device)
    pos_ids = cache_pos.unsqueeze(0)
    mask_kwargs = {"config": model.config, "input_embeds": inputs_embeds,
                   "attention_mask": attn, "cache_position": cache_pos,
                   "past_key_values": past_kv}
    cmm = {"full_attention": create_causal_mask(**mask_kwargs)}
    if model.model.has_sliding_layers:
        cmm["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
    hidden = inputs_embeds
    pos_emb = model.model.rotary_emb(hidden, pos_ids)
    for dl in model.model.layers[: model.config.num_hidden_layers]:
        hidden = dl(hidden, attention_mask=cmm[dl.attention_type],
                    position_ids=pos_ids, past_key_value=past_kv,
                    use_cache=True, cache_position=cache_pos,
                    position_embeddings=pos_emb)[0]
    hidden = model.model.norm(hidden)
    logits = model.lm_head(hidden).float()
    return logits

def leading_accept(window, greedy):
    """cllm convention: matches between window[1:] and greedy[:-1] from start; +1 for window[0]."""
    mismatch = (window[:, 1:] != greedy)
    return int((mismatch.cumsum(dim=-1) == 0).sum(dim=-1).item()) + 1

def prefill_prompt(prompt_ids):
    past_kv = DynamicCache()
    out = model(input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids),
                past_key_values=past_kv, use_cache=True, return_dict=True)
    past_kv = out.past_key_values
    logits = out.logits.float()
    first_correct = int(torch.argmax(logits[0, -1]).item())
    return past_kv, first_correct

df = pd.read_parquet(DATA_PATH).head(NUM_PROMPTS)
random.seed(0)
results = []

with torch.no_grad():
    for idx, row in df.iterrows():
        msgs = list(row["prompt"]) if hasattr(row["prompt"], "__iter__") else row["prompt"]
        chat = [{"role": str(m["role"]), "content": str(m["content"])} for m in msgs]
        text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inp = tokenizer([text], return_tensors="pt").to(model.device)
        prompt_ids = inp["input_ids"]
        P = prompt_ids.shape[1]

        # ---- Setup A: PURE NOISE init ----
        past_kv_a, first_correct = prefill_prompt(prompt_ids)
        rand_tokens = [random.randrange(tokenizer.vocab_size) for _ in range(N_BLOCK)]
        window_a0 = torch.tensor([rand_tokens], dtype=torch.long, device=model.device)
        logits_a0 = forward_with_kv(window_a0, past_kv_a, P)
        # accept_iter1 in pure-noise: count leading matches where preds[i-1] == window[i]
        greedy_a0 = torch.argmax(logits_a0[:, :-1, :], dim=-1)
        accept_a1 = leading_accept(window_a0, greedy_a0)
        # Cascade update: window_a1 = preds_a0
        # Note: cllm's standard refresh is more subtle; here we use the "full preds = next draft"
        # which is the canonical Jacobi step.  preds[i] = pred for window pos i, made from
        # context [prompt | window[0..i-1]]. Standard Jacobi: window_new[i] = preds[i].
        # But there's an indexing issue: logits[i] predicts position i+1. So preds[0] is the
        # pred for window pos 1, preds[i] is pred for window pos i+1. So window_new[0] is...
        # actually preds[-1] from the *prompt* position predicts window[0] (always AR[0]).
        # In cllm's accept, mismatch = (out[1:] != argmax(logits[:-1])) — so logits[i] for i
        # in [0, L-1] predicts out[i+1]. The pred for out[0] uses the last prompt logit (NOT
        # in our slice). So for the cascade update we use:
        #   window_new[0] = argmax(prompt's last logit) = AR[0] (already known from prefill)
        #   window_new[i] = argmax(logits[i-1]) for i in [1..N-1]
        # which is: window_new = [first_correct] + argmax(logits[:-1])  length N.
        preds_a0 = torch.argmax(logits_a0[:, :-1, :], dim=-1).squeeze(0).tolist()
        window_a1_list = [first_correct] + preds_a0  # length N
        window_a1 = torch.tensor([window_a1_list[:N_BLOCK]], dtype=torch.long, device=model.device)
        # Trim KV to just prompt
        past_kv_a.delete_false_key_value(N_BLOCK)
        logits_a1 = forward_with_kv(window_a1, past_kv_a, P)
        greedy_a1 = torch.argmax(logits_a1[:, :-1, :], dim=-1)
        accept_a2 = leading_accept(window_a1, greedy_a1)

        # ---- Setup B: first_correct + random init (cllm convention) ----
        past_kv_b, first_correct_b = prefill_prompt(prompt_ids)
        rand_31 = [random.randrange(tokenizer.vocab_size) for _ in range(N_BLOCK - 1)]
        window_b0 = torch.tensor([[first_correct_b] + rand_31], dtype=torch.long, device=model.device)
        logits_b0 = forward_with_kv(window_b0, past_kv_b, P)
        greedy_b0 = torch.argmax(logits_b0[:, :-1, :], dim=-1)
        accept_b1 = leading_accept(window_b0, greedy_b0)
        # Cascade update: same as cllm — next_token + refresh
        next_token = int(torch.argmax(logits_b0[0, accept_b1 - 1]).item())
        refresh_logits = logits_b0[0, accept_b1:-1, :]
        refresh = torch.argmax(refresh_logits, dim=-1).tolist() if refresh_logits.shape[0] > 0 else []
        window_b1_list = [next_token] + refresh  # shrinks each iter (cllm)
        window_b1 = torch.tensor([window_b1_list], dtype=torch.long, device=model.device)
        # Trim KV to prompt + accept_b1 (the accepted prefix stays in KV)
        past_kv_b.delete_false_key_value(N_BLOCK - accept_b1)
        logits_b1 = forward_with_kv(window_b1, past_kv_b, P + accept_b1)
        greedy_b1 = torch.argmax(logits_b1[:, :-1, :], dim=-1)
        accept_b2 = leading_accept(window_b1, greedy_b1)

        results.append({
            "prompt_idx": int(idx),
            "pureN_iter1_accept": accept_a1, "pureN_iter2_accept": accept_a2,
            "cllm_iter1_accept": accept_b1, "cllm_iter2_accept": accept_b2,
        })
        print(f"  p{idx}: pureN (iter1={accept_a1}, iter2={accept_a2})  "
              f"cllm (iter1={accept_b1}, iter2={accept_b2})")

from statistics import mean, stdev
print()
def m(k): return mean(r[k] for r in results)
print(f"[{LABEL}] AGGREGATE (n={len(results)}):")
print(f"  PURE-NOISE init:  iter1 mean={m('pureN_iter1_accept'):.2f}  iter2 mean={m('pureN_iter2_accept'):.2f}  total/2iters={m('pureN_iter1_accept')+m('pureN_iter2_accept'):.2f} → TPF={(m('pureN_iter1_accept')+m('pureN_iter2_accept'))/2:.3f}")
print(f"  CLLM init:        iter1 mean={m('cllm_iter1_accept'):.2f}  iter2 mean={m('cllm_iter2_accept'):.2f}  total/2iters={m('cllm_iter1_accept')+m('cllm_iter2_accept'):.2f} → TPF={(m('cllm_iter1_accept')+m('cllm_iter2_accept'))/2:.3f}")

out_p = Path(OUT_JSON)
out_p.parent.mkdir(parents=True, exist_ok=True)
with out_p.open("w") as f:
    json.dump({"label": LABEL, "N_BLOCK": N_BLOCK, "results": results}, f)
print(f"[{LABEL}] wrote -> {OUT_JSON}")
