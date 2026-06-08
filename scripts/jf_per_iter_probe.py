"""
Run real Jacobi inference; log per-iter accept counts within each call.

Tests user's hypothesis: cons-trained models accept many tokens at iter 1
(when draft is fully random) but accept the same as others on subsequent iters.

We monkey-patch jacobi_forward_greedy to capture per-iter num_accepted before
returning, then aggregate across rollouts.

Env: same as jf_inference_trace.py + OUT_JSON for results.
"""
import os, sys, json, random
from pathlib import Path
sys.path.append(str(Path("/mnt/weka/home/hao.zhang/shao/JacobiForcing").resolve()))
sys.path.append(str(Path("/mnt/weka/home/hao.zhang/shao/JacobiForcing/JacobiForcing").resolve()))

import torch, pandas as pd
from transformers import Qwen2ForCausalLM, AutoTokenizer
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.cache_utils import DynamicCache

# Monkey-patch DynamicCache with the cllm helper (same as cllm modeling file).
def _delete_false_key_value(self, num_of_false_tokens):
    for layer_idx in range(len(self.key_cache)):
        self.key_cache[layer_idx] = self.key_cache[layer_idx][..., :-num_of_false_tokens, :]
        self.value_cache[layer_idx] = self.value_cache[layer_idx][..., :-num_of_false_tokens, :]
DynamicCache.delete_false_key_value = _delete_false_key_value

MODEL_PATH = os.environ["TPF_MODEL_PATH"]
DATA_PATH = os.environ["TPF_DATA_PATH"]
NUM_PROMPTS = int(os.environ.get("NUM_PROMPTS", "10"))
N_BLOCK = int(os.environ.get("N_TOKEN_SEQ_LEN", "32"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "512"))
MAX_CALLS = int(os.environ.get("MAX_CALLS", "1024"))
DRAFT_INIT = os.environ.get("DRAFT_INIT", "uniform").lower()
OUT_JSON = os.environ["OUT_JSON"]
LABEL = os.environ.get("LABEL", Path(MODEL_PATH).name)

# Patched jacobi_forward_greedy: same as cllm impl, but appends each iter's
# num_accepted to a global list `_PER_ITER_LOG`.
_PER_ITER_LOG = []

def jacobi_forward_greedy_logged(
    self, input_ids, attention_mask, past_key_values, use_cache, prefill_phase,
    n_token_seq_len, tokenizer, eos_token_id, **_kw,
):
    eos_id = eos_token_id
    eos_enabled = eos_id is not None
    if prefill_phase:
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            past_key_values=past_key_values, use_cache=True, return_dict=True,
        )
        past_key_values = outputs.past_key_values
        hidden_states = self.model.norm(outputs.last_hidden_state)
        logits = self.lm_head(hidden_states).float()
        prefill_drafted_n_gram = torch.argmax(logits[:, -n_token_seq_len-1:-1, :], dim=-1)
        first_correct_token = prefill_drafted_n_gram[0]
        if past_key_values is not None and n_token_seq_len > 0:
            past_key_values.delete_false_key_value(n_token_seq_len)
        return past_key_values, first_correct_token, prefill_drafted_n_gram, 0

    assert past_key_values is not None
    out, device = input_ids, input_ids.device
    accepted_n_gram = out.clone()  # preallocated full N_BLOCK
    total_accepted, itr = 0, 0
    per_iter_accepts_this_call = []

    while total_accepted < n_token_seq_len:
        itr += 1
        inputs_embeds = self.model.embed_tokens(out)
        attn = torch.ones_like(out, device=out.device)
        past_seen = past_key_values.get_seq_length()
        cache_pos = torch.arange(past_seen, past_seen + out.shape[1], device=device)
        pos_ids = cache_pos.unsqueeze(0)
        mask_kwargs = {"config": self.config, "input_embeds": inputs_embeds,
                       "attention_mask": attn, "cache_position": cache_pos,
                       "past_key_values": past_key_values}
        cmm = {"full_attention": create_causal_mask(**mask_kwargs)}
        if self.model.has_sliding_layers:
            cmm["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
        hidden = inputs_embeds
        pos_emb = self.model.rotary_emb(hidden, pos_ids)
        for dl in self.model.layers[: self.model.config.num_hidden_layers]:
            hidden = dl(hidden, attention_mask=cmm[dl.attention_type],
                        position_ids=pos_ids, past_key_value=past_key_values,
                        use_cache=True, cache_position=cache_pos,
                        position_embeddings=pos_emb)[0]
        hidden = self.model.norm(hidden)
        logits = self.lm_head(hidden).float()
        greedy = torch.argmax(logits[:, :-1, :], dim=-1)
        mismatch = (out[:, 1:] != greedy)
        accepted = (mismatch.cumsum(dim=-1) == 0).sum(dim=-1) + 1
        L = out.shape[1]
        num_accepted_raw = int(accepted[0])
        num_accepted = num_accepted_raw
        if eos_enabled:
            eos_in = (out[0, :num_accepted_raw] == eos_id)
            if eos_in.any():
                num_accepted = int(torch.nonzero(eos_in, as_tuple=False)[0]) + 1
        if num_accepted > 0:
            accepted_n_gram[:, total_accepted:total_accepted+num_accepted] = out[:, :num_accepted].clone()
        per_iter_accepts_this_call.append(num_accepted)
        total_accepted += num_accepted
        if eos_enabled and (out[0, :num_accepted] == eos_id).any():
            cur = past_key_values.get_seq_length()
            todel = max(0, cur - total_accepted)
            if todel: past_key_values.delete_false_key_value(todel)
            _PER_ITER_LOG.append(per_iter_accepts_this_call)
            return past_key_values, torch.full((1,1), eos_id, device=device, dtype=out.dtype), accepted_n_gram[:, :total_accepted], itr
        has_rej = (num_accepted_raw < L)
        if has_rej:
            past_key_values.delete_false_key_value(out.shape[1]-num_accepted_raw)
            next_tok = torch.argmax(logits[:, num_accepted_raw-1, :], dim=-1, keepdim=True)
            if eos_enabled and next_tok.item() == eos_id:
                accepted_n_gram[:, total_accepted:total_accepted+1] = next_tok
                total_accepted += 1
                per_iter_accepts_this_call[-1] += 1
                cur = past_key_values.get_seq_length()
                todel = max(0, cur - total_accepted)
                if todel: past_key_values.delete_false_key_value(todel)
                _PER_ITER_LOG.append(per_iter_accepts_this_call)
                return past_key_values, next_tok, accepted_n_gram[:, :total_accepted], itr
            out = next_tok
            q_probs_rem = logits[:, num_accepted_raw:-1, :]
            if q_probs_rem.shape[1] > 0:
                q_samp = torch.argmax(q_probs_rem, dim=-1)
                out = torch.cat((out, q_samp), dim=-1)
        else:
            next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            accepted_n_gram[:, total_accepted:total_accepted+1] = next_tok
            total_accepted += 1
            per_iter_accepts_this_call[-1] += 1
            if eos_enabled and next_tok.item() == eos_id:
                cur = past_key_values.get_seq_length()
                todel = max(0, cur - total_accepted)
                if todel: past_key_values.delete_false_key_value(todel)
                _PER_ITER_LOG.append(per_iter_accepts_this_call)
                return past_key_values, next_tok, accepted_n_gram[:, :total_accepted], itr
    _PER_ITER_LOG.append(per_iter_accepts_this_call)
    return past_key_values, next_tok, accepted_n_gram[:, :total_accepted], itr

Qwen2ForCausalLM.jacobi_forward_greedy = jacobi_forward_greedy_logged

print(f"[{LABEL}] loading {MODEL_PATH}")
model = Qwen2ForCausalLM.from_pretrained(
    MODEL_PATH, device_map="cuda", torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model.eval()
eos_id = tokenizer.eos_token_id
alt_eos_id = 151645

df = pd.read_parquet(DATA_PATH).head(NUM_PROMPTS)
random.seed(0)

all_call_iters = []  # list of per-call lists; each inner list is per-iter accept counts

with torch.no_grad():
    for idx, row in df.iterrows():
        msgs = list(row["prompt"]) if hasattr(row["prompt"], "__iter__") else row["prompt"]
        chat = [{"role": str(m["role"]), "content": str(m["content"])} for m in msgs]
        text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        input_ids = inputs["input_ids"]
        attn = torch.full_like(input_ids, 1, device=model.device)
        prompt_len = input_ids.shape[1]
        generated = input_ids
        prev_len = prompt_len
        total_new = 0
        calls = 0
        prefill = True
        past_kv = None
        first_correct = None
        prefill_ngram = None
        prompt_call_start = len(_PER_ITER_LOG)
        stop = None
        while True:
            gen_part = generated[0, prompt_len:]
            hit = (gen_part == eos_id).any().item() if eos_id else False
            if not hit: hit = (gen_part == alt_eos_id).any().item()
            if hit: stop = "eos"; break
            if total_new >= MAX_NEW_TOKENS: stop = "max_new_tokens"; break
            if calls >= MAX_CALLS: stop = "max_calls"; break
            if prefill:
                q = []
                for _ in range(N_BLOCK):
                    if DRAFT_INIT == "uniform": t = random.randrange(tokenizer.vocab_size)
                    elif DRAFT_INIT == "mask": t = 151643
                    else: t = random.choice(generated[0].tolist())
                    q.append(torch.tensor([t], dtype=torch.long, device=model.device).unsqueeze(0))
                prefill_draft = torch.cat(q, dim=1)
                prefill_in = torch.cat((input_ids, prefill_draft), dim=-1)
                past_kv, first_correct, prefill_ngram, itc = model.jacobi_forward_greedy(
                    input_ids=prefill_in, attention_mask=attn, past_key_values=None,
                    use_cache=True, prefill_phase=True, n_token_seq_len=N_BLOCK,
                    tokenizer=tokenizer, eos_token_id=eos_id)
                prefill = False
                generated = input_ids
            else:
                if calls == 1:
                    cur_in = prefill_ngram
                else:
                    q = []
                    for _ in range(N_BLOCK - 1):
                        if DRAFT_INIT == "uniform": t = random.randrange(tokenizer.vocab_size)
                        elif DRAFT_INIT == "mask": t = 151643
                        else: t = random.choice(generated[0].tolist())
                        q.append(torch.tensor([t], dtype=torch.long, device=model.device).unsqueeze(0))
                    qs = torch.cat(q, dim=1)
                    cur_in = torch.cat((first_correct.view(1, -1), qs), dim=-1)
                past_kv, first_correct, accepted_ng, itc = model.jacobi_forward_greedy(
                    input_ids=cur_in, attention_mask=None, past_key_values=past_kv,
                    use_cache=True, prefill_phase=False, n_token_seq_len=N_BLOCK,
                    tokenizer=tokenizer, eos_token_id=eos_id)
                generated = torch.cat((generated, accepted_ng), dim=-1)
            calls += 1
            added = generated.shape[1] - prev_len
            if added > 0: total_new += added
            prev_len = generated.shape[1]
        # Collect this prompt's per-call per-iter accepts
        prompt_calls = _PER_ITER_LOG[prompt_call_start:]
        all_call_iters.append({"prompt_idx": int(idx), "calls": prompt_calls, "stop": stop, "total_new": total_new})
        sum_iters = sum(len(c) for c in prompt_calls)
        sum_acc = sum(sum(c) for c in prompt_calls)
        tpf = sum_acc / sum_iters if sum_iters else 0
        first_iter_accepts = [c[0] for c in prompt_calls if c]
        rest_iter_accepts = [a for c in prompt_calls for a in c[1:]] if any(len(c)>1 for c in prompt_calls) else []
        from statistics import mean as _m
        print(f"  p{idx}: calls={len(prompt_calls)} iters={sum_iters} acc={sum_acc} TPF={tpf:.3f} "
              f"first-iter-avg={_m(first_iter_accepts) if first_iter_accepts else 0:.2f} "
              f"rest-iter-avg={_m(rest_iter_accepts) if rest_iter_accepts else 0:.2f}")

out_p = Path(OUT_JSON)
out_p.parent.mkdir(parents=True, exist_ok=True)
with out_p.open("w") as f:
    json.dump({"label": LABEL, "model_path": MODEL_PATH, "N_BLOCK": N_BLOCK,
               "DRAFT_INIT": DRAFT_INIT, "prompts": all_call_iters}, f)
print(f"[{LABEL}] wrote -> {OUT_JSON}")
