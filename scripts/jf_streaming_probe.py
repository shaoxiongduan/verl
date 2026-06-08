"""
Streaming-window Jacobi probe.

Hypothesis: if we never let the cascade 'reset' to a fresh [first_correct |
random] block (which causes the bad iter-0 accept=1 every call), but instead
maintain a sliding window of size N where only the new tail positions are
refilled with random each iter, then EVERY iter looks like cllm's 'iter 1+'
regime (k3 accepts 5-6 tokens per iter consistently).

Algorithm:
  - Prefill prompt; initial window = [first_correct | random_(N-1)]
  - while not done:
      forward(window) → logits, accepted = leading prefix-match + 1
      commit accepted tokens to output (advance KV cache)
      slide window forward by `accepted`:
        new_window = [next_token | refresh_(N-1-accepted) | random_(accepted)]
      record per-iter accepted

Where:
  next_token  = argmax(logits at first mismatch)         [1 token]
  refresh     = argmax(logits at positions accepted..N-2) [N-1-accepted tokens]
  random_K    = K fresh draft init tokens (uniform / prompt_sample / mask)

This keeps the window size constant at N and never re-introduces a fully-random
suffix. Only `accepted` new positions per iter come in as random.

Compare per-iter accept distribution and overall TPF vs standard Jacobi.

Env: TPF_MODEL_PATH, TPF_DATA_PATH, NUM_PROMPTS, N_TOKEN_SEQ_LEN, DRAFT_INIT,
     MAX_NEW_TOKENS, OUT_JSON, LABEL.
"""
import os, sys, json, random
from pathlib import Path
sys.path.append(str(Path("/mnt/weka/home/hao.zhang/shao/JacobiForcing").resolve()))
sys.path.append(str(Path("/mnt/weka/home/hao.zhang/shao/JacobiForcing/JacobiForcing").resolve()))

import torch, pandas as pd
from transformers import Qwen2ForCausalLM, AutoTokenizer
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.cache_utils import DynamicCache

def _delete_false_key_value(self, n):
    for i in range(len(self.key_cache)):
        self.key_cache[i] = self.key_cache[i][..., :-n, :]
        self.value_cache[i] = self.value_cache[i][..., :-n, :]
DynamicCache.delete_false_key_value = _delete_false_key_value

MODEL_PATH = os.environ["TPF_MODEL_PATH"]
DATA_PATH = os.environ["TPF_DATA_PATH"]
NUM_PROMPTS = int(os.environ.get("NUM_PROMPTS", "10"))
N_BLOCK = int(os.environ.get("N_TOKEN_SEQ_LEN", "32"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "512"))
DRAFT_INIT = os.environ.get("DRAFT_INIT", "uniform").lower()
OUT_JSON = os.environ["OUT_JSON"]
LABEL = os.environ.get("LABEL", Path(MODEL_PATH).name)
MASK_TOKEN_ID = 151643

print(f"[{LABEL}] loading {MODEL_PATH}  N_BLOCK={N_BLOCK} DRAFT_INIT={DRAFT_INIT}")
model = Qwen2ForCausalLM.from_pretrained(
    MODEL_PATH, device_map="cuda", torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model.eval()
eos_id = tokenizer.eos_token_id
alt_eos_id = 151645

def sample_random_tok(generated_list):
    if DRAFT_INIT == "uniform":
        return random.randrange(tokenizer.vocab_size)
    elif DRAFT_INIT == "mask":
        return MASK_TOKEN_ID
    else:
        return random.choice(generated_list)

def run_streaming(prompt_ids):
    """Return (generated_token_list, per_iter_accepts, stop_reason)."""
    P = prompt_ids.shape[1]
    device = prompt_ids.device
    # Prefill: forward prompt to seed KV and get first_correct
    past_kv = DynamicCache()
    out = model(input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids),
                past_key_values=past_kv, use_cache=True, return_dict=True,
                output_hidden_states=False)
    past_kv = out.past_key_values
    logits = out.logits.float()
    first_correct = int(torch.argmax(logits[0, -1]).item())
    if first_correct == eos_id or first_correct == alt_eos_id:
        return [first_correct], [1], "eos"
    # Initial window: [first_correct | (N-1) random]
    generated_tokens = []
    rand_tail = [sample_random_tok(prompt_ids[0].tolist() + generated_tokens) for _ in range(N_BLOCK - 1)]
    window = torch.tensor([[first_correct] + rand_tail], dtype=torch.long, device=device)

    per_iter_accepts = []
    total_new = 0
    stop = None
    while total_new < MAX_NEW_TOKENS:
        # Forward on window (KV holds [prompt | committed])
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
        greedy = torch.argmax(logits[:, :-1, :], dim=-1)  # [1, N-1]
        mismatch = (window[:, 1:] != greedy)
        accepted_count = int((mismatch.cumsum(dim=-1) == 0).sum(dim=-1).item()) + 1
        # EOS handling: cap at first EOS in accepted prefix
        accepted_count_raw = accepted_count
        committed_so_far = window[0, :accepted_count].tolist()
        if eos_id in committed_so_far:
            accepted_count = committed_so_far.index(eos_id) + 1
            committed_so_far = committed_so_far[:accepted_count]
            stop = "eos"
        elif alt_eos_id in committed_so_far:
            accepted_count = committed_so_far.index(alt_eos_id) + 1
            committed_so_far = committed_so_far[:accepted_count]
            stop = "eos"
        generated_tokens.extend(committed_so_far)
        total_new += accepted_count
        per_iter_accepts.append(accepted_count)
        if stop == "eos":
            break

        # Trim KV: keep [prompt | committed]
        # past_kv had [prompt | committed_prev | window]; need to drop window and re-add committed
        # Actually past_kv currently has [prompt | committed_prev | window]. We want to delete
        # window and keep [prompt | committed_prev | window[:accepted]].
        cur_len = past_kv.get_seq_length()
        desired = P + total_new
        to_delete = cur_len - desired
        if to_delete > 0:
            past_kv.delete_false_key_value(to_delete)
        # Build next window:
        #   next_token = argmax(logits at first mismatch position = accepted_count - 1)
        #   refresh    = argmax(logits at positions [accepted_count_raw, N-1)) length N - 1 - accepted_count_raw
        #   random_tail = `accepted_count_raw` new random tokens to fill the window back to N
        # Note: we use accepted_count_raw for slicing logits to match what cllm does for KV alignment.
        next_token = int(torch.argmax(logits[0, accepted_count_raw - 1]).item())
        refresh_logits = logits[0, accepted_count_raw : -1, :]  # may be empty if accepted_count_raw >= N-1
        refresh = torch.argmax(refresh_logits, dim=-1).tolist() if refresh_logits.shape[0] > 0 else []
        rand_tail = [sample_random_tok(prompt_ids[0].tolist() + generated_tokens) for _ in range(accepted_count_raw)]
        new_window_list = [next_token] + refresh + rand_tail
        # Should be length N
        assert len(new_window_list) == N_BLOCK, f"window size mismatch: {len(new_window_list)} vs {N_BLOCK}"
        window = torch.tensor([new_window_list], dtype=torch.long, device=device)

    if stop is None:
        stop = "max_new_tokens" if total_new >= MAX_NEW_TOKENS else "max_iters"
    return generated_tokens, per_iter_accepts, stop

df = pd.read_parquet(DATA_PATH).head(NUM_PROMPTS)
random.seed(0)
all_results = []
with torch.no_grad():
    for idx, row in df.iterrows():
        msgs = list(row["prompt"]) if hasattr(row["prompt"], "__iter__") else row["prompt"]
        chat = [{"role": str(m["role"]), "content": str(m["content"])} for m in msgs]
        text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inp = tokenizer([text], return_tensors="pt").to(model.device)
        gen, accs, stop = run_streaming(inp["input_ids"])
        n_iters = len(accs)
        n_new = sum(accs)
        tpf = n_new / n_iters if n_iters else 0
        first3 = accs[:3] if len(accs) >= 3 else accs
        rest_mean = sum(accs[3:]) / max(1, len(accs) - 3) if len(accs) > 3 else 0
        print(f"  p{idx}: iters={n_iters} acc={n_new} TPF={tpf:.3f} stop={stop} "
              f"first3={first3} rest_mean={rest_mean:.2f}")
        all_results.append({"prompt_idx": int(idx), "per_iter_accepts": accs,
                            "stop": stop, "total_new": n_new})

out_p = Path(OUT_JSON)
out_p.parent.mkdir(parents=True, exist_ok=True)
with out_p.open("w") as f:
    json.dump({"label": LABEL, "N_BLOCK": N_BLOCK, "DRAFT_INIT": DRAFT_INIT,
               "results": all_results}, f)
print(f"[{LABEL}] wrote -> {OUT_JSON}")
