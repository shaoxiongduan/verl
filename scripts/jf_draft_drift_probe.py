"""
Per-iter draft drift probe.

For each model, run two inference modes (chunked / streaming) on a small set of
prompts. At every iter, record the draft tokens that the model SEES as input.
Then compute distributional statistics to measure how far each iter's draft
drifts from the training noise distribution (uniform random over vocab).

Hypothesis (user): training noise was uniform random. Iter-0 input matches this
distribution (ID). After iter 1+, the draft suffix becomes model predictions,
which are heavily biased toward AR-plausible tokens — OOD for cons training.

For chunked decoding: the draft distribution likely shifts a lot per iter
within a call (from random → mixed → mostly-AR-correct as block fills).

For streaming: in steady state, the draft distribution should reach a more
consistent OOD steady state.

Metrics per iter:
  - token entropy of the draft (uniform random ≈ log(vocab_size) ≈ 11.95)
  - fraction of draft tokens that are in TOP_K most common vocab (low for
    uniform random; high for AR-text since AR uses common words)
  - mean token id rank under model's clean LM prior (low = common; high = rare)
  - decoded text snippet (qualitative)
  - count of unique tokens (uniform random ≈ N; repetitive draft < N)

Env: TPF_MODEL_PATH, TPF_DATA_PATH, NUM_PROMPTS, MODE (chunked|streaming),
     N_TOKEN_SEQ_LEN, OUT_JSON, LABEL, MAX_ITERS.
"""
import os, sys, json, random, math
from pathlib import Path
from collections import Counter
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
NUM_PROMPTS = int(os.environ.get("NUM_PROMPTS", "5"))
N_BLOCK = int(os.environ.get("N_TOKEN_SEQ_LEN", "32"))
MAX_ITERS = int(os.environ.get("MAX_ITERS", "12"))
MODE = os.environ.get("MODE", "chunked").lower()  # chunked or streaming
OUT_JSON = os.environ["OUT_JSON"]
LABEL = os.environ.get("LABEL", Path(MODEL_PATH).name)

print(f"[{LABEL}] loading  N={N_BLOCK}  MODE={MODE}  MAX_ITERS={MAX_ITERS}")
model = Qwen2ForCausalLM.from_pretrained(
    MODEL_PATH, device_map="cuda", torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model.eval()
V = tokenizer.vocab_size

def forward_with_kv(window, past_kv):
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
    return model.lm_head(hidden).float()

def prefill_prompt(prompt_ids):
    past_kv = DynamicCache()
    out = model(input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids),
                past_key_values=past_kv, use_cache=True, return_dict=True)
    return out.past_key_values, int(torch.argmax(out.logits.float()[0, -1]).item())

df = pd.read_parquet(DATA_PATH).head(NUM_PROMPTS)
random.seed(0)

per_iter_drafts = []  # list of [iter_idx, draft_tokens_list]

with torch.no_grad():
    for idx, row in df.iterrows():
        msgs = list(row["prompt"]) if hasattr(row["prompt"], "__iter__") else row["prompt"]
        chat = [{"role": str(m["role"]), "content": str(m["content"])} for m in msgs]
        text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inp = tokenizer([text], return_tensors="pt").to(model.device)
        prompt_ids = inp["input_ids"]
        P = prompt_ids.shape[1]
        past_kv, first_correct = prefill_prompt(prompt_ids)
        rand31 = [random.randrange(V) for _ in range(N_BLOCK - 1)]
        window = torch.tensor([[first_correct] + rand31], dtype=torch.long, device=model.device)

        prompt_per_iter = []
        for t in range(MAX_ITERS):
            prompt_per_iter.append({"iter": t, "draft": window[0].tolist()})
            logits = forward_with_kv(window, past_kv)
            greedy = torch.argmax(logits[:, :-1, :], dim=-1)
            mismatch = (window[:, 1:] != greedy)
            accept = int((mismatch.cumsum(dim=-1) == 0).sum(dim=-1).item()) + 1
            if MODE == "chunked":
                # cllm shrinking window: out = [next_token | refresh]
                if accept >= window.shape[1]:
                    # All accepted, append next token and break (or extend? we just break)
                    break
                next_tok = int(torch.argmax(logits[0, accept - 1]).item())
                refresh_logits = logits[0, accept:-1, :]
                refresh = torch.argmax(refresh_logits, dim=-1).tolist() if refresh_logits.shape[0] > 0 else []
                new_window_list = [next_tok] + refresh
                past_kv.delete_false_key_value(window.shape[1] - accept)
                window = torch.tensor([new_window_list], dtype=torch.long, device=model.device)
            else:  # streaming
                # Keep window size N: [next_token | refresh | random_tail]
                accept_raw = accept
                next_tok = int(torch.argmax(logits[0, accept_raw - 1]).item())
                refresh_logits = logits[0, accept_raw:-1, :]
                refresh = torch.argmax(refresh_logits, dim=-1).tolist() if refresh_logits.shape[0] > 0 else []
                rand_tail = [random.randrange(V) for _ in range(accept_raw)]
                new_window_list = [next_tok] + refresh + rand_tail
                assert len(new_window_list) == N_BLOCK
                past_kv.delete_false_key_value(window.shape[1] - accept_raw)
                window = torch.tensor([new_window_list], dtype=torch.long, device=model.device)
            if window.shape[1] <= 1:
                break
        per_iter_drafts.append({"prompt_idx": int(idx), "iters": prompt_per_iter})

# Decode drafts and compute stats per iter
TOP_K = 1000  # top common tokens; we approximate by token id range (Qwen common tokens lower ids)
# A better proxy: use the model's lm_head bias or rough frequency, but let's just use entropy
# and qualitative inspection.

print()
print(f"=" * 90)
print(f"PER-ITER DRAFT STATS  [{LABEL} mode={MODE}]")
print(f"=" * 90)
print(f"{'iter':<5}{'mean H':>8}{'uniq/N':>8}{'mean text len':>16}{'sample tokens (first prompt)':>40}")
# Compute per iter across all prompts
from collections import defaultdict
by_iter = defaultdict(list)  # iter -> list of (tokens)
for prompt in per_iter_drafts:
    for it in prompt["iters"]:
        by_iter[it["iter"]].append(it["draft"])

uniform_H = math.log(V)
print(f"  (uniform random entropy bound ≈ {uniform_H:.2f})\n")

for it_idx in sorted(by_iter.keys())[:15]:
    drafts = by_iter[it_idx]
    if not drafts: continue
    # Mean unique fraction
    uniq_fracs = [len(set(d)) / len(d) for d in drafts if d]
    # Compute per-draft entropy approximation
    Hs = []
    text_lens = []
    for d in drafts:
        cnt = Counter(d)
        H = 0
        for c in cnt.values():
            p = c / len(d)
            H -= p * math.log(p)
        Hs.append(H)
        text = tokenizer.decode(d)
        text_lens.append(len(text))
    # Sample first prompt's draft text
    sample_text = tokenizer.decode(drafts[0]) if drafts else ""
    sample_text = sample_text.replace("\n", " ").replace("\r", " ")[:60]
    print(f"{it_idx:<5}{sum(Hs)/len(Hs):>8.2f}{sum(uniq_fracs)/len(uniq_fracs):>8.2f}{sum(text_lens)/len(text_lens):>16.1f}    {sample_text!r}")

# Save
out_p = Path(OUT_JSON)
out_p.parent.mkdir(parents=True, exist_ok=True)
with out_p.open("w") as f:
    json.dump({"label": LABEL, "mode": MODE, "N_BLOCK": N_BLOCK,
               "MAX_ITERS": MAX_ITERS, "per_iter_drafts": per_iter_drafts}, f)
print(f"\n[{LABEL}] wrote -> {OUT_JSON}")
