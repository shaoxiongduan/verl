"""
Probe: does the model preserve a partially-correct draft, or does it undo good
tokens because it was trained on pure-noise contexts?

For each model:
  1. Generate K=32 AR tokens greedily from a prompt -> AR_tokens[0..31].
  2. For each K in [0, 1, 2, 4, 8, 16, 24]:
       draft = [AR_tokens[0..K-1] | uniform_random[K..31]]
       Run one forward pass on [prompt | draft].
       For position i in [0..31]:
         logit_at_i predicts token at position i+1.
         pred[i+1] = argmax(logit_at_i)
       Measure:
         - leading_match: length of leading prefix where pred[1..] matches AR.
           At iter 0 (K=0), this measures how good a single forward gets from random.
           At K>0, this measures how much the correct prefix helps.
         - per-pos preservation: for positions 1..K (where input was correct AR),
           does pred[i] == AR[i]? If NOT, model "undid" correct tokens.
         - per-pos extension: for positions K+1..31 (input was random),
           does pred[i] == AR[i]? Measures cascade extension.
  3. Compare across models. Hypothesis: cons-trained models have higher
     "undoing rate" because they were trained on pure noise.

Env:
  TPF_MODEL_PATH (required), TPF_DATA_PATH (required), NUM_PROMPTS (default 5),
  N_BLOCK (default 32), OUT_JSON (required).
"""
import os, sys, json, random
from pathlib import Path
sys.path.append(str(Path("/mnt/weka/home/hao.zhang/shao/JacobiForcing").resolve()))
sys.path.append(str(Path("/mnt/weka/home/hao.zhang/shao/JacobiForcing/JacobiForcing").resolve()))

import torch, pandas as pd
from transformers import Qwen2ForCausalLM, AutoTokenizer

MODEL_PATH = os.environ["TPF_MODEL_PATH"]
DATA_PATH = os.environ["TPF_DATA_PATH"]
NUM_PROMPTS = int(os.environ.get("NUM_PROMPTS", "5"))
N_BLOCK = int(os.environ.get("N_BLOCK", "32"))
OUT_JSON = os.environ["OUT_JSON"]
LABEL = os.environ.get("LABEL", Path(MODEL_PATH).name)

K_VALUES = [0, 1, 2, 4, 8, 16, 24]

print(f"[{LABEL}] loading {MODEL_PATH}")
model = Qwen2ForCausalLM.from_pretrained(
    MODEL_PATH, device_map="cuda", torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model.eval()

df = pd.read_parquet(DATA_PATH).head(NUM_PROMPTS)
random.seed(0)

all_results = []
with torch.no_grad():
    for pidx, row in df.iterrows():
        messages = list(row["prompt"]) if hasattr(row["prompt"], "__iter__") else row["prompt"]
        chat = [{"role": str(m["role"]), "content": str(m["content"])} for m in messages]
        text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        prompt_ids = inputs["input_ids"]
        P = prompt_ids.shape[1]

        # AR greedy reference for N_BLOCK + 1 tokens
        ar = prompt_ids.clone()
        for _ in range(N_BLOCK + 1):
            logits = model(ar).logits
            nxt = logits[0, -1].argmax(dim=-1).view(1, 1)
            ar = torch.cat([ar, nxt], dim=-1)
        ar_continuation = ar[0, P:P + N_BLOCK + 1].tolist()  # length N_BLOCK+1

        prompt_results = {"prompt_idx": int(pidx), "ar_continuation": ar_continuation, "K_results": {}}

        for K in K_VALUES:
            # Build draft of length N_BLOCK: first K are correct AR tokens; rest random
            draft = []
            for j in range(N_BLOCK):
                if j < K:
                    draft.append(ar_continuation[j])
                else:
                    draft.append(random.randrange(tokenizer.vocab_size))
            draft_t = torch.tensor([draft], dtype=torch.long, device=model.device)
            full = torch.cat([prompt_ids, draft_t], dim=-1)
            logits = model(full).logits  # [1, P+N, V]
            # logit at position P+i-1 predicts token at P+i. We want predictions for
            # positions in the draft block, i.e., predictions at j=0..N-1 where
            # pred[j] is from logit at position P-1+j and replaces draft[j].
            # Actually: standard Jacobi convention -- logit at position p predicts p+1.
            # Position P-1 (last prompt token) -> predicts draft[0]. So:
            # pred[0] = argmax(logits[0, P-1])
            # pred[j] = argmax(logits[0, P-1+j]) for j in [0..N-1]
            preds = logits[0, P - 1 : P - 1 + N_BLOCK].argmax(dim=-1).tolist()
            # Compare preds[j] vs ar_continuation[j] for j in [0..N-1]
            match_pos = [int(preds[j] == ar_continuation[j]) for j in range(N_BLOCK)]
            # Leading prefix match
            leading = 0
            for m in match_pos:
                if m: leading += 1
                else: break
            # Of correct-input positions (0..K-1), how many preds match AR?
            preserve_in_K = sum(match_pos[:K]) if K > 0 else 0
            # Of random-input positions (K..N-1), how many preds match AR?
            extend_after_K = sum(match_pos[K:])
            prompt_results["K_results"][K] = {
                "leading_match": leading,
                "preserve_in_K": preserve_in_K,
                "K": K,
                "extend_after_K": extend_after_K,
                "match_pos": match_pos,
                "preds": preds,
            }
        all_results.append(prompt_results)
        print(f"  p{pidx}: K=0 lead={prompt_results['K_results'][0]['leading_match']}  "
              f"K=8 lead={prompt_results['K_results'][8]['leading_match']}  "
              f"K=16 lead={prompt_results['K_results'][16]['leading_match']}")

out_p = Path(OUT_JSON)
out_p.parent.mkdir(parents=True, exist_ok=True)
with out_p.open("w") as f:
    json.dump({"label": LABEL, "model_path": MODEL_PATH, "N_BLOCK": N_BLOCK,
               "K_values": K_VALUES, "results": all_results}, f)
print(f"\n[{LABEL}] wrote -> {OUT_JSON}")
