"""
Probe: do lucky-correct tokens deep in the block get PRESERVED or OVERWRITTEN as
the cascade progresses?

Simulate Jacobi iters explicitly:
  iter 0 input  = [seed | uniform_random_(N-1)]
  iter 0 output = preds_0 = argmax of one forward pass
  iter 1 input  = [seed | preds_0[1..N-1]]  (standard Jacobi: draft suffix replaced
                                              with previous-iter predictions)
  iter 1 output = preds_1
  ... and so on for T iters.

For each position i in the block, mark per iter:
  - "correct"  = preds_t[i] == AR[i]
  - "first_correct_iter" = min t s.t. preds_t[i] == AR[i] (None if never)
  - "stable"   = preds_t[i] == AR[i] for ALL t >= first_correct_iter
  - "flipped_back" = correct at some iter, then wrong later

A model that overwrites lucky correct tokens will show LOW stability rate
(correct at iter k, wrong at iter k+1). A stable model preserves lucky correct
tokens once predicted.

Output: per-model stats:
  - retention rate: P(correct at iter t+1 | correct at iter t, position past frontier)
  - flip-back rate: fraction of positions that became correct then went wrong

Env: same as probe_partial_prefix.py
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
N_ITERS = int(os.environ.get("N_ITERS", "8"))
OUT_JSON = os.environ["OUT_JSON"]
LABEL = os.environ.get("LABEL", Path(MODEL_PATH).name)

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

        # AR greedy reference
        ar = prompt_ids.clone()
        for _ in range(N_BLOCK + 1):
            logits = model(ar).logits
            nxt = logits[0, -1].argmax(dim=-1).view(1, 1)
            ar = torch.cat([ar, nxt], dim=-1)
        ar_continuation = ar[0, P:P + N_BLOCK + 1].tolist()  # length N_BLOCK+1
        # AR target at position i means: the token that should appear at draft
        # position i to be AR-equivalent (i.e., ar_continuation[i]).

        # Initial draft: [seed = ar_continuation[0]] + uniform random for N_BLOCK-1
        draft = [ar_continuation[0]] + [random.randrange(tokenizer.vocab_size) for _ in range(N_BLOCK - 1)]

        per_iter_correct = []  # list of length N_ITERS+1; each is bool list of length N_BLOCK
        per_iter_preds = []    # list of pred token ids

        # Record iter 0 INPUT as the starting state (before any forward)
        # Actually, the "draft state" we care about is what the model SEES at each iter.
        # iter 0 input = draft; preds = model output given draft
        for t in range(N_ITERS):
            draft_t = torch.tensor([draft], dtype=torch.long, device=model.device)
            full = torch.cat([prompt_ids, draft_t], dim=-1)
            logits = model(full).logits
            preds = logits[0, P - 1 : P - 1 + N_BLOCK].argmax(dim=-1).tolist()
            correct = [int(preds[i] == ar_continuation[i]) for i in range(N_BLOCK)]
            per_iter_correct.append(correct)
            per_iter_preds.append(preds)
            # Cascade update: position 0 stays as seed (= ar_continuation[0]); positions
            # 1..N-1 take previous iter's predictions at those positions.
            draft = [ar_continuation[0]] + preds[0:N_BLOCK - 1]
            # Note: preds[i] is the prediction made at position i. In Jacobi, the
            # new draft token at position j is preds[j-1] (the prediction at position
            # j-1 forecasts the token at j). So draft_new[j] = preds[j-1].

        # Aggregate stats
        # 1. first_correct_iter[i] = min t s.t. per_iter_correct[t][i] == 1, or None
        first_correct = [None] * N_BLOCK
        for i in range(N_BLOCK):
            for t in range(N_ITERS):
                if per_iter_correct[t][i]:
                    first_correct[i] = t
                    break

        # 2. After first_correct[i], how often does position i stay correct?
        retention = [None] * N_BLOCK  # fraction of subsequent iters where still correct
        for i in range(N_BLOCK):
            if first_correct[i] is None: continue
            t0 = first_correct[i]
            subseq = [per_iter_correct[t][i] for t in range(t0, N_ITERS)]
            retention[i] = sum(subseq) / len(subseq)

        # 3. Flip-back: position became correct, then went wrong at least once
        flipped_back = [False] * N_BLOCK
        for i in range(N_BLOCK):
            if first_correct[i] is None: continue
            t0 = first_correct[i]
            if any(per_iter_correct[t][i] == 0 for t in range(t0 + 1, N_ITERS)):
                flipped_back[i] = True

        all_results.append({
            "prompt_idx": int(pidx),
            "per_iter_correct": per_iter_correct,
            "first_correct": first_correct,
            "retention": retention,
            "flipped_back": flipped_back,
        })
        n_first_correct = sum(1 for x in first_correct if x is not None)
        n_flipped = sum(1 for x in flipped_back if x)
        print(f"  p{pidx}: ever-correct={n_first_correct}/{N_BLOCK}  flipped_back={n_flipped}/{n_first_correct}  "
              f"per-iter correct counts: {[sum(c) for c in per_iter_correct]}")

out_p = Path(OUT_JSON)
out_p.parent.mkdir(parents=True, exist_ok=True)
with out_p.open("w") as f:
    json.dump({"label": LABEL, "model_path": MODEL_PATH,
               "N_BLOCK": N_BLOCK, "N_ITERS": N_ITERS,
               "results": all_results}, f)
print(f"\n[{LABEL}] wrote -> {OUT_JSON}")
