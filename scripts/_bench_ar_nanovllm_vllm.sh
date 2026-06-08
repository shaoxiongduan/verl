#!/usr/bin/env bash
# AR-only TPS bench for vLLM vs nanovllm at BS=1, 4.
set -eu
cd /mnt/weka/home/hao.zhang/shao/verl
source .venv/bin/activate

MODEL=/mnt/weka/home/hao.zhang/.cache/huggingface/hub/models--JacobiForcing--JacobiForcing_Math_7B_v1/snapshots/e65283c1b3d205b23c2bdf9946158035c409d3a1
PROMPTS=eval_passk/eval_prompts_tpf.jsonl

echo "============== vLLM AR =============="
for BS in 1 4 16 32 64; do
  echo "--- vllm AR BS=$BS ---"
  VLLM_DIAG_MODE=ar JACOBI_K=32 \
    python3 scripts/_diag_vllm_speed.py \
    --model "$MODEL" --prompts_jsonl "$PROMPTS" \
    --max_new_tokens 256 --max_num_seqs $BS --n_prompts 64 2>&1 | grep "^\[diag\] mode\|^\[diag\] n_prompts\|^\[diag\] avg"
done

echo
echo "============== nanovllm AR =============="
python3 << 'PY' 2>&1
import sys, os, time, json
sys.path.insert(0, "/mnt/weka/home/hao.zhang/shao/Decode-Learning")
from nanovllm import LLM, SamplingParams

MODEL="/mnt/weka/home/hao.zhang/.cache/huggingface/hub/models--JacobiForcing--JacobiForcing_Math_7B_v1/snapshots/e65283c1b3d205b23c2bdf9946158035c409d3a1"
PROMPTS="/mnt/weka/home/hao.zhang/shao/verl/eval_passk/eval_prompts_tpf.jsonl"

prompts_raw = [json.loads(l) for l in open(PROMPTS)][:64]
llm = LLM(MODEL, max_model_len=4096, tensor_parallel_size=1)
tok = llm.tokenizer
chat_texts = [tok.apply_chat_template([{"role":"user","content":p["input"]}],
              tokenize=False, add_generation_prompt=True) for p in prompts_raw]

# Warm up
sp_warm = SamplingParams(temperature=0.0, max_tokens=32, decode_strategy="autoregressive")
_ = llm.generate(chat_texts[:1], sp_warm)

# Bench sequential rounds with BS prompts each
for bs in [1, 4, 16, 32, 64]:
    sp = SamplingParams(temperature=0.0, max_tokens=256, decode_strategy="autoregressive")
    n_rounds = 64 // bs
    all_tok = 0; all_dt = 0
    for r in range(n_rounds):
        sub = chat_texts[r*bs:(r+1)*bs]
        t0 = time.time()
        outs = llm.generate(sub, sp)
        dt = time.time() - t0
        for o in outs:
            tok_ids = o.get("token_ids") if isinstance(o, dict) else None
            if tok_ids is None:
                # outs are sequences with .completion_token_ids or .output_ids depending on API
                text = o.get("text","") if isinstance(o, dict) else ""
                all_tok += len(tok.encode(text))
            else:
                all_tok += len(tok_ids)
        all_dt += dt
    tps = all_tok/max(1e-9, all_dt)
    print(f"[nv-ar] BS={bs} wall={all_dt:.2f}s tokens={all_tok} TPS={tps:.1f} ms/tok={1000*all_dt/all_tok:.2f}")
llm.exit() if hasattr(llm, "exit") else None
PY

echo "DONE"
