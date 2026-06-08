#!/usr/bin/env bash
# Try multiple vLLM AR BS=1 configs + nanovllm AR BS=1, find what's eating perf.
set -eu
cd /mnt/weka/home/hao.zhang/shao/verl
source .venv/bin/activate

export MODEL=/mnt/weka/home/hao.zhang/.cache/huggingface/hub/models--JacobiForcing--JacobiForcing_Math_7B_v1/snapshots/e65283c1b3d205b23c2bdf9946158035c409d3a1
export PROMPTS=/mnt/weka/home/hao.zhang/shao/verl/eval_passk/eval_prompts_tpf.jsonl

cases() {
  # AR_CASE  GMU  EAGER  ATTN_BACKEND  CHUNKED
  cat <<EOF
default 0.6 0 - 1
eager 0.6 1 - 1
gmu09 0.9 0 - 1
eager_gmu09 0.9 1 - 1
nochunk 0.6 0 - 0
eager_nochunk 0.6 1 - 0
fa3 0.6 0 FLASH_ATTN_VLLM_V1 1
eager_fa3 0.6 1 FLASH_ATTN_VLLM_V1 1
EOF
}

cases | while read CASE GMU EAGER ATTN CHUNKED; do
  echo "============================="
  echo "vLLM CASE=$CASE GMU=$GMU EAGER=$EAGER ATTN=$ATTN CHUNKED=$CHUNKED"
  echo "============================="
  ATTN_ARG=""
  [ "$ATTN" != "-" ] && ATTN_ARG="$ATTN" || ATTN_ARG=""
  AR_CASE=$CASE GMU=$GMU EAGER=$EAGER ATTN_BACKEND="$ATTN_ARG" CHUNKED=$CHUNKED \
    python3 scripts/_diag_vllm_ar_configs.py 2>&1 | grep -E "^\[case=|^========|Capturing CUDA graphs"
  echo
done

echo "============================="
echo "nanovllm AR BS=1"
echo "============================="
python3 << 'PY' 2>&1
import sys, os, time, json
sys.path.insert(0, "/mnt/weka/home/hao.zhang/shao/Decode-Learning")
from nanovllm import LLM, SamplingParams

MODEL = os.environ["MODEL"]
PROMPTS = os.environ["PROMPTS"]
prompts_raw = [json.loads(l) for l in open(PROMPTS)][:8]
llm = LLM(MODEL, max_model_len=4096, tensor_parallel_size=1)
tok = llm.tokenizer
chat_texts = [
    tok.apply_chat_template([{"role":"user","content":p["input"]}],
                             tokenize=False, add_generation_prompt=True)
    for p in prompts_raw
]
sp_warm = SamplingParams(temperature=0.0, max_tokens=32, decode_strategy="autoregressive")
_ = llm.generate(chat_texts[:1], sp_warm)
sp = SamplingParams(temperature=0.0, max_tokens=256, decode_strategy="autoregressive")
all_tok = 0; all_dt = 0
for p in chat_texts:
    t0 = time.time()
    outs = llm.generate([p], sp)
    dt = time.time() - t0
    for o in outs:
        text = o.get("text","") if isinstance(o, dict) else getattr(o, "text", "")
        all_tok += len(tok.encode(text))
    all_dt += dt
print(f"[nanovllm-ar] BS=1 wall={all_dt:.2f}s tokens={all_tok} TPS={all_tok/all_dt:.1f} ms_tok={1000*all_dt/all_tok:.2f}")
PY

echo "DONE"
