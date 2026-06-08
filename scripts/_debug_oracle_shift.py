"""Debug version: trace 5 iters of oracle_shift sim with keep_M=0 on math_k3."""
import json, random, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/math_k3_ds_step_300"
PROMPTS = "/mnt/weka/home/hao.zhang/shao/verl/eval_passk/deepscaler_tpf_prompts_16.jsonl"
K = 32
VOCAB = 152064
KEEP_M = 0
SEED = 42

print(f"loading {MODEL}", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda:0")
model.eval()
rng = random.Random(SEED)

prompts = [json.loads(l) for l in open(PROMPTS)]
p = prompts[0]
chat = [{"role": "user", "content": p["input"]}]
text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
prompt_ids = tok(text, return_tensors="pt").input_ids[0].tolist()
print(f"\nPrompt (first 100 chars): {text[:100]}")

committed = list(prompt_ids)
draft = [rng.randrange(VOCAB) for _ in range(K)]

print(f"\n=== Initial draft (K=32 uniform random) ===")
print(f"draft tokens: {draft[:8]}... (first 8 of 32)")
print(f"draft decoded: {tok.decode(draft[:8])!r}")

for it in range(5):
    L = len(committed)
    print(f"\n\n=== ITER {it+1} ===")
    print(f"committed length: {L}")
    print(f"last 4 committed tokens: {committed[-4:]} = {[tok.decode([t]) for t in committed[-4:]]!r}")
    print(f"draft K tokens (first 6): {draft[:6]} = {[tok.decode([t]) for t in draft[:6]]!r}")

    with torch.no_grad():
        inp = torch.tensor([committed + draft], dtype=torch.long, device=model.device)
        out = model(input_ids=inp)
        logits_K = out.logits[0, L - 1 : L - 1 + K, :]
        target_argmax = logits_K.argmax(dim=-1).cpu().tolist()
        # Show top-3 at first 4 positions
        top3 = logits_K[:4].float().topk(3, dim=-1)

    print(f"target_argmax (first 8): {target_argmax[:8]} = {[tok.decode([t]) for t in target_argmax[:8]]!r}")
    print(f"top-3 at first 4 positions (each: rank tokens):")
    for j in range(4):
        ts = top3.indices[j].cpu().tolist()
        ps = top3.values[j].cpu().tolist()
        print(f"  pos {j}: top1={ts[0]}({tok.decode([ts[0]])!r},lp={ps[0]:.2f})  top2={ts[1]}({tok.decode([ts[1]])!r},lp={ps[1]:.2f})  top3={ts[2]}({tok.decode([ts[2]])!r},lp={ps[2]:.2f})")
    # Match check
    match = [int(draft[j] == target_argmax[j]) for j in range(8)]
    print(f"draft==target match (first 8): {match}")

    n_acc = 0
    for j in range(K):
        if draft[j] == target_argmax[j]:
            n_acc += 1
        else:
            break
    print(f"n_acc = {n_acc}")
    for j in range(n_acc):
        committed.append(draft[j])
    if n_acc < K:
        bonus = target_argmax[n_acc]
        committed.append(bonus)
        print(f"committed bonus = target_argmax[{n_acc}] = {bonus} ({tok.decode([bonus])!r})")

    # Build next draft: keep_M=0 → all noise
    new_draft = [rng.randrange(VOCAB) for _ in range(K)]
    draft = new_draft
    print(f"next iter draft (first 6): {draft[:6]} = {[tok.decode([t]) for t in draft[:6]]!r}")
