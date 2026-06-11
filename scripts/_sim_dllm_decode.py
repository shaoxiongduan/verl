"""dLLM-style decoding schemes on (causal) JacobiForcing models, with quality grading.

Modes:
  vanilla : streaming Jacobi, prefix-match commit (AR-greedy-equivalent).
            Reference for BOTH quality and TPF.
  dllm    : DiffusionGemma-style block-complete canvas decode, NO verification:
            per step, causal forward over [committed | canvas]; candidates =
            argmax (cand_temp=0) or Gumbel-sampled (logits/T + gumbel);
            CONFIDENCE KEEP: keep positions with entropy <= keep_tau, replace
            the rest with fresh random tokens; block commits when the argmax
            is stable for `stable_needed` consecutive steps OR step cap hits
            (committing the current candidates as-is — quality may drift!).
  hybrid  : streaming Jacobi with n_refine dLLM-style refinement passes
            between verification passes. Refinement = confidence-keep +
            rerandomize on sampled candidates (draft building only, commits
            nothing); verification = standard prefix-match commit. Preserves
            AR-greedy equivalence exactly; TPF counts ALL forwards.

Quality: extract \\boxed{...} (else last number) and compare to expected_answer.
"""
from __future__ import annotations
import argparse, json, random, re
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

STOP_IDS = {151645, 151643}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompts_jsonl", required=True)
    p.add_argument("--out_jsonl", required=True)
    p.add_argument("--mode", choices=["vanilla", "dllm", "hybrid"], required=True)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--max_new", type=int, default=512)
    p.add_argument("--max_fwd", type=int, default=512)
    p.add_argument("--cand_temp", type=float, default=0.0, help="0 = argmax candidates")
    p.add_argument("--keep_tau", type=float, default=1.0, help="entropy threshold for keeping")
    p.add_argument("--max_steps_per_block", type=int, default=16, help="dllm step cap per block")
    p.add_argument("--stable_needed", type=int, default=2)
    p.add_argument("--n_refine", type=int, default=1, help="hybrid refinement passes per verify")
    p.add_argument("--n_prompts", type=int, default=16)
    p.add_argument("--prompt_field", default="problem")
    p.add_argument("--answer_field", default="expected_answer")
    p.add_argument("--vocab_size", type=int, default=152064)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def fwd(model, committed, draft):
    L = len(committed)
    K = len(draft)
    inp = torch.tensor([committed + draft], dtype=torch.long, device=model.device)
    return model(input_ids=inp).logits[0, L - 1 : L - 1 + K, :]


def candidates(logits, temp):
    if temp <= 0:
        return logits.argmax(dim=-1).cpu().tolist()
    g = -torch.log(-torch.log(torch.rand_like(logits.float())))
    return (logits.float() / temp + g).argmax(dim=-1).cpu().tolist()


def entropies(logits):
    lp = F.log_softmax(logits.float(), dim=-1)
    ent = -(lp.exp() * lp).sum(dim=-1)
    return ent.cpu().tolist()


def acc_len(cur, draft):
    n = 0
    for a, b in zip(cur, draft):
        if a != b:
            break
        n += 1
    return n


@torch.no_grad()
def decode_vanilla(model, prompt_ids, args, rng):
    K = args.K
    committed = list(prompt_ids)
    np_ = len(committed)
    draft = [rng.randrange(args.vocab_size) for _ in range(K)]
    total, n_fwd = 0, 0
    while total < args.max_new and n_fwd < args.max_fwd:
        logits = fwd(model, committed, draft)
        cur = logits.argmax(dim=-1).cpu().tolist()
        n_fwd += 1
        n_acc = acc_len(cur, draft)
        if n_acc > 0:
            toks = cur[:n_acc]
            hit = False
            for si, t in enumerate(toks):
                if t in STOP_IDS:
                    toks = toks[: si + 1]; hit = True; break
            committed += toks; total += len(toks)
            if hit or total >= args.max_new:
                break
        shifted = cur[n_acc:K]
        draft = shifted + [rng.randrange(args.vocab_size) for _ in range(n_acc)]
    return committed[np_:], n_fwd


@torch.no_grad()
def decode_dllm(model, prompt_ids, args, rng):
    K = args.K
    committed = list(prompt_ids)
    np_ = len(committed)
    total, n_fwd = 0, 0
    done = False
    while total < args.max_new and n_fwd < args.max_fwd and not done:
        canvas = [rng.randrange(args.vocab_size) for _ in range(K)]
        prev_argmax = None
        stable = 0
        final = None
        for _ in range(args.max_steps_per_block):
            if n_fwd >= args.max_fwd:
                break
            logits = fwd(model, committed, canvas)
            n_fwd += 1
            am = logits.argmax(dim=-1).cpu().tolist()
            if prev_argmax is not None and am == prev_argmax:
                stable += 1
            else:
                stable = 0
            prev_argmax = am
            if stable >= args.stable_needed - 1:
                final = am
                break
            cand = candidates(logits, args.cand_temp)
            ent = entropies(logits)
            canvas = [cand[j] if ent[j] <= args.keep_tau
                      else rng.randrange(args.vocab_size) for j in range(K)]
            final = am
        # commit the whole block (no verification)
        toks = final or []
        for si, t in enumerate(toks):
            if t in STOP_IDS:
                toks = toks[: si + 1]; done = True; break
        committed += toks
        total += len(toks)
    return committed[np_:], n_fwd


@torch.no_grad()
def decode_hybrid(model, prompt_ids, args, rng):
    K = args.K
    committed = list(prompt_ids)
    np_ = len(committed)
    draft = [rng.randrange(args.vocab_size) for _ in range(K)]
    total, n_fwd = 0, 0
    while total < args.max_new and n_fwd < args.max_fwd:
        # n_refine dLLM-style draft-refinement passes (no commit)
        for _ in range(args.n_refine):
            if n_fwd >= args.max_fwd:
                break
            logits = fwd(model, committed, draft)
            n_fwd += 1
            cand = candidates(logits, args.cand_temp)
            ent = entropies(logits)
            draft = [cand[j] if ent[j] <= args.keep_tau
                     else rng.randrange(args.vocab_size) for j in range(K)]
        if n_fwd >= args.max_fwd:
            break
        # verification pass: standard prefix-match commit
        logits = fwd(model, committed, draft)
        cur = logits.argmax(dim=-1).cpu().tolist()
        n_fwd += 1
        n_acc = acc_len(cur, draft)
        if n_acc > 0:
            toks = cur[:n_acc]
            hit = False
            for si, t in enumerate(toks):
                if t in STOP_IDS:
                    toks = toks[: si + 1]; hit = True; break
            committed += toks; total += len(toks)
            if hit or total >= args.max_new:
                break
        shifted = cur[n_acc:K]
        draft = shifted + [rng.randrange(args.vocab_size) for _ in range(n_acc)]
    return committed[np_:], n_fwd


BOX = re.compile(r"\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}")
NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_answer(text):
    m = BOX.findall(text)
    if m:
        return m[-1].strip().replace(",", "").replace(" ", "")
    nums = NUM.findall(text)
    return nums[-1].replace(",", "") if nums else ""


def norm(s):
    s = str(s).strip().replace(",", "").replace(" ", "").replace("$", "")
    try:
        return f"{float(s):g}"
    except ValueError:
        return s


def main():
    args = parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 device_map="cuda:0")
    model.eval()
    prompts = [json.loads(l) for l in open(args.prompts_jsonl)][: args.n_prompts]
    rng = random.Random(args.seed)
    dec = {"vanilla": decode_vanilla, "dllm": decode_dllm, "hybrid": decode_hybrid}[args.mode]
    tot_t = tot_f = n_right = 0
    with open(args.out_jsonl, "w") as f:
        for i, p in enumerate(prompts):
            chat = [{"role": "user", "content": p[args.prompt_field]}]
            text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            pid = tok(text, return_tensors="pt").input_ids[0].tolist()
            resp_ids, n_fwd = dec(model, pid, args, rng)
            resp = tok.decode(resp_ids)
            pred = extract_answer(resp)
            right = norm(pred) == norm(p[args.answer_field])
            n_right += right
            tot_t += len(resp_ids); tot_f += n_fwd
            f.write(json.dumps({"batch_idx": i, "n_tokens": len(resp_ids),
                                "n_forwards": n_fwd, "tpf": len(resp_ids)/max(1,n_fwd),
                                "pred": pred, "gold": p[args.answer_field],
                                "correct": bool(right), "completion": resp}) + "\n")
            print(f"[dllm] [{i+1}/{len(prompts)}] {args.mode} tok={len(resp_ids)} "
                  f"fwd={n_fwd} TPF={len(resp_ids)/max(1,n_fwd):.2f} "
                  f"{'OK' if right else 'WRONG'} pred={pred[:18]!r}", flush=True)
    print(f"[dllm] {args.mode} cand_temp={args.cand_temp} keep_tau={args.keep_tau} "
          f"n_refine={args.n_refine} cap={args.max_steps_per_block}: "
          f"CORPUS TPF={tot_t}/{tot_f}={tot_t/max(1,tot_f):.3f}  "
          f"ACC={n_right}/{len(prompts)}={n_right/len(prompts):.2f}")


if __name__ == "__main__":
    main()
