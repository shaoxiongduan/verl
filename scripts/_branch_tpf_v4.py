"""Branching TPF v4 — content-rich recording for mechanism analysis.

Same branching protocol as v3 (full-window samples at T=1.0, greedy-acceptance
horizon rollouts; content of commits is branch-invariant = greedy path), but
records everything needed to explain WHY winners win:

per branch point:
  ctx_tail      : last 48 committed tokens (qualitative context)
  future        : >= K tokens of the greedy continuation from this state
                  (reference rollout; branch-invariant "what will be decoded")
  vanilla       : {toks, cum}                      (argmax window)
  cands         : [{toks, cum, logp, rank}, ...]   (sampled windows)
                  logp/rank: per-token log-prob and rank under the sampling
                  distribution at each window position.
"""
from __future__ import annotations
import argparse, json, random
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

STOP_IDS = {151645, 151643}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompts_jsonl", required=True)
    p.add_argument("--out_jsonl", required=True)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--max_new", type=int, default=512)
    p.add_argument("--max_iters", type=int, default=512)
    p.add_argument("--n_alt", type=int, default=15)
    p.add_argument("--h", type=int, default=6)
    p.add_argument("--branch_every", type=int, default=10)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--n_prompts", type=int, default=8)
    p.add_argument("--vocab_size", type=int, default=152064)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def fwd(model, committed, draft):
    L = len(committed)
    K = len(draft)
    inp = torch.tensor([committed + draft], dtype=torch.long, device=model.device)
    logits = model(input_ids=inp).logits[0, L - 1 : L - 1 + K, :]
    return logits.argmax(dim=-1).cpu().tolist(), logits


def acc_len(cur, draft):
    n = 0
    for a, b in zip(cur, draft):
        if a != b:
            break
        n += 1
    return n


@torch.no_grad()
def roll(model, committed, draft, h, K, fill_seed, vocab, need_tokens=None):
    """Vanilla greedy Jacobi from (committed, draft). If need_tokens is set,
    run until that many tokens commit (or 4*h iters); else run exactly h iters.
    Returns (cum_per_forward, committed_tokens)."""
    rng = random.Random(fill_seed)
    committed = list(committed)
    draft = list(draft)
    cum, out = [], []
    tot, it = 0, 0
    lim = h if need_tokens is None else 4 * h
    stopped = False
    while it < lim:
        if need_tokens is not None and tot >= need_tokens:
            break
        if stopped:
            cum.append(tot); it += 1; continue
        cur, _ = fwd(model, committed, draft)
        it += 1
        n_acc = acc_len(cur, draft)
        if n_acc > 0:
            toks = cur[:n_acc]
            for si, t in enumerate(toks):
                if t in STOP_IDS:
                    toks = toks[: si + 1]; stopped = True; break
            committed += toks; out += toks; tot += len(toks)
        cum.append(tot)
        shifted = cur[n_acc:K]
        draft = shifted + [rng.randrange(vocab) for _ in range(n_acc)]
    return cum, out


@torch.no_grad()
def run_prompt(model, prompt_ids, args, rng):
    K = args.K
    committed = list(prompt_ids)
    draft = [rng.randrange(args.vocab_size) for _ in range(K)]
    total, n_fwd, n_commits = 0, 0, 0
    records = []
    while total < args.max_new and n_fwd < args.max_iters:
        cur, logits = fwd(model, committed, draft)
        n_fwd += 1
        n_acc = acc_len(cur, draft)
        if n_acc > 0:
            toks = cur[:n_acc]
            hit = False
            for si, t in enumerate(toks):
                if t in STOP_IDS:
                    toks = toks[: si + 1]; hit = True; break
            committed += toks
            total += len(toks)
            n_commits += 1
            if hit or total >= args.max_new:
                break
        shifted = cur[n_acc:K]
        fill = [rng.randrange(args.vocab_size) for _ in range(n_acc)]
        vanilla_next = shifted + fill

        if n_acc > 0 and n_commits % args.branch_every == 0:
            probs = F.softmax(logits[n_acc:K].float() / args.temp, dim=-1)  # (K-n_acc, V)
            sorted_idx = probs.argsort(dim=-1, descending=True)
            fill_seed = rng.randrange(1 << 30)

            # reference greedy future for the whole window span
            _, future = roll(model, committed, vanilla_next, args.h, K,
                             fill_seed, args.vocab_size, need_tokens=K)
            van_cum, _ = roll(model, committed, vanilla_next, args.h, K,
                              fill_seed, args.vocab_size)

            cands = []
            for _ in range(args.n_alt):
                samp = torch.multinomial(probs, 1).squeeze(-1)        # (K-n_acc,)
                lp = probs.gather(-1, samp.unsqueeze(-1)).squeeze(-1).log()
                rk = (sorted_idx == samp.unsqueeze(-1)).float().argmax(dim=-1)
                cnd = samp.cpu().tolist() + fill
                cum, _ = roll(model, committed, cnd, args.h, K, fill_seed,
                              args.vocab_size)
                cands.append({"toks": cnd, "cum": cum,
                              "logp": [round(x, 3) for x in lp.cpu().tolist()],
                              "rank": rk.cpu().int().tolist()})

            records.append({
                "pos": total,
                "n_shift": K - n_acc,
                "ctx_tail": committed[-48:],
                "future": future[:K],
                "vanilla": {"toks": vanilla_next, "cum": van_cum},
                "cands": cands,
            })
        draft = vanilla_next
    return {"n_tokens": total, "n_forwards": n_fwd, "branches": records}


def main():
    args = parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 device_map="cuda:0")
    model.eval()
    prompts = [json.loads(l) for l in open(args.prompts_jsonl)][: args.n_prompts]
    rng = random.Random(args.seed)
    with open(args.out_jsonl, "w") as f:
        for i, p in enumerate(prompts):
            chat = [{"role": "user", "content": p["input"]}]
            text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            pid = tok(text, return_tensors="pt").input_ids[0].tolist()
            r = run_prompt(model, pid, args, rng)
            r["batch_idx"] = i
            f.write(json.dumps(r) + "\n"); f.flush()
            print(f"[branch4] [{i+1}/{len(prompts)}] tok={r['n_tokens']} "
                  f"branch_points={len(r['branches'])}", flush=True)


if __name__ == "__main__":
    main()
