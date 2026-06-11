"""Branching TPF experiment v2 — fixed redo of _sim_jacobi_lookahead.py.

Fixes over v1:
  1. TRUE self-acceptance (production streaming Jacobi: accept = prefix where
     this iter's argmax == this iter's input draft). No external "clean
     reference", no bonus token — same dynamics as the stop-fixed jsim.
  2. HORIZON-h outcome instead of myopic next-iter n_acc: each candidate next
     draft is rolled forward h vanilla iterations; outcome = tokens committed
     over those h forwards (subsequent TPF). Greedy rollout is deterministic
     given the draft, so the branch outcome has no sampling noise — observed
     spread IS the policy's headroom at that state.
  3. Branches are measurement-only: the main trajectory always continues with
     the vanilla argmax draft, so branch points along a trajectory are
     comparable and the main decode is unperturbed.

Per branch point we record:
  - tpf_vanilla_h   : h-horizon TPF of the vanilla (argmax-shifted) draft
  - tpf_alts_h      : h-horizon TPF of each sampled-alternative draft
  - nacc1_*         : immediate (1-iter) n_acc of each candidate, to test
                      whether the v1 myopic proxy correlates with horizon TPF

Alternative drafts: positions [0, K-n_fill) of the candidate window are
resampled token-wise from softmax(logits/T) at the previous forward (the
distribution the policy actually had), positions beyond keep the same random
fill as vanilla (controlled).
"""
from __future__ import annotations
import argparse, json, random
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompts_jsonl", required=True)
    p.add_argument("--out_jsonl", required=True)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--max_new", type=int, default=512)
    p.add_argument("--max_iters", type=int, default=512)
    p.add_argument("--n_alt", type=int, default=3, help="sampled alternatives per branch point")
    p.add_argument("--horizon", type=int, default=6, help="iters to roll each branch")
    p.add_argument("--branch_every", type=int, default=8, help="branch at every Nth commit")
    p.add_argument("--temp", type=float, default=1.0, help="sampling temp for alternative drafts")
    p.add_argument("--n_prompts", type=int, default=8)
    p.add_argument("--vocab_size", type=int, default=152064)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


STOP_IDS = {151645, 151643}


@torch.no_grad()
def fwd_argmax_logits(model, committed, draft):
    L = len(committed)
    K = len(draft)
    inp = torch.tensor([committed + draft], dtype=torch.long, device=model.device)
    logits = model(input_ids=inp).logits[0, L - 1 : L - 1 + K, :]
    return logits.argmax(dim=-1).cpu().tolist(), logits


def accept_len(cur, draft):
    n = 0
    for a, b in zip(cur, draft):
        if a != b:
            break
        n += 1
    return n


@torch.no_grad()
def rollout_h(model, committed, draft, h, K, rng, vocab):
    """Vanilla streaming Jacobi for h forwards from (committed, draft).
    Returns tokens committed (stops counted; truncates at first stop)."""
    committed = list(committed)
    draft = list(draft)
    tot = 0
    for _ in range(h):
        cur, _ = fwd_argmax_logits(model, committed, draft)
        n_acc = accept_len(cur, draft)
        if n_acc > 0:
            toks = cur[:n_acc]
            for si, t in enumerate(toks):
                if t in STOP_IDS:
                    toks = toks[: si + 1]
                    break
            committed += toks
            tot += len(toks)
            if toks[-1] in STOP_IDS:
                break
        shifted = cur[n_acc:K]
        draft = shifted + [rng.randrange(vocab) for _ in range(n_acc)]
    return tot


@torch.no_grad()
def run_prompt(model, prompt_ids, args, rng):
    K = args.K
    committed = list(prompt_ids)
    draft = [rng.randrange(args.vocab_size) for _ in range(K)]
    total, n_fwd, n_commits = 0, 0, 0
    branch_records = []
    while total < args.max_new and n_fwd < args.max_iters:
        cur, logits = fwd_argmax_logits(model, committed, draft)
        n_fwd += 1
        n_acc = accept_len(cur, draft)
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

        # vanilla next draft
        shifted = cur[n_acc:K]
        fill = [rng.randrange(args.vocab_size) for _ in range(n_acc)]
        vanilla_next = shifted + fill

        if n_acc > 0 and n_commits % args.branch_every == 0:
            # BRANCH POINT: candidates differ in the shifted (model-derived)
            # positions, resampled from the policy's own distribution.
            probs = F.softmax(logits[n_acc:K].float() / args.temp, dim=-1)
            cands = []
            for _ in range(args.n_alt):
                alt = torch.multinomial(probs, 1).squeeze(-1).cpu().tolist()
                cands.append(alt + fill)
            rec = {"pos": total, "n_acc_now": n_acc}
            seed0 = rng.randrange(1 << 30)
            van_rng = random.Random(seed0)
            rec["tpf_vanilla_h"] = rollout_h(model, committed, vanilla_next,
                                             args.horizon, K, van_rng, args.vocab_size) / args.horizon
            rec["nacc1_vanilla"] = None  # computed below with same forward
            alts_tpf, alts_n1 = [], []
            for cnd in cands:
                a_rng = random.Random(seed0)  # identical fill randomness across branches
                alts_tpf.append(rollout_h(model, committed, cnd, args.horizon, K,
                                          a_rng, args.vocab_size) / args.horizon)
                c1, _ = fwd_argmax_logits(model, committed, cnd)
                alts_n1.append(accept_len(c1, cnd))
            c1v, _ = fwd_argmax_logits(model, committed, vanilla_next)
            rec["nacc1_vanilla"] = accept_len(c1v, vanilla_next)
            rec["tpf_alts_h"] = alts_tpf
            rec["nacc1_alts"] = alts_n1
            branch_records.append(rec)

        draft = vanilla_next
    return {"n_tokens": total, "n_forwards": n_fwd,
            "tpf": total / max(1, n_fwd), "branches": branch_records}


def main():
    args = parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 device_map="cuda:0")
    model.eval()
    prompts = [json.loads(l) for l in open(args.prompts_jsonl)][: args.n_prompts]
    rng = random.Random(args.seed)
    rows = []
    with open(args.out_jsonl, "w") as f:
        for i, p in enumerate(prompts):
            chat = [{"role": "user", "content": p["input"]}]
            text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            pid = tok(text, return_tensors="pt").input_ids[0].tolist()
            r = run_prompt(model, pid, args, rng)
            r["batch_idx"] = i
            rows.append(r)
            f.write(json.dumps(r) + "\n"); f.flush()
            print(f"[branch2] [{i+1}/{len(prompts)}] tok={r['n_tokens']} TPF={r['tpf']:.2f} "
                  f"branch_points={len(r['branches'])}", flush=True)

    # aggregate
    import statistics as st
    B = [b for r in rows for b in r["branches"]]
    if not B:
        print("[branch2] no branch points"); return
    van = [b["tpf_vanilla_h"] for b in B]
    best = [max([b["tpf_vanilla_h"]] + b["tpf_alts_h"]) for b in B]
    alt_mean = [st.mean(b["tpf_alts_h"]) for b in B]
    win = sum(1 for b in B if max(b["tpf_alts_h"]) > b["tpf_vanilla_h"])
    # myopia check: rank-correlation between 1-iter n_acc and h-horizon tpf among candidates
    import itertools
    conc = disc = 0
    for b in B:
        xs = [b["nacc1_vanilla"]] + b["nacc1_alts"]
        ys = [b["tpf_vanilla_h"]] + b["tpf_alts_h"]
        for (x1, y1), (x2, y2) in itertools.combinations(zip(xs, ys), 2):
            if (x1 - x2) * (y1 - y2) > 0: conc += 1
            elif (x1 - x2) * (y1 - y2) < 0: disc += 1
    tau = (conc - disc) / max(1, conc + disc)
    print(f"\n[branch2] branch points: {len(B)}")
    print(f"[branch2] h-TPF vanilla mean={st.mean(van):.3f}  alts mean={st.mean(alt_mean):.3f}  "
          f"best-of-all mean={st.mean(best):.3f}  oracle headroom=+{st.mean(best)-st.mean(van):.3f}")
    print(f"[branch2] an alternative beats vanilla at {100*win/len(B):.0f}% of branch points")
    print(f"[branch2] myopia check: Kendall-tau(1-iter n_acc, h-TPF) = {tau:+.3f}")


if __name__ == "__main__":
    main()
