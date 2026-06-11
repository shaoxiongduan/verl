"""Branching TPF experiment v3 — horizon sweep, group-size sweep, luck check.

Extends v2:
  - Each candidate window is rolled ONCE for h_max forwards, recording the
    CUMULATIVE committed-token count after every forward → outcomes at every
    horizon h=1..h_max from a single rollout.
  - n_alt sampled alternatives (default 15) → group-size curves: best-of-
    {1+3, 1+7, 1+15} computed by subsetting the same data.
  - Luck check: the h_ref-winning alternative (and vanilla) are re-rolled with
    `repl` different random-fill seeds. Greedy acceptance is deterministic
    given (state, draft, fill-seq), so if the winner's advantage persists
    across fill seeds it is a property of the DRAFT TOKENS, not of the random
    tail refills.
  - Win mechanism: at the winner's first token-level divergence j* from the
    vanilla draft, check whether its token equals the token eventually
    committed at that offset (= the greedy continuation, which is branch-
    invariant). "right_guess" = it guessed the greedy token at a position
    where argmax (under the previous iter's conditioning) had it wrong;
    "context" = the divergent tokens never got committed within the horizon
    (they only changed conditioning); "unreached" = commits didn't reach j*.
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
    p.add_argument("--h_max", type=int, default=12)
    p.add_argument("--h_ref", type=int, default=6, help="horizon used to pick the winner")
    p.add_argument("--repl", type=int, default=2, help="extra fill-seed replications of winner+vanilla")
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
def roll_cum(model, committed, draft, h_max, K, fill_seed, vocab):
    """Roll h_max vanilla iterations; return (cum committed per forward,
    committed tokens list)."""
    rng = random.Random(fill_seed)
    committed = list(committed)
    draft = list(draft)
    cum, toks_out = [], []
    tot = 0
    stopped = False
    for _ in range(h_max):
        if stopped:
            cum.append(tot)
            continue
        cur, _ = fwd(model, committed, draft)
        n_acc = acc_len(cur, draft)
        if n_acc > 0:
            toks = cur[:n_acc]
            for si, t in enumerate(toks):
                if t in STOP_IDS:
                    toks = toks[: si + 1]; stopped = True; break
            committed += toks
            toks_out += toks
            tot += len(toks)
        cum.append(tot)
        shifted = cur[n_acc:K]
        draft = shifted + [rng.randrange(vocab) for _ in range(n_acc)]
    return cum, toks_out


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
            probs = F.softmax(logits[n_acc:K].float() / args.temp, dim=-1)
            cands = []
            for _ in range(args.n_alt):
                alt = torch.multinomial(probs, 1).squeeze(-1).cpu().tolist()
                cands.append(alt + fill)

            fill_seed = rng.randrange(1 << 30)
            van_cum, van_toks = roll_cum(model, committed, vanilla_next,
                                         args.h_max, K, fill_seed, args.vocab_size)
            alt_cums, alt_toks = [], []
            for cnd in cands:
                c, t = roll_cum(model, committed, cnd, args.h_max, K,
                                fill_seed, args.vocab_size)
                alt_cums.append(c); alt_toks.append(t)

            # winner by h_ref
            hr = args.h_ref - 1
            wi = max(range(len(cands)), key=lambda i: alt_cums[i][hr])
            rec = {"pos": total, "van_cum": van_cum, "alt_cums": alt_cums,
                   "winner": wi}

            # luck check: replicate winner & vanilla with different fill seeds
            repl = []
            for _ in range(args.repl):
                fs = rng.randrange(1 << 30)
                vc, _ = roll_cum(model, committed, vanilla_next, args.h_max, K,
                                 fs, args.vocab_size)
                wc, _ = roll_cum(model, committed, cands[wi], args.h_max, K,
                                 fs, args.vocab_size)
                repl.append({"van": vc, "win": wc})
            rec["repl"] = repl

            # win mechanism: first divergence of winner draft vs vanilla draft
            wd = cands[wi]
            jstar = next((j for j in range(K) if wd[j] != vanilla_next[j]), None)
            mech = "identical"
            if jstar is not None:
                greedy_toks = van_toks  # branch-invariant greedy continuation
                if len(greedy_toks) > jstar:
                    mech = "right_guess" if wd[jstar] == greedy_toks[jstar] else "wrong_guess"
                else:
                    mech = "unreached_context"
            rec["jstar"] = jstar
            rec["mech"] = mech
            records.append(rec)

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
            print(f"[branch3] [{i+1}/{len(prompts)}] tok={r['n_tokens']} "
                  f"branch_points={len(r['branches'])}", flush=True)

    import statistics as st
    B = [b for r in rows for b in r["branches"]]
    if not B:
        print("[branch3] no branch points"); return
    H = args.h_max
    print(f"\n[branch3] branch points: {len(B)}   (n_alt={args.n_alt}, h_max={H})")
    print(f"[branch3] {'h':>3s} {'vanilla':>8s} {'best4':>7s} {'best8':>7s} {'best16':>7s} "
          f"{'hd4':>6s} {'hd8':>6s} {'hd16':>6s} {'tau_h_vs_h%d' % args.h_ref:>12s}")
    import itertools
    for h in range(1, H + 1):
        van = [b["van_cum"][h-1] / h for b in B]
        outs = {}
        for g, na in (("4", 3), ("8", 7), ("16", 15)):
            outs[g] = [max([b["van_cum"][h-1]] + [c[h-1] for c in b["alt_cums"][:na]]) / h for b in B]
        # tau between rank at horizon h and rank at h_ref, across candidates within a state
        conc = disc = 0
        hr = args.h_ref - 1
        for b in B:
            xs = [b["van_cum"][h-1]] + [c[h-1] for c in b["alt_cums"]]
            ys = [b["van_cum"][hr]] + [c[hr] for c in b["alt_cums"]]
            for (x1, y1), (x2, y2) in itertools.combinations(zip(xs, ys), 2):
                d = (x1 - x2) * (y1 - y2)
                conc += d > 0; disc += d < 0
        tau = (conc - disc) / max(1, conc + disc)
        print(f"[branch3] {h:3d} {st.mean(van):8.3f} {st.mean(outs['4']):7.3f} "
              f"{st.mean(outs['8']):7.3f} {st.mean(outs['16']):7.3f} "
              f"{st.mean(outs['4'])-st.mean(van):6.3f} {st.mean(outs['8'])-st.mean(van):6.3f} "
              f"{st.mean(outs['16'])-st.mean(van):6.3f} {tau:12.3f}")

    # luck check at h_ref
    hr = args.h_ref - 1
    orig_adv, repl_adv = [], []
    for b in B:
        wa = b["alt_cums"][b["winner"]][hr] - b["van_cum"][hr]
        if wa <= 0:  # winner didn't beat vanilla originally
            continue
        orig_adv.append(wa / args.h_ref)
        ra = st.mean((r["win"][hr] - r["van"][hr]) / args.h_ref for r in b["repl"])
        repl_adv.append(ra)
    if orig_adv:
        persist = sum(1 for a in repl_adv if a > 0) / len(repl_adv)
        print(f"\n[branch3] LUCK CHECK (winners that beat vanilla at h={args.h_ref}: {len(orig_adv)}/{len(B)})")
        print(f"[branch3] original advantage mean={st.mean(orig_adv):.3f}  "
              f"replicated (new fill seeds) mean={st.mean(repl_adv):.3f}  "
              f"advantage persists (>0) in {100*persist:.0f}% of cases")
    mechs = {}
    for b in B:
        wa = b["alt_cums"][b["winner"]][hr] - b["van_cum"][hr]
        if wa > 0:
            mechs[b["mech"]] = mechs.get(b["mech"], 0) + 1
    print(f"[branch3] win mechanisms: {mechs}")


if __name__ == "__main__":
    main()
