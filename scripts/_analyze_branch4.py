"""Analyze branch4 records: what content properties drive horizon TPF?

Per candidate (and vanilla), against the branch-invariant greedy `future`:
  prefix    : longest prefix matching future
  match     : # positions matching future (within the n_shift model-derived
              region only; the fill region is shared across candidates)
  run       : longest contiguous match run anywhere in the window
  mrank     : mean sampling rank of the candidate's tokens (vanilla = 0)
  outcome   : cum[h-1] / h

Outputs:
  - within-state Kendall-tau of outcome vs each feature (what predicts wins)
  - winner-vs-vanilla / loser-vs-vanilla feature deltas
  - conditioning-wins: winners that match the future LESS than vanilla
  - token classes at positions where winner matches future and vanilla doesn't
  - qualitative dumps of the top-N branch points by winner advantage
"""
from __future__ import annotations
import argparse, itertools, json, re, statistics as st
from collections import Counter
from transformers import AutoTokenizer


def feats(toks, future, n_shift):
    m = [int(j < len(future) and toks[j] == future[j]) for j in range(n_shift)]
    prefix = 0
    for x in m:
        if not x: break
        prefix += 1
    run = best = 0
    for x in m:
        run = run + 1 if x else 0
        best = max(best, run)
    return {"prefix": prefix, "match": sum(m), "run": best, "mvec": m}


def tau(xs, ys):
    conc = disc = 0
    for (x1, y1), (x2, y2) in itertools.combinations(zip(xs, ys), 2):
        d = (x1 - x2) * (y1 - y2)
        conc += d > 0; disc += d < 0
    return conc, disc


def classify(tok, t):
    s = tok.decode([t])
    if re.fullmatch(r"\s*\d+", s): return "digit"
    if "\\" in s or s.strip() in "(){}[]^_=+-*/<>$": return "latex/math"
    if s.strip() in {",", ".", ";", ":"}: return "punct"
    if s.startswith(" ") and s[1:2].isalpha(): return "word-start"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--model_tok", default="ckpts_hf/math_k3_ds_step_300")
    ap.add_argument("--h", type=int, default=6)
    ap.add_argument("--n_qual", type=int, default=3)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model_tok)
    rows = [json.loads(l) for l in open(args.file)]
    B = [b for r in rows for b in r["branches"]]
    h = args.h

    t_conc = {k: 0 for k in ("prefix", "match", "run", "mrank")}
    t_disc = {k: 0 for k in ("prefix", "match", "run", "mrank")}
    dwin, dlose = [], []
    cond_wins = 0
    n_wins = 0
    gain_cls = Counter()
    quals = []
    for b in B:
        ns = b["n_shift"]
        fut = b["future"]
        van_f = feats(b["vanilla"]["toks"], fut, ns)
        van_o = b["vanilla"]["cum"][h-1] / h
        cands = []
        for c in b["cands"]:
            f = feats(c["toks"], fut, ns)
            f["mrank"] = st.mean(c["rank"]) if c["rank"] else 0.0
            f["out"] = c["cum"][h-1] / h
            f["toks"] = c["toks"]
            cands.append(f)
        allc = [{"prefix": van_f["prefix"], "match": van_f["match"], "run": van_f["run"],
                 "mrank": 0.0, "out": van_o}] + cands
        outs = [c["out"] for c in allc]
        for k in t_conc:
            c_, d_ = tau([c[k] for c in allc], outs)
            t_conc[k] += c_; t_disc[k] += d_
        wi = max(range(len(cands)), key=lambda i: cands[i]["out"])
        w = cands[wi]
        if w["out"] > van_o:
            n_wins += 1
            dwin.append({k: w[k] - van_f[k] for k in ("prefix", "match", "run")})
            if w["match"] <= van_f["match"]:
                cond_wins += 1
            for j in range(ns):
                if w["mvec"][j] and not van_f["mvec"][j]:
                    gain_cls[classify(tok, w["toks"][j])] += 1
            quals.append((w["out"] - van_o, b, wi))
        losers = [c for c in cands if c["out"] < van_o]
        for c in losers:
            dlose.append({k: c[k] - van_f[k] for k in ("prefix", "match", "run")})

    print(f"=== {args.file}: {len(B)} branch points, winners beat vanilla at {n_wins}")
    print("\nWithin-state predictiveness of features for horizon outcome (Kendall-tau):")
    for k in t_conc:
        t = (t_conc[k] - t_disc[k]) / max(1, t_conc[k] + t_disc[k])
        print(f"  tau(outcome, {k:6s}) = {t:+.3f}")
    md = lambda L, k: st.mean(d[k] for d in L) if L else float("nan")
    print(f"\nWinner − vanilla deltas:  prefix {md(dwin,'prefix'):+.2f}  "
          f"match {md(dwin,'match'):+.2f}  run {md(dwin,'run'):+.2f}")
    print(f"Loser  − vanilla deltas:  prefix {md(dlose,'prefix'):+.2f}  "
          f"match {md(dlose,'match'):+.2f}  run {md(dlose,'run'):+.2f}")
    print(f"Conditioning wins (winner matches future <= vanilla): {cond_wins}/{n_wins}")
    print(f"Token classes where winner matches future & vanilla doesn't: "
          f"{dict(gain_cls.most_common(6))}")

    quals.sort(key=lambda x: -x[0])
    for adv, b, wi in quals[: args.n_qual]:
        ns = b["n_shift"]; fut = b["future"]
        w = b["cands"][wi]
        print("\n" + "#" * 90)
        print(f"branch@pos={b['pos']}  winner adv=+{adv:.2f} TPF over h  "
              f"(vanilla {b['vanilla']['cum'][h-1]}/{h} vs winner {w['cum'][h-1]}/{h} tokens)")
        print(f"CTX  …{tok.decode(b['ctx_tail'][-24:])!r}")
        print(f"FUT  {tok.decode(fut[:ns])!r}")
        def render(toks):
            out = []
            for j in range(ns):
                s = tok.decode([toks[j]]).replace("\n", "\\n")
                mark = "✓" if (j < len(fut) and toks[j] == fut[j]) else "·"
                out.append(f"{mark}{s}")
            return " ".join(out)
        print(f"VAN  {render(b['vanilla']['toks'])}")
        print(f"WIN  {render(w['toks'])}")


if __name__ == "__main__":
    main()
