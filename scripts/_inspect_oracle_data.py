"""Inspect the big-data dataset for:
  - response length per prompt (did all hit max_new cap?)
  - repetition rate in committed tokens
  - distribution of n_acc per cycle (any signs of degenerate-loop)
  - per-prompt stats
"""
from __future__ import annotations
import argparse
import numpy as np
from collections import Counter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--tokenizer", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    z = np.load(args.data)
    pi = z["prompt_idx"]; ci = z["cycle_idx"]
    nacc = z["n_acc"]
    warm = z["warm_argmax"]; verify = z["verify_argmax"]
    K = warm.shape[1]
    print(f"data: {args.data}")
    print(f"total cycles: {len(pi)}")
    print(f"unique prompts: {len(set(pi.tolist()))}")
    print(f"K = {K}")

    # Per-prompt: response length (cycles) and total tokens
    from collections import defaultdict
    per_prompt = defaultdict(list)
    for i in range(len(pi)):
        per_prompt[int(pi[i])].append((int(ci[i]), int(nacc[i]), warm[i], verify[i]))
    for p in per_prompt:
        per_prompt[p].sort(key=lambda x: x[0])

    print(f"\nPer-prompt response stats:")
    print(f"{'prompt':>7} {'n_cyc':>7} {'tot_tok':>9} {'mean_nacc':>10} {'n_acc=K (full conv)':>20} {'last_cyc_n_acc':>15}")
    cycles_per_prompt = []
    tokens_per_prompt = []
    for p, cycles in sorted(per_prompt.items()):
        n_cyc = len(cycles)
        tot_tok = sum(c[1] + 1 for c in cycles)  # commits = n_acc+1
        mean_n = np.mean([c[1] for c in cycles])
        n_full = sum(1 for c in cycles if c[1] == K)
        last_n = cycles[-1][1]
        cycles_per_prompt.append(n_cyc)
        tokens_per_prompt.append(tot_tok)
        print(f"{p:>7} {n_cyc:>7} {tot_tok:>9} {mean_n:>10.2f} {n_full:>20} {last_n:>15}")

    print(f"\nResponse length summary (across {len(per_prompt)} prompts):")
    print(f"  tokens per prompt: min={min(tokens_per_prompt)}  max={max(tokens_per_prompt)}  median={int(np.median(tokens_per_prompt))}  mean={np.mean(tokens_per_prompt):.0f}")
    print(f"  cycles per prompt: min={min(cycles_per_prompt)}  max={max(cycles_per_prompt)}  median={int(np.median(cycles_per_prompt))}  mean={np.mean(cycles_per_prompt):.1f}")

    # Repetition detection in committed tokens (n_acc + 1) over each prompt
    print(f"\nRepetition analysis:")
    rep_per_prompt = []
    for p, cycles in sorted(per_prompt.items()):
        committed = []
        for cyc_idx, n_acc, w, v in cycles:
            committed.extend(w[:n_acc].tolist())
            if n_acc < K:
                committed.append(int(v[n_acc]))
            else:
                pass
        if len(committed) >= 50:
            # Count 5-gram repetitions
            ngs = [tuple(committed[i:i+5]) for i in range(len(committed)-5)]
            most = Counter(ngs).most_common(1)
            if most:
                top_ng, top_c = most[0]
                rep_rate = top_c / max(1, len(ngs))
                rep_per_prompt.append((p, rep_rate, top_c, len(committed), top_ng))

    rep_per_prompt.sort(key=lambda x: -x[1])
    print(f"  Top-10 repeating prompts (5-gram repetition):")
    print(f"  {'prompt':>7} {'rep_rate':>9} {'count':>6} {'tot_tok':>9}  most_common_5gram_first_tok_ids")
    for p, rr, c, n, ng in rep_per_prompt[:10]:
        print(f"  {p:>7} {rr:>9.3f} {c:>6} {n:>9}  {ng[:3]}...")

    n_with_rep = sum(1 for r in rep_per_prompt if r[1] > 0.05)
    print(f"\n  Prompts with >5% 5-gram repetition: {n_with_rep} / {len(rep_per_prompt)}")
    print(f"  mean rep_rate across prompts: {np.mean([r[1] for r in rep_per_prompt]):.3f}")

    # Check if last cycles tend to have very high n_acc (indicating model in repetition mode)
    print(f"\nMean n_acc by cycle position within prompt:")
    cyc_buckets = defaultdict(list)
    for p, cycles in per_prompt.items():
        for i, (cyc_idx, n_acc, _, _) in enumerate(cycles):
            frac = i / len(cycles)
            bucket = min(9, int(frac * 10))
            cyc_buckets[bucket].append(n_acc)
    print(f"  pos_in_prompt:        ", end="")
    for b in range(10):
        print(f"{b/10:>4.1f}-{(b+1)/10:>4.1f} ", end="")
    print()
    print(f"  mean n_acc:           ", end="")
    for b in range(10):
        print(f"{np.mean(cyc_buckets[b]):>9.2f} ", end="")
    print()
    print(f"  n cycles in bucket:   ", end="")
    for b in range(10):
        print(f"{len(cyc_buckets[b]):>9d} ", end="")
    print()


if __name__ == "__main__":
    main()
