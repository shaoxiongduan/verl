"""Inspect cascade trajectories: how much of each K-token draft is 'correct prefix'
(matches target_argmax / AR continuation) vs 'noise' (diverges from AR)?

For each trajectory record:
  draft[i]         = what the JacobiProposer guessed at position i (from prev iter)
  target_argmax[i] = what the model now says is the AR token at position i
  prefix_match_len = leading positions where draft[i] == target_argmax[i]

The 'correct prefix' is the cascade-converged region. The remaining
K - prefix_match_len positions are where the draft diverges from AR — the
'true noise' that cons loss should focus on.
"""
import json, glob, sys
from collections import Counter, defaultdict
from pathlib import Path

# Find trajectories from a smoke or production run.
candidates = sorted(glob.glob(
    "/mnt/weka/home/hao.zhang/shao/verl/ckpts/jacobi_onpolicy_smoke/"
    "jf_math_jacobi_onpolicy_smoke5_rot/jacobi_traj.jsonl.*"
))
if not candidates:
    print("No trajectory files found.")
    sys.exit(1)
print(f"Loading from: {candidates[0]}")

# Load AutoTokenizer to decode
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(
    "/mnt/weka/home/hao.zhang/.cache/huggingface/hub/models--JacobiForcing--JacobiForcing_Math_7B_v1/snapshots/e65283c1b3d205b23c2bdf9946158035c409d3a1"
)

# Aggregate stats
prefix_match_dist = Counter()
draft_len_dist = Counter()
n_acc_dist = Counter()
per_req_records = defaultdict(list)

with open(candidates[0]) as f:
    for line in f:
        rec = json.loads(line)
        # Old-format records have arrays per slot
        if "req_ids" not in rec:
            continue
        for slot, rid in enumerate(rec["req_ids"]):
            if rid.startswith("slot_"):
                continue  # warmup
            k = rec["num_draft"][slot]
            if k == 0:
                continue
            draft = rec["draft"][slot]
            targ = rec["target_argmax"][slot]
            n_acc = rec["n_acc"][slot]
            if len(draft) != k or len(targ) != k:
                continue
            # Compute leading match between draft and target_argmax
            prefix = 0
            for d, t in zip(draft, targ):
                if d == t:
                    prefix += 1
                else:
                    break
            prefix_match_dist[prefix] += 1
            draft_len_dist[k] += 1
            n_acc_dist[n_acc] += 1
            per_req_records[rid].append((draft, targ, n_acc, prefix))

print(f"\n=== Total trajectory records: {sum(prefix_match_dist.values())} ===")
print(f"K (draft length) distribution: {dict(draft_len_dist)}")
print(f"n_acc (accepted spec tokens) distribution:")
total = sum(n_acc_dist.values())
for k in sorted(n_acc_dist.keys()):
    print(f"  n_acc={k}: {n_acc_dist[k]} ({100*n_acc_dist[k]/total:.1f}%)")

print(f"\nprefix_match_len (leading draft positions where draft[i]==target_argmax[i]):")
for k in sorted(prefix_match_dist.keys())[:20]:
    print(f"  prefix={k:>3}: {prefix_match_dist[k]:>6} ({100*prefix_match_dist[k]/total:.1f}%)")
larger = sum(c for k, c in prefix_match_dist.items() if k > 20)
print(f"  prefix>20: {larger}")

print(f"\nprefix == n_acc check: should be CLOSE (cascade convergence ≈ accepted)")
match = 0
diff = []
for trajs in per_req_records.values():
    for draft, targ, n_acc, prefix in trajs:
        if prefix == n_acc:
            match += 1
        diff.append(prefix - n_acc)
print(f"  exact matches: {match}/{total} ({100*match/total:.1f}%)")
print(f"  mean (prefix - n_acc): {sum(diff)/len(diff):.2f}")

# Show 3 concrete examples
print(f"\n=== 3 example trajectories (decoded) ===")
shown = 0
for rid, trajs in per_req_records.items():
    for draft, targ, n_acc, prefix in trajs:
        if prefix < 2 or prefix > 8:
            continue
        K = len(draft)
        # Decode each token individually so we can see boundaries
        draft_strs = [tok.decode([t]).replace("\n", "\\n") for t in draft]
        targ_strs  = [tok.decode([t]).replace("\n", "\\n") for t in targ]
        markers    = ["✓" if d == t else "✗" for d, t in zip(draft, targ)]
        print(f"\n  req={rid[:8]}  K={K}  n_acc={n_acc}  prefix_match={prefix}")
        print(f"    draft :  " + " ".join(f"{s:>10}" for s in draft_strs[:12]))
        print(f"    target:  " + " ".join(f"{s:>10}" for s in targ_strs[:12]))
        print(f"    match :  " + " ".join(f"{s:>10}" for s in markers[:12]))
        shown += 1
        if shown >= 3:
            sys.exit(0)
