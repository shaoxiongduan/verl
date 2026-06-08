"""Load on-policy Jacobi trajectory JSONL into per-request `List[Trajectory]`.

The vLLM jacobi_vllm_plugin writes one JSONL record per spec-decode forward to
`{traj_path}.<PID>`. Each record has per-slot arrays:

  {"iter": int,
   "req_ids":     [str, ...],           # stable vLLM request IDs per slot
   "num_draft":   [int, ...],           # K per slot (0 = no spec)
   "n_acc":       [int, ...],           # accepted speculative tokens per slot
   "draft":       [[int, ...], ...],    # K-token draft per slot
   "target_argmax": [[int, ...], ...],  # model argmax at each spec position
   "bonus":       [int, ...]}           # = target_argmax[n_acc] per slot

For each request, the position advances by `n_acc + 1` per forward (the +1 is
the bonus token vLLM always commits). This module reconstructs
(start, end, draft) tuples per request as `Trajectory` objects.

USAGE
-----

    from scripts.consistency.trajectory_loader import load_trajectories
    by_req = load_trajectories("/path/to/traj.jsonl")
    # by_req: dict[req_id -> list[Trajectory]] sorted by start

    from scripts.consistency.onpolicy_select import greedy_non_overlapping
    selected = greedy_non_overlapping(by_req[req_id], response_len=len(response))
"""

from __future__ import annotations

import json
import glob
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

from scripts.consistency.onpolicy_select import Trajectory


def parse_per_request_records(
    records: list[dict],
    *,
    use_target_argmax: bool = False,
) -> list[Trajectory]:
    """Convert in-memory trajectory records (one request's worth) into a sorted
    list of `Trajectory` objects.

    Each record is the per-spec-decode-iter dict produced by the
    `jacobi_vllm_plugin` AFTER the per-request restructuring (i.e., NOT the
    legacy per-forward batched format with `req_ids`/`num_draft` arrays). Each
    record has: iter, num_draft (int), n_acc (int), draft (list[int]),
    target_argmax (list[int]), bonus (int).

    Position reconstruction: at each iter, the request advances by `n_acc + 1`
    tokens (n_acc accepted spec tokens + 1 bonus). The trajectory covers
    [pos_before, pos_before + num_draft) at that iter.

    This is the in-memory analogue of `load_trajectories(file_path)`: same
    output, no file I/O. Use this when trajectories arrive via
    `TokenOutput.extra_fields["jacobi_trajectories"]`.
    """
    pos = 0
    out: list[Trajectory] = []
    for rec in records:
        try:
            k = int(rec["num_draft"])
        except (KeyError, TypeError, ValueError):
            continue
        if k <= 0:
            continue
        tokens_src = rec.get("target_argmax") if use_target_argmax else rec.get("draft")
        if tokens_src is None or len(tokens_src) != k:
            continue
        try:
            n_acc = int(rec["n_acc"])
        except (KeyError, TypeError, ValueError):
            n_acc = 0
        # `n_acc` = positions where the cascade COMMITTED this iter's draft
        # (= the prefix that matches the final response by construction).
        # Cons loss masks these positions out so dflash decay applies only
        # to the truly-noisy tail (positions [n_acc, K)).
        # Also pull target_argmax (the model's predictions at each spec
        # position at this iter) for multi-tile noise diversification.
        target_argmax = rec.get("target_argmax")
        target_argmax_t = None
        if target_argmax is not None and len(target_argmax) == k:
            target_argmax_t = tuple(int(x) for x in target_argmax)
        out.append(Trajectory(
            start=pos,
            end=pos + k,
            draft=tuple(int(x) for x in tokens_src),
            iter_id=int(rec.get("iter", -1)),
            prefix_match_len=max(0, min(k, n_acc)),
            target_argmax=target_argmax_t,
        ))
        pos += n_acc + 1
    out.sort(key=lambda t: t.start)
    return out


def _iter_records(paths: Iterable[str]) -> Iterable[dict]:
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # Tolerate a half-written final line (writer may have crashed).
                    continue


def _resolve_paths(traj_path: str) -> List[str]:
    """Resolve a base traj_path to all per-PID JSONL files.

    The plugin writes `{traj_path}.{PID}` per process. If the caller gives the
    base path, glob the .{PID} suffixes. If they give a literal file, use it.
    """
    if os.path.isfile(traj_path):
        return [traj_path]
    # Try glob `{traj_path}.*` for per-PID files.
    candidates = sorted(glob.glob(traj_path + ".*"))
    return candidates


def load_trajectories(
    traj_path: str,
    *,
    iters: Optional[Iterable[int]] = None,
    use_target_argmax: bool = False,
) -> Dict[str, List[Trajectory]]:
    """Read JSONL trajectory file(s); return per-request lists of Trajectory.

    Args:
      traj_path: either a single file or a base path whose per-PID files we glob.
      iters: optional iter id whitelist; only records whose `iter` is in this
        set are loaded (rest skipped). Default: all.
      use_target_argmax: if True, the Trajectory.draft tokens are
        `target_argmax[i]` (= "what the model would produce given this prefix")
        rather than `draft[i]` (= what the proposer actually proposed).
        For cons-RL training, `draft` is the right choice — it's the cascade
        input the model sees. `target_argmax` is the AR target.

    Returns:
      Dict mapping str(request_id) -> sorted list of Trajectory (by start).
      Each Trajectory has start, end=start+num_draft, and the draft tokens.

    Position reconstruction:
      For each request, the next forward starts at:
        pos_after = pos_before + n_acc[i] + 1  (the +1 is the bonus token)
      So the trajectory for THAT forward covers [pos_before, pos_before + K).
      We initialize pos_before = 0 (= first response token) for each new req.

    Records where num_draft[i] == 0 (request didn't speculate that forward,
    e.g., was prefilling) are skipped and do not advance the position counter.
    """
    paths = _resolve_paths(traj_path)
    iter_set = set(iters) if iters is not None else None

    # request_id -> current position in its response
    pos_by_req: Dict[str, int] = defaultdict(int)
    out: Dict[str, List[Trajectory]] = defaultdict(list)

    for rec in _iter_records(paths):
        if iter_set is not None and rec["iter"] not in iter_set:
            continue
        req_ids = rec.get("req_ids")
        if req_ids is None:
            # Plugin without req_ids patch — can't load multi-request files.
            raise ValueError(
                "Trajectory record missing 'req_ids' field. "
                "Re-install the jacobi_vllm_plugin (the model_runner patch adds it)."
            )
        num_draft = rec["num_draft"]
        n_acc = rec["n_acc"]
        drafts = rec["draft"]
        targets = rec.get("target_argmax", [None] * len(num_draft))
        iter_id = int(rec["iter"])

        for slot, rid in enumerate(req_ids):
            k = int(num_draft[slot])
            if k <= 0:
                # No speculation in this slot this forward (e.g., prefill).
                # Don't advance position; cascade hasn't run for this req yet.
                continue
            draft = drafts[slot]
            if len(draft) != k:
                # Shouldn't happen, but be defensive against malformed records.
                continue
            start = pos_by_req[rid]
            end = start + k
            tokens = tuple(int(x) for x in (targets[slot] if use_target_argmax else draft))
            if len(tokens) != k:
                continue
            out[rid].append(Trajectory(
                start=start,
                end=end,
                draft=tokens,
                iter_id=iter_id,
            ))
            # Advance position by accepted + bonus.
            pos_by_req[rid] = start + int(n_acc[slot]) + 1

    # Sort each request's trajectories by start position.
    return {rid: sorted(trajs, key=lambda t: t.start) for rid, trajs in out.items()}


def trajectory_stats(by_req: Dict[str, List[Trajectory]]) -> dict:
    """Summary stats over a loaded trajectory set."""
    if not by_req:
        return {"n_requests": 0, "n_trajectories": 0}
    n_req = len(by_req)
    n_traj = sum(len(v) for v in by_req.values())
    lens = [t.length for trajs in by_req.values() for t in trajs]
    max_pos = max((trajs[-1].end if trajs else 0) for trajs in by_req.values())
    avg_per_req = n_traj / n_req if n_req else 0
    return {
        "n_requests": n_req,
        "n_trajectories": n_traj,
        "avg_trajs_per_req": round(avg_per_req, 2),
        "draft_len_min": min(lens) if lens else 0,
        "draft_len_max": max(lens) if lens else 0,
        "max_response_pos": max_pos,
    }


# ============================================================
# Tests — run as: python -m scripts.consistency.trajectory_loader
# ============================================================

def _write_test_jsonl(path: str, records: List[dict]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def test_single_req_dense_cascade():
    """One request, K=4, simulate 4 forwards advancing by varying n_acc."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "traj.jsonl")
        # Request "r0" speculates K=4 each forward. n_acc varies.
        # Forward 1: pos starts at 0, accept 2, advance to 3.
        # Forward 2: pos starts at 3, accept 1, advance to 5.
        # Forward 3: pos starts at 5, accept 3, advance to 9.
        # Forward 4: pos starts at 9, accept 0, advance to 10.
        records = [
            {"iter": 1, "req_ids": ["r0"], "num_draft": [4], "n_acc": [2],
             "draft": [[10, 11, 12, 13]], "target_argmax": [[20, 21, 22, 23]], "bonus": [99]},
            {"iter": 2, "req_ids": ["r0"], "num_draft": [4], "n_acc": [1],
             "draft": [[14, 15, 16, 17]], "target_argmax": [[24, 25, 26, 27]], "bonus": [99]},
            {"iter": 3, "req_ids": ["r0"], "num_draft": [4], "n_acc": [3],
             "draft": [[18, 19, 20, 21]], "target_argmax": [[28, 29, 30, 31]], "bonus": [99]},
            {"iter": 4, "req_ids": ["r0"], "num_draft": [4], "n_acc": [0],
             "draft": [[22, 23, 24, 25]], "target_argmax": [[32, 33, 34, 35]], "bonus": [99]},
        ]
        _write_test_jsonl(path, records)
        by_req = load_trajectories(path)
        _assert(list(by_req.keys()) == ["r0"], f"keys: {list(by_req.keys())}")
        trajs = by_req["r0"]
        _assert(len(trajs) == 4, f"n_trajs: {len(trajs)}")
        # start positions: 0, 3, 5, 9
        starts = [t.start for t in trajs]
        _assert(starts == [0, 3, 5, 9], f"starts: {starts}")
        ends = [t.end for t in trajs]
        _assert(ends == [4, 7, 9, 13], f"ends: {ends}")
        drafts = [t.draft for t in trajs]
        _assert(drafts[0] == (10, 11, 12, 13), f"draft 0: {drafts[0]}")
        _assert(drafts[3] == (22, 23, 24, 25), f"draft 3: {drafts[3]}")
        print(f"  single_req_dense: 4 trajs, starts={starts}, ends={ends}  ✓")


def test_multi_req_interleaved():
    """Two requests sharing forwards; positions tracked independently."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "traj.jsonl")
        records = [
            # Forward 1: r0 and r1 both speculate K=2.
            {"iter": 1, "req_ids": ["r0", "r1"], "num_draft": [2, 2],
             "n_acc": [1, 0], "draft": [[100, 101], [200, 201]],
             "target_argmax": [[110, 111], [210, 211]], "bonus": [99, 99]},
            # Forward 2: only r0 speculates; r1 is prefilling (num_draft=0).
            {"iter": 2, "req_ids": ["r0", "r1"], "num_draft": [2, 0],
             "n_acc": [2, 0], "draft": [[102, 103], []],
             "target_argmax": [[112, 113], []], "bonus": [99, -1]},
            # Forward 3: both speculate again.
            {"iter": 3, "req_ids": ["r0", "r1"], "num_draft": [2, 2],
             "n_acc": [0, 1], "draft": [[104, 105], [202, 203]],
             "target_argmax": [[114, 115], [212, 213]], "bonus": [99, 99]},
        ]
        _write_test_jsonl(path, records)
        by_req = load_trajectories(path)
        _assert(set(by_req.keys()) == {"r0", "r1"}, f"keys: {by_req.keys()}")

        r0 = by_req["r0"]
        # r0 positions: 0 -> +2 (1 acc + 1 bonus) -> 2 -> +3 (2+1) -> 5 -> +1 (0+1) -> 6
        _assert([t.start for t in r0] == [0, 2, 5], f"r0 starts: {[t.start for t in r0]}")
        _assert([t.end for t in r0] == [2, 4, 7], f"r0 ends: {[t.end for t in r0]}")

        r1 = by_req["r1"]
        # r1 positions: 0 -> +1 (0+1) -> 1 ... skip forward 2 (num_draft=0) ... -> 1 -> +2 (1+1) -> 3
        _assert([t.start for t in r1] == [0, 1], f"r1 starts: {[t.start for t in r1]}")
        _assert([t.end for t in r1] == [2, 3], f"r1 ends: {[t.end for t in r1]}")

        print(f"  multi_req: r0 starts={[t.start for t in r0]}, r1 starts={[t.start for t in r1]}  ✓")


def test_use_target_argmax_flag():
    """use_target_argmax=True swaps draft tokens for the model's argmax."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "traj.jsonl")
        records = [
            {"iter": 1, "req_ids": ["r0"], "num_draft": [3], "n_acc": [1],
             "draft": [[10, 11, 12]], "target_argmax": [[20, 21, 22]], "bonus": [99]},
        ]
        _write_test_jsonl(path, records)
        by_req_draft = load_trajectories(path)
        by_req_targ = load_trajectories(path, use_target_argmax=True)
        _assert(by_req_draft["r0"][0].draft == (10, 11, 12), "draft mode")
        _assert(by_req_targ["r0"][0].draft == (20, 21, 22), "target_argmax mode")
        print("  use_target_argmax: ✓")


def test_missing_req_ids_field():
    """Old-format records without req_ids should raise a clear error."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "traj.jsonl")
        records = [
            {"iter": 1, "num_draft": [4], "n_acc": [2],
             "draft": [[10, 11, 12, 13]], "target_argmax": [[20, 21, 22, 23]], "bonus": [99]},
        ]
        _write_test_jsonl(path, records)
        try:
            load_trajectories(path)
            _assert(False, "should have raised")
        except ValueError as e:
            _assert("req_ids" in str(e), f"error msg: {e}")
        print("  missing_req_ids raises: ✓")


def test_glob_per_pid_files():
    """traj_path resolves per-PID files via glob."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        base = os.path.join(td, "traj.jsonl")
        # Two per-PID files, simulating two vLLM EngineCore subprocesses.
        _write_test_jsonl(base + ".111", [
            {"iter": 1, "req_ids": ["A"], "num_draft": [2], "n_acc": [0],
             "draft": [[1, 2]], "target_argmax": [[3, 4]], "bonus": [99]},
        ])
        _write_test_jsonl(base + ".222", [
            {"iter": 1, "req_ids": ["B"], "num_draft": [2], "n_acc": [1],
             "draft": [[5, 6]], "target_argmax": [[7, 8]], "bonus": [99]},
        ])
        by_req = load_trajectories(base)
        _assert(set(by_req.keys()) == {"A", "B"}, f"keys: {by_req.keys()}")
        _assert(len(by_req["A"]) == 1 and len(by_req["B"]) == 1, "one traj each")
        print("  glob_per_pid: ✓")


def test_zero_num_draft_skipped():
    """num_draft=0 entries don't create a trajectory but also don't advance pos."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "traj.jsonl")
        records = [
            {"iter": 1, "req_ids": ["r0"], "num_draft": [4], "n_acc": [2],
             "draft": [[10, 11, 12, 13]], "target_argmax": [[20, 21, 22, 23]], "bonus": [99]},
            # No speculation forward.
            {"iter": 2, "req_ids": ["r0"], "num_draft": [0], "n_acc": [0],
             "draft": [[]], "target_argmax": [[]], "bonus": [-1]},
            {"iter": 3, "req_ids": ["r0"], "num_draft": [4], "n_acc": [1],
             "draft": [[30, 31, 32, 33]], "target_argmax": [[40, 41, 42, 43]], "bonus": [99]},
        ]
        _write_test_jsonl(path, records)
        by_req = load_trajectories(path)
        trajs = by_req["r0"]
        _assert(len(trajs) == 2, f"n_trajs: {len(trajs)}")
        # Pos after first traj: 0 + 2 + 1 = 3. So second traj starts at 3.
        _assert([t.start for t in trajs] == [0, 3], f"starts: {[t.start for t in trajs]}")
        print("  zero_num_draft_skipped: ✓")


def test_stats():
    """trajectory_stats returns reasonable numbers."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "traj.jsonl")
        records = [
            {"iter": 1, "req_ids": ["A", "B"], "num_draft": [4, 4], "n_acc": [2, 1],
             "draft": [[1, 2, 3, 4], [5, 6, 7, 8]], "target_argmax": [[1, 2, 3, 4], [5, 6, 7, 8]], "bonus": [9, 9]},
            {"iter": 2, "req_ids": ["A"], "num_draft": [4], "n_acc": [0],
             "draft": [[10, 11, 12, 13]], "target_argmax": [[10, 11, 12, 13]], "bonus": [99]},
        ]
        _write_test_jsonl(path, records)
        by_req = load_trajectories(path)
        stats = trajectory_stats(by_req)
        _assert(stats["n_requests"] == 2, f"n_requests: {stats}")
        _assert(stats["n_trajectories"] == 3, f"n_trajectories: {stats}")
        _assert(stats["draft_len_min"] == 4 and stats["draft_len_max"] == 4, f"lens: {stats}")
        print(f"  stats: {stats}  ✓")


def test_integration_with_selector():
    """End-to-end: load trajectories, run greedy selector, verify disjoint."""
    import tempfile
    from scripts.consistency.onpolicy_select import greedy_non_overlapping
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "traj.jsonl")
        # Simulate 8 forwards on one request, K=16, n_acc varying.
        # Position advances by n_acc+1 per forward.
        records = []
        n_accs = [3, 1, 5, 0, 2, 4, 1, 3]
        K = 16
        pos = 0
        for it, na in enumerate(n_accs):
            records.append({
                "iter": it + 1,
                "req_ids": ["r0"],
                "num_draft": [K],
                "n_acc": [na],
                "draft": [[1000 + it * 100 + j for j in range(K)]],
                "target_argmax": [[2000 + it * 100 + j for j in range(K)]],
                "bonus": [99],
            })
            pos += na + 1
        _write_test_jsonl(path, records)
        by_req = load_trajectories(path)
        trajs = by_req["r0"]
        starts = [t.start for t in trajs]
        # Pos: 0, 4, 6, 12, 13, 16, 21, 23
        expected_starts = [0, 4, 6, 12, 13, 16, 21, 23]
        _assert(starts == expected_starts, f"starts: {starts}")
        # All trajs have length K=16, end = start + 16
        ends = [t.end for t in trajs]
        _assert(ends == [s + K for s in expected_starts], f"ends: {ends}")
        # Greedy non-overlapping pick:
        # (0, 16) -> skip (4,20), (6,22), (12,28), (13,29) -> pick (16, 32) -> skip (21,37) -> pick (23,39)
        # Wait — (23,39).start=23 < (16,32).end=32, so (23,39) overlaps. Skip it.
        # Actually let's trace:
        # picked=[(0,16)]. Next start>=16 needed. (4,20) skip. (6,22) skip. (12,28) skip. (13,29) skip.
        # (16,32): start=16 >= 16, pick. picked=[(0,16), (16,32)]. Next start>=32 needed.
        # (21,37) skip. (23,39) skip.
        sel = greedy_non_overlapping(trajs)
        sel_starts = [t.start for t in sel]
        _assert(sel_starts == [0, 16], f"selector picked: {sel_starts}")
        # Verify drafts preserved through selection
        _assert(sel[0].draft[0] == 1000, f"sel[0] draft[0]: {sel[0].draft[0]}")
        _assert(sel[1].draft[0] == 1500, f"sel[1] draft[0]: {sel[1].draft[0]}")  # iter 6 -> 2000 + 5*100... wait
        # iter index 5 (the 6th record, K=16, draft = [1500..1515])
        # but we picked the one starting at pos 16 -> that's record index 5 (since pos 16 happens at iter 6 = index 5)
        # Let me recompute: pos starts at 0. After iter 1 (n_acc=3): pos=4. iter 2 (n_acc=1): pos=6.
        # iter 3 (n_acc=5): pos=12. iter 4 (n_acc=0): pos=13. iter 5 (n_acc=2): pos=16. iter 6 (n_acc=4): pos=21.
        # So pos 16 starts at record index 5 (iter 6). draft for iter 6 = [1000 + 5*100 + j] = [1500+j].
        # sel[1].draft = (1500, 1501, ..., 1515). ✓
        print(f"  integration_with_selector: {len(trajs)} trajs -> selected {len(sel)} non-overlapping  ✓")


if __name__ == "__main__":
    print("Running trajectory_loader tests:")
    test_single_req_dense_cascade()
    test_multi_req_interleaved()
    test_use_target_argmax_flag()
    test_missing_req_ids_field()
    test_glob_per_pid_files()
    test_zero_num_draft_skipped()
    test_stats()
    test_integration_with_selector()
    print("\nAll tests passed.")
