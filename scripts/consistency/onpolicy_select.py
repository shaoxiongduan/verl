"""Greedy non-overlapping selection of on-policy Jacobi trajectories.

A Jacobi rollout produces many (start, end, draft) trajectories per response —
one per spec-decode forward call. They densely overlap because spec-decoding
advances by `n_acc + 1` tokens per iter (typically 3-6 at K=16), so consecutive
trajectories share most of their span.

For cons loss training we pack `[prompt | noisy_block | clean_block | noisy ...]`
where each (noisy, clean) pair covers a contiguous N-position range of the
response. The clean is `response[start:end]` and the noisy is the trajectory's
draft. To avoid duplicating cons supervision on the same response positions
within one packed sequence, we pick a SUBSET of trajectories whose [start, end)
ranges are mutually disjoint.

Greedy algorithm (by start):
  1. Sort trajectories by start position.
  2. Pick the FIRST trajectory; remember its end.
  3. Skip subsequent trajectories whose start < last_picked_end.
  4. Pick the next trajectory with start >= last_picked_end. Repeat.

There will be gaps (response positions not covered by any picked trajectory).
That is intentional; cons supervision on the picked subset still trains the
model on the actual cascade-evolved input distribution at THOSE positions.

Conventions:
  - Intervals are half-open [start, end). end = start + len(draft).
  - Two intervals A=[a, b), B=[c, d) overlap iff a < d AND c < b.
  - Non-overlap: c >= b (B starts at or after A's end).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class Trajectory:
    """One Jacobi cascade snapshot covering a half-open range of response positions."""
    start: int                  # inclusive start position in the response
    end: int                    # exclusive end position; end - start == len(draft)
    draft: tuple[int, ...]      # the noisy draft tokens proposed at these positions
    iter_id: int = -1           # global iter id from the rollout (for traceability; optional)
    prefix_match_len: int = 0   # leading draft positions that match the response by
                                 # cascade construction (= n_acc, the spec-accepted count
                                 # at this iter). draft[:prefix_match_len] == response at
                                 # the same positions. Cons-loss masks these out.
    target_argmax: tuple[int, ...] | None = None
        # the model's argmax predictions at each position [start, end) at this
        # iter — different from `draft` (the input drafts at that iter).
        # Across iters that cover the same response position, these argmax
        # values form an "on-policy alternative" pool used for multi-tile
        # noise diversification in cons training.

    def __post_init__(self):
        if self.end <= self.start:
            raise ValueError(f"end ({self.end}) must be > start ({self.start})")
        if len(self.draft) != self.end - self.start:
            raise ValueError(
                f"draft length ({len(self.draft)}) must equal end - start "
                f"({self.end - self.start})"
            )
        if self.target_argmax is not None and len(self.target_argmax) != self.end - self.start:
            raise ValueError(
                f"target_argmax length ({len(self.target_argmax)}) must equal "
                f"end - start ({self.end - self.start}) if provided"
            )

    @property
    def length(self) -> int:
        return self.end - self.start


def greedy_non_overlapping(
    trajectories: Iterable[Trajectory],
    *,
    response_len: int | None = None,
) -> List[Trajectory]:
    """Return a non-overlapping subset of `trajectories`, picked greedily by start.

    Args:
      trajectories: iterable of Trajectory; positions are in the response coords.
      response_len: optional upper bound on positions; any trajectory with
        end > response_len is dropped (cannot supervise positions past response).

    Returns:
      List of picked Trajectories. Disjoint intervals. Sorted by start.

    Properties (verifiable by test):
      - selected[i].end <= selected[i+1].start  (strict non-overlap)
      - selected starts are increasing
      - if A has the smallest start in the input, A is always picked
      - greedy property: at every step, we pick the next trajectory with the
        smallest valid start

    Note: greedy-by-start is NOT optimal for "max coverage" but matches what
    cons-RL training needs — one cons sample per disjoint chunk, in pack order.
    """
    sorted_traj = sorted(trajectories, key=lambda t: (t.start, t.end))
    selected: List[Trajectory] = []
    last_end = 0
    for t in sorted_traj:
        if response_len is not None and t.end > response_len:
            continue
        if t.start < last_end:
            continue                    # overlaps with previously picked
        selected.append(t)
        last_end = t.end
    return selected


# ============================================================
# Test harness — run as a script: python -m scripts.consistency.onpolicy_select
# ============================================================

def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _t(start: int, end: int, label: int = 0) -> Trajectory:
    """Test helper: build a trajectory whose draft is just [label] * length."""
    return Trajectory(start=start, end=end, draft=tuple([label] * (end - start)), iter_id=label)


def _check_disjoint_sorted(sel: List[Trajectory]) -> None:
    for i in range(len(sel) - 1):
        _assert(sel[i].end <= sel[i+1].start,
                f"overlap between picked {sel[i]} and {sel[i+1]}")
        _assert(sel[i].start < sel[i+1].start,
                f"non-increasing starts at i={i}")


def test_user_example():
    """User's example: trajectories (1,16), (4,20), ..., (15,31), (20,36), ...
    Expected pick: (1,16) and (20,36) and so on."""
    # Build a dense set: starts every 1 position, K=16 each (half-open: [s, s+16))
    trajs = [_t(s, s + 16, label=s) for s in range(1, 50)]
    sel = greedy_non_overlapping(trajs)
    _check_disjoint_sorted(sel)
    _assert(sel[0].start == 1 and sel[0].end == 17, f"first should be (1,17), got {sel[0]}")
    # User said (20, 36) is second. With K=16 half-open, (20, 36) is [20, 36) which has
    # length 16. After (1, 17), next non-overlap needs start >= 17. We pick the
    # smallest start >= 17, which is 17 in our dense set. So (17, 33), not (20, 36).
    _assert(sel[1].start == 17 and sel[1].end == 33, f"second should be (17,33), got {sel[1]}")
    print(f"  user_example: picked {len(sel)} trajs starting at {[t.start for t in sel[:5]]}...")
    # Verify the gap pattern when traj starts have stride > 1
    trajs2 = [_t(s, s + 16) for s in [1, 4, 8, 15, 20, 25, 33, 41]]
    sel2 = greedy_non_overlapping(trajs2)
    _check_disjoint_sorted(sel2)
    starts2 = [t.start for t in sel2]
    # Pick (1,17) — skip (4,20), (8,24), (15,31) — pick (20,36) — skip (25,41), (33,49) — pick (41,57)
    # Next non-overlap after end=36 needs start >= 36; (33,49) starts at 33 < 36 -> skipped.
    _assert(starts2 == [1, 20, 41], f"sparse example: expected starts [1, 20, 41], got {starts2}")
    print(f"  sparse_example: starts = {starts2}  ✓")


def test_empty():
    sel = greedy_non_overlapping([])
    _assert(sel == [], f"empty input should give empty list, got {sel}")
    print("  empty: ✓")


def test_single():
    sel = greedy_non_overlapping([_t(5, 21)])
    _assert(len(sel) == 1 and sel[0].start == 5, f"single: got {sel}")
    print("  single: ✓")


def test_all_overlap():
    """All trajectories overlap with the first; only first should be picked."""
    trajs = [_t(0, 16), _t(1, 17), _t(5, 21), _t(15, 31)]
    sel = greedy_non_overlapping(trajs)
    _check_disjoint_sorted(sel)
    _assert(len(sel) == 1 and sel[0].start == 0, f"all_overlap: got {sel}")
    print(f"  all_overlap: picked {len(sel)}  ✓")


def test_no_overlap():
    """Already-disjoint trajectories should all be picked."""
    trajs = [_t(0, 16), _t(16, 32), _t(32, 48), _t(48, 64)]
    sel = greedy_non_overlapping(trajs)
    _check_disjoint_sorted(sel)
    _assert(len(sel) == 4, f"no_overlap: expected all 4 picked, got {len(sel)}")
    starts = [t.start for t in sel]
    _assert(starts == [0, 16, 32, 48], f"no_overlap starts: {starts}")
    print(f"  no_overlap: picked all 4  ✓")


def test_boundary_touch():
    """A.end == B.start means they touch but don't overlap; both should be picked."""
    trajs = [_t(0, 16), _t(16, 32)]
    sel = greedy_non_overlapping(trajs)
    _assert(len(sel) == 2, f"boundary_touch: expected 2 picked, got {len(sel)}")
    print(f"  boundary_touch: both picked  ✓")


def test_unsorted_input():
    """Algorithm must sort internally; order of input shouldn't matter."""
    trajs = [_t(20, 36), _t(1, 17), _t(40, 56), _t(15, 31)]
    sel = greedy_non_overlapping(trajs)
    _check_disjoint_sorted(sel)
    starts = [t.start for t in sel]
    # Sort by start: 1, 15, 20, 40 -> pick 1, skip 15 (overlap), pick 20, pick 40
    _assert(starts == [1, 20, 40], f"unsorted: expected [1, 20, 40], got {starts}")
    print(f"  unsorted: starts = {starts}  ✓")


def test_response_len_filter():
    """Trajectories that extend past response_len should be dropped."""
    trajs = [_t(0, 16), _t(16, 32), _t(32, 48)]
    sel = greedy_non_overlapping(trajs, response_len=40)
    starts = [t.start for t in sel]
    # (32, 48) extends past 40 -> drop. Pick (0, 16), (16, 32).
    _assert(starts == [0, 16], f"response_len: expected [0, 16], got {starts}")
    print(f"  response_len=40: dropped (32,48), kept {starts}  ✓")


def test_variable_lengths():
    """Trajectories with different lengths still picked greedily by start."""
    trajs = [_t(0, 10), _t(5, 25), _t(10, 16), _t(20, 36)]
    sel = greedy_non_overlapping(trajs)
    _check_disjoint_sorted(sel)
    starts = [t.start for t in sel]
    # Sort: (0,10), (5,25), (10,16), (20,36). Pick (0,10), skip (5,25), pick (10,16), pick (20,36).
    _assert(starts == [0, 10, 20], f"var_lengths: expected [0,10,20], got {starts}")
    print(f"  variable_lengths: starts = {starts}  ✓")


def test_invariant_drafts_preserved():
    """The picked trajectories must retain their original draft tokens."""
    t1 = Trajectory(0, 4, (10, 20, 30, 40), iter_id=1)
    t2 = Trajectory(4, 8, (50, 60, 70, 80), iter_id=2)
    t3 = Trajectory(2, 6, (99, 99, 99, 99), iter_id=3)   # overlaps t1
    sel = greedy_non_overlapping([t1, t2, t3])
    _assert(len(sel) == 2, f"picked count: {len(sel)}")
    _assert(sel[0].draft == (10, 20, 30, 40), f"t1 draft preserved: {sel[0].draft}")
    _assert(sel[1].draft == (50, 60, 70, 80), f"t2 draft preserved: {sel[1].draft}")
    _assert(sel[0].iter_id == 1 and sel[1].iter_id == 2, "iter_ids preserved")
    print(f"  drafts_preserved: ✓")


def test_realistic_jacobi_trajectory_set():
    """Simulate a real rollout. K=16, response length 256, n_acc varies 0-12.
    Each trajectory starts at sum(n_acc[:i]+1) for the prior i forwards."""
    K = 16
    response_len = 256
    pos = 0
    trajs = []
    iter_id = 0
    # Simulate n_acc varying per iter (typical for k3: avg ~4 accepted per iter at K=16)
    import random
    random.seed(0)
    while pos + K <= response_len:
        n_acc = random.randint(0, K - 1)        # 0 to K-1 accepted
        trajs.append(_t(pos, pos + K, label=iter_id))
        pos += n_acc + 1                         # advance by accepted + bonus
        iter_id += 1
    sel = greedy_non_overlapping(trajs, response_len=response_len)
    _check_disjoint_sorted(sel)
    coverage = sum(t.length for t in sel)
    print(f"  realistic_jacobi: {len(trajs)} trajs -> picked {len(sel)}, "
          f"coverage = {coverage}/{response_len} ({100*coverage/response_len:.0f}%)")
    starts = [t.start for t in sel]
    print(f"    picked starts: {starts}")


if __name__ == "__main__":
    print("Running onpolicy_select tests:")
    test_empty()
    test_single()
    test_all_overlap()
    test_no_overlap()
    test_boundary_touch()
    test_unsorted_input()
    test_response_len_filter()
    test_variable_lengths()
    test_invariant_drafts_preserved()
    test_user_example()
    test_realistic_jacobi_trajectory_set()
    print("\nAll tests passed.")
