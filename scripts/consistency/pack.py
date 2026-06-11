"""Build the Decode-Learning interleaved `[prompt, k_0, last_0, ...]` batch
from verl's standard (prompt, response) rollouts.

  k_j   (noisy block):  initialized from existing sequence tokens (matches
                         JF's `random.choice(generated_ids)` init)
  last_j (clean block): copy of the j-th block of the rollout response

Both k_j and last_j get the SAME position_ids — they represent two views
of the same Jacobi step (essential for RoPE consistency at the model
level).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _build_alt_pools(traj_list, r_start: int, N: int) -> list[list[int]]:
    """For each of the N positions in window [r_start, r_start+N), collect
    alternative tokens from every cascade iter whose own window covered that
    position. Each alternative is one iter's argmax prediction at that
    response position — a "model-believed-plausible" next-token at that step.

    Returns: list of length N; pool[i] = list of distinct alternative tokens
    seen across iters at response position (r_start + i). Empty list if no
    iter covered that position (rare).

    `traj_list` is the per-sample list of Trajectory objects (NOT only the
    greedy-non-overlapping subset — we need the full set for diversity).
    Each Trajectory has start, end, and—if available—draft (the model's INPUT
    drafts) plus target_argmax (the model's PREDICTIONS at those positions).
    The cascade target_argmax at iter t is what we want here; if a Trajectory
    only has `draft`, we fall back to using draft as a proxy.
    """
    pools: list[list[int]] = [[] for _ in range(N)]
    seen_per_pos: list[set] = [set() for _ in range(N)]
    if not traj_list:
        return pools
    for traj in traj_list:
        # Accept Trajectory (dataclass) or legacy tuple (start, end, draft, pml).
        if hasattr(traj, "start"):
            s = int(traj.start); e = int(traj.end)
            toks = getattr(traj, "target_argmax", None) or getattr(traj, "draft", None)
        else:
            s = int(traj[0]); e = int(traj[1])
            toks = traj[2] if len(traj) > 2 else None
        if toks is None:
            continue
        # Map each iter position p in [s, e) to our window offset (p - r_start).
        for j_off, t in enumerate(toks):
            p = s + j_off
            if p < r_start or p >= r_start + N:
                continue
            i = p - r_start
            t_int = int(t)
            if t_int in seen_per_pos[i]:
                continue
            seen_per_pos[i].add(t_int)
            pools[i].append(t_int)
    return pools


@dataclass
class InterleavedBatch:
    input_ids: torch.Tensor          # (B, Lmax) int64
    position_ids: torch.Tensor       # (B, Lmax) int64 — shared positions per pair/triple
    prompt_lens: torch.Tensor        # (B,) int64
    num_pairs: torch.Tensor          # (B,) int64 — T per sample
    block_lens: torch.Tensor         # (B,) int64 — N per sample (currently uniform)
    seq_lens: torch.Tensor           # (B,) int64
    pad_mask: torch.Tensor           # (B, Lmax) bool — True where valid
    noisy_mask: torch.Tensor         # (B, Lmax) bool — True at ANY noisy position
    # Multi-noise tiling (2-block layout only): K independent noisy tiles per
    # clean block, all sharing the same RoPE positions as the clean block.
    # K=1 (default) is the historical [noisy | clean] layout. K>1 enables
    # diffusion-style Monte-Carlo averaging over noise samples.
    num_noisy_tiles: int = 1
    # triple_mode only:
    triple_mode: bool = False        # True if batch was built with the 3-block layout
    marked_mask: torch.Tensor | None = None    # (B, Lmax) bool — True at marker-side noisy positions only
    unmarked_mask: torch.Tensor | None = None  # (B, Lmax) bool — True at no-marker-side noisy positions only
    # On-policy only: per-sample, per-pair prefix-match length. dflash_ce loss
    # skips positions [0, prefix_match_len[b, j]) within pair j (those are
    # cascade-converged = match response, no noise to denoise) and shifts the
    # decay weight to start at position `prefix_match_len[b, j]`.
    onpolicy_prefix_lens: torch.Tensor | None = None  # (B, T_max) int64
    # Per-pair noisy/clean start positions in the packed sequence. Needed when
    # clean block sizes vary across pairs (on-policy bridge fix — see below).
    # When None, falls back to uniform-stride layout (off-policy / triple_mode).
    #   noisy_starts[b, j] = first packed position of pair j's noisy block
    #   clean_starts[b, j] = first packed position of pair j's (extended) clean
    noisy_starts: torch.Tensor | None = None  # (B, T_max) int64
    clean_starts: torch.Tensor | None = None  # (B, T_max) int64
    # Per-position pair index and role tensors. Used by the SDPA mask and
    # position-identification helpers when block layout is non-uniform
    # (on-policy bridge mode). For uniform layouts these stay None and the
    # legacy modular-arithmetic path runs.
    #   role_per_pos[b, l]: 0=prompt, 1=noisy, 2=clean (incl. bridge ext), 3=pad
    #   pair_idx_per_pos[b, l]: pair index (0..T-1) for noisy/clean, -1 else
    role_per_pos: torch.Tensor | None = None       # (B, Lmax) int8
    pair_idx_per_pos: torch.Tensor | None = None    # (B, Lmax) int64
    # v11 canvas mode (CONSISTENCY_CANVAS_FRAC > 0): per-pair mode flag.
    # Canvas pairs are dLLM-style training pairs: noisy tile = clean window
    # with round(f*N) positions renoised to fresh uniform-random tokens
    # (f ~ Uniform{CONSISTENCY_CANVAS_LEVELS}, far-weighted position choice
    # w_j ∝ 0.5 + j/N), fully-bidirectional intra-tile attention, marker ON,
    # uniform-weight loss. Causal (non-canvas) pairs keep v9 behavior exactly.
    # Both stay None when canvas mode is off (bit-identical legacy path).
    canvas_pairs: torch.Tensor | None = None       # (B, T_max) bool
    canvas_mask: torch.Tensor | None = None        # (B, Lmax) bool — noisy positions of canvas pairs

    def to(self, device):
        return InterleavedBatch(
            input_ids=self.input_ids.to(device),
            position_ids=self.position_ids.to(device),
            prompt_lens=self.prompt_lens.to(device),
            num_pairs=self.num_pairs.to(device),
            block_lens=self.block_lens.to(device),
            seq_lens=self.seq_lens.to(device),
            pad_mask=self.pad_mask.to(device),
            noisy_mask=self.noisy_mask.to(device),
            num_noisy_tiles=self.num_noisy_tiles,
            triple_mode=self.triple_mode,
            marked_mask=self.marked_mask.to(device) if self.marked_mask is not None else None,
            unmarked_mask=self.unmarked_mask.to(device) if self.unmarked_mask is not None else None,
            onpolicy_prefix_lens=self.onpolicy_prefix_lens.to(device) if self.onpolicy_prefix_lens is not None else None,
            noisy_starts=self.noisy_starts.to(device) if self.noisy_starts is not None else None,
            clean_starts=self.clean_starts.to(device) if self.clean_starts is not None else None,
            role_per_pos=self.role_per_pos.to(device) if self.role_per_pos is not None else None,
            pair_idx_per_pos=self.pair_idx_per_pos.to(device) if self.pair_idx_per_pos is not None else None,
            canvas_pairs=self.canvas_pairs.to(device) if self.canvas_pairs is not None else None,
            canvas_mask=self.canvas_mask.to(device) if self.canvas_mask is not None else None,
        )


def build_interleaved_batch(
    prompt_ids: list[torch.Tensor],
    response_ids: list[torch.Tensor],
    block_size: int,
    pad_id: int,
    max_pairs: int | None = None,
    generator: torch.Generator | None = None,
    triple_mode: bool = False,
    num_noisy_tiles: int = 1,
    cascade_drafts: "list[list] | None" = None,
        # Per-sample list of cascade drafts. Each entry can be either:
        #   - a (start, end, draft_tokens) tuple — legacy (prefix_match_len=0)
        #   - a Trajectory object — has prefix_match_len populated
        # We accept both for incremental migration.
) -> InterleavedBatch:
    """Build the interleaved batch from a list of (prompt, response) tensors.

    For each sample with response of length R:
      T = ceil(R / block_size) clean blocks, each of length N=block_size
        (last block right-padded with pad_id to N if R % N != 0)
      For each clean block last_j (positions [P + (2j+1)N, P + (2j+2)N) in the
      packed sequence), construct a paired noisy block k_j (positions
      [P + 2jN, P + (2j+1)N)) by sampling `block_size` tokens uniformly
      with replacement from (prompt + response). This matches the JF
      inference prefill init (random.choice from generated_ids).

    The total packed length per sample is P + 2*T*N. Across the batch,
    we pad to the max packed length with pad_id.

    cascade_drafts (on-policy mode): when provided, overrides the uniform
      j*N stride with per-sample selected trajectories. cascade_drafts[b] is
      a list of (start, end, draft_tokens) tuples for sample b — typically
      the output of `greedy_non_overlapping(load_trajectories(...))`. Each
      trajectory must have length == block_size. The clean block at pair j
      covers response[start_j : end_j], and the noisy block uses draft_tokens
      from the trajectory (cascade-evolved input the model actually saw at
      inference time). Position IDs are set to the response positions so
      RoPE matches what the model would compute under AR generation. Samples
      whose cascade_drafts[b] is None or empty fall back to uniform-stride
      regular layout for that sample.

      Incompatible with triple_mode and num_noisy_tiles > 1 (would require
      multiple noise draws per block, but on-policy provides only one).
    """
    assert len(prompt_ids) == len(response_ids)
    B = len(prompt_ids)
    N = int(block_size)
    K = int(num_noisy_tiles)
    assert K >= 1, f"num_noisy_tiles must be >= 1, got {K}"
    if triple_mode and K != 1:
        raise ValueError(
            f"triple_mode is incompatible with num_noisy_tiles > 1 (got K={K}); "
            f"multi-noise tiling is only implemented for the 2-block layout."
        )
    # On-policy mode: with K=1 only tile 0 (= the cascade draft) is used.
    # With K>1, tile 0 is the cascade draft and tiles 1..K-1 are resampled from
    # the empirical noise bank (CONSISTENCY_NOISE_BANK_PATH) with per-tile
    # RNG seeds for diversity. Triple-mode is still incompatible.
    if cascade_drafts is not None:
        if triple_mode:
            raise ValueError(
                "cascade_drafts is incompatible with triple_mode"
            )
        if len(cascade_drafts) != B:
            raise ValueError(
                f"cascade_drafts length ({len(cascade_drafts)}) must equal batch size ({B})"
            )

    # Per-sample packed lengths.
    #   2-block (K=1):       [prompt | noisy(N) | clean(N) | noisy(N) | clean(N) | ... ]
    #   2-block multi-noise: [prompt | noisy1(N) ... noisyK(N) | clean(N) | noisy1(N) ... | ...]
    #   3-block:             [prompt | marked(N) | unmarked(N) | clean(N) | ... ]
    blocks_per_group = 3 if triple_mode else (K + 1)
    P = torch.tensor([int(p.numel()) for p in prompt_ids], dtype=torch.long)
    Rn = torch.tensor([int(r.numel()) for r in response_ids], dtype=torch.long)
    T_full = ((Rn + N - 1) // N).long()

    # On-policy override: per-sample T is the number of selected trajectories.
    # We still allocate Lmax based on the WORST CASE (uniform-stride T_full)
    # so triple_mode / multi-noise can later flip back to AR mode if desired.
    # For now in on-policy mode, T_b is just len(cascade_drafts[b]) — but we
    # validate trajectory length == N and start/end fit in the response.
    def _unpack_traj(item):
        """Accept both (start, end, draft) tuples and Trajectory objects."""
        if hasattr(item, "start") and hasattr(item, "end") and hasattr(item, "draft"):
            return int(item.start), int(item.end), tuple(item.draft), int(getattr(item, "prefix_match_len", 0))
        s, e, d = item[0], item[1], item[2]
        return int(s), int(e), tuple(d), 0

    # On-policy variable-clean layout: when consecutive selected trajectories
    # leave a gap in the response (s_{j+1} > e_j), the bridge tokens
    # response[e_j : s_{j+1}) are appended to pair j's clean block. This makes
    # pair j+1's noisy attend to the actual response prefix at inference-time
    # context positions, fixing a training-vs-inference attention mismatch.
    # Per-pair clean_size = (s_{j+1} - s_j) for j < T-1, else N.
    onpol_clean_sizes: list[list[int]] | None = None
    if cascade_drafts is not None:
        onpol_T = torch.zeros(B, dtype=torch.long)
        onpol_clean_sizes = []
        for b in range(B):
            drafts_b = cascade_drafts[b] or []
            # Apply max_pairs cap BEFORE clean-size computation so we don't try
            # to look at trajectories beyond what we'll actually pack.
            # CONSISTENCY_PAIR_SAMPLE=random takes a uniform subset over the
            # WHOLE response (order preserved — bridge mode needs sorted starts)
            # instead of the first N windows, which biases cons coverage to the
            # first ~N*K response tokens and leaves late positions untrained.
            if max_pairs is not None and len(drafts_b) > int(max_pairs):
                import os as _os
                import random as _random
                if _os.environ.get("CONSISTENCY_PAIR_SAMPLE", "first").lower() == "random":
                    sel = sorted(_random.sample(range(len(drafts_b)), int(max_pairs)))
                    drafts_b = [drafts_b[i] for i in sel]
                else:
                    drafts_b = drafts_b[: int(max_pairs)]
            for item in drafts_b:
                s, e, d, _ = _unpack_traj(item)
                if e - s != N:
                    raise ValueError(
                        f"sample {b}: trajectory length {e - s} != block_size {N}; "
                        f"set JACOBI_K == CONSISTENCY_BLOCK_SIZE for on-policy mode"
                    )
                if e > int(Rn[b].item()):
                    raise ValueError(
                        f"sample {b}: trajectory end {e} > response_len {int(Rn[b].item())}"
                    )
                if len(d) != N:
                    raise ValueError(
                        f"sample {b}: draft len {len(d)} != N={N}"
                    )
            onpol_T[b] = len(drafts_b)
            # Per-pair clean size = distance to next pair (extended clean covers
            # original N response tokens + bridge to next pair).
            cs = []
            for j in range(len(drafts_b)):
                s_j, e_j, _, _ = _unpack_traj(drafts_b[j])
                if j < len(drafts_b) - 1:
                    s_next, _, _, _ = _unpack_traj(drafts_b[j + 1])
                    gap = s_next - e_j
                    assert gap >= 0, f"sample {b} pair {j}: negative gap {gap} — greedy selector bug"
                    cs.append(N + gap)
                else:
                    cs.append(N)
            onpol_clean_sizes.append(cs)
        # Update cascade_drafts in place if we capped any sample. We keep the
        # full per-sample list semantically — the iteration below uses T_b.
        T_full = onpol_T
    if max_pairs is not None and cascade_drafts is None:
        T_full = torch.minimum(T_full, torch.tensor(int(max_pairs), dtype=torch.long))
    # Per-sample packed length. For on-policy bridge mode, sum per-pair extents
    # because clean block sizes vary. Off-policy keeps the uniform formula.
    if cascade_drafts is not None:
        per_sample_pack_lens = []
        for b in range(B):
            extent = int(P[b].item()) + sum(K * N + cs for cs in onpol_clean_sizes[b])
            per_sample_pack_lens.append(extent)
        seq_lens = torch.tensor(per_sample_pack_lens, dtype=torch.long)
    else:
        seq_lens = P + blocks_per_group * T_full * N
    Lmax_raw = int(seq_lens.max().item()) if B > 0 else 0
    # flex_attention backward Triton kernel is unreliable when Lmax isn't
    # a multiple of its internal block tile (typically 128). Pad up.
    BLOCK_TILE = 128
    Lmax = ((Lmax_raw + BLOCK_TILE - 1) // BLOCK_TILE) * BLOCK_TILE
    Lmax = max(Lmax, BLOCK_TILE)

    input_ids = torch.full((B, Lmax), int(pad_id), dtype=torch.long)
    position_ids = torch.zeros((B, Lmax), dtype=torch.long)
    pad_mask = torch.zeros((B, Lmax), dtype=torch.bool)
    noisy_mask = torch.zeros((B, Lmax), dtype=torch.bool)
    marked_mask = torch.zeros((B, Lmax), dtype=torch.bool) if triple_mode else None
    unmarked_mask = torch.zeros((B, Lmax), dtype=torch.bool) if triple_mode else None
    # Per-pair prefix-match length, populated only in on-policy mode.
    T_max = int(T_full.max().item()) if B > 0 else 0
    onpolicy_prefix_lens = (
        torch.zeros((B, max(T_max, 1)), dtype=torch.long)
        if cascade_drafts is not None and T_max > 0
        else None
    )
    # On-policy variable-layout helpers. noisy_starts[b, j] / clean_starts[b, j]
    # store the packed-position of the noisy / clean block start for pair j.
    # role_per_pos[b, l] and pair_idx_per_pos[b, l] are per-position lookups
    # for the SDPA mask (so it doesn't need to invert variable strides).
    if cascade_drafts is not None and T_max > 0:
        noisy_starts = torch.zeros((B, T_max), dtype=torch.long)
        clean_starts = torch.zeros((B, T_max), dtype=torch.long)
        role_per_pos = torch.zeros((B, Lmax), dtype=torch.int8)
        pair_idx_per_pos = torch.full((B, Lmax), -1, dtype=torch.long)
    else:
        noisy_starts = None
        clean_starts = None
        role_per_pos = None
        pair_idx_per_pos = None

    import os as _os
    # Default is "uniform" — "sample" mode (draw from prompt+response) leaks the
    # training-prompt distribution into the denoiser. Confirmed failure mode in
    # FINDINGS.md (HE+ pass@8 collapsed 0.762 → 0.601 by step 120). Keep "sample"
    # available behind the env var for ablation but never as the silent default.
    _noise_source = _os.environ.get("CONSISTENCY_NOISE_SOURCE", "uniform").lower()
    _vocab_size = int(_os.environ.get("CONSISTENCY_VOCAB_SIZE", "152064"))
    # dFlash-style mask-token init. When CONSISTENCY_NOISE_SOURCE=mask, every
    # noisy-block position gets the SAME single token id — a consistent "predict
    # here" signal the model can specialize on (cf. dFlash arXiv:2602.06036).
    # Default mask id = Qwen2.5 pad_id (151643), which is reserved and never
    # appears in clean responses.
    _mask_token_id = int(_os.environ.get("CONSISTENCY_MASK_TOKEN_ID", "151643"))

    # Multi-noise tiling is incompatible with mask noise (all K tiles would be
    # identical — defeats the Monte-Carlo averaging purpose). Reject early.
    if K > 1 and _noise_source == "mask":
        raise ValueError(
            f"CONSISTENCY_NUM_NOISY_TILES={K} requires a stochastic noise source; "
            f"got CONSISTENCY_NOISE_SOURCE='mask' which is deterministic."
        )

    # v11 canvas mode. Per-pair coin: with prob CONSISTENCY_CANVAS_FRAC a pair
    # is built in canvas mode (dLLM corruption→clean: bidir + marker + uniform
    # loss); otherwise it keeps the v9 causal construction. The canvas noisy
    # tile ALWAYS uses fresh uniform-random renoise (spec §2.4/§3.1: decode
    # re-noises with fresh randoms, so training must too) regardless of
    # CONSISTENCY_NOISE_SOURCE.
    _canvas_frac = float(_os.environ.get("CONSISTENCY_CANVAS_FRAC", "0.0"))
    _canvas_levels = [
        float(x) for x in _os.environ.get(
            "CONSISTENCY_CANVAS_LEVELS", "1.0,0.75,0.5,0.25,0.125"
        ).split(",") if x.strip()
    ]
    _canvas_plausible = float(_os.environ.get("CONSISTENCY_CANVAS_PLAUSIBLE_FRAC", "0.0"))
    # v11.1: canvas tile construction mode.
    #   "levels"    (default) — v11.0 f-mixture corruption of the clean window.
    #   "empirical" — match the MEASURED hybrid-decode state distribution
    #     (trace audit 2026-06-12): under the keep-everything update the canvas
    #     holds the model's own previous predictions — correct with a
    #     position-decaying probability p(j) (P_NEAR at the boundary → P_FAR at
    #     depth), wrong-but-plausible otherwise — plus a fresh-noise tail of
    #     n_new ~ U{0..MAX_COMMIT} newly-entered positions. Wrong tokens are
    #     drawn from the policy's own cascade prediction pools (the on-policy
    #     side-channel already recorded at rollout; fresh noise fallback).
    #     A LEVELS_FRAC fraction of canvas pairs keeps the f-mixture for
    #     coverage of the all-noise regime (fresh window starts).
    _canvas_construction = _os.environ.get("CONSISTENCY_CANVAS_CONSTRUCTION", "levels").lower()
    _canvas_levels_frac = float(_os.environ.get("CONSISTENCY_CANVAS_LEVELS_FRAC", "0.25"))
    _canvas_p_near = float(_os.environ.get("CONSISTENCY_CANVAS_P_NEAR", "0.6"))
    _canvas_p_far = float(_os.environ.get("CONSISTENCY_CANVAS_P_FAR", "0.1"))
    _canvas_max_commit = int(_os.environ.get("CONSISTENCY_CANVAS_MAX_COMMIT", "8"))
    if _canvas_construction not in ("levels", "empirical"):
        raise ValueError(
            f"CONSISTENCY_CANVAS_CONSTRUCTION must be 'levels' or 'empirical', "
            f"got {_canvas_construction!r}"
        )
    if _canvas_frac > 0.0:
        if triple_mode:
            raise ValueError("CONSISTENCY_CANVAS_FRAC > 0 is incompatible with triple_mode")
        if K != 1:
            raise ValueError(
                f"CONSISTENCY_CANVAS_FRAC > 0 requires CONSISTENCY_NUM_NOISY_TILES=1, got K={K}"
            )
        if not _canvas_levels:
            raise ValueError("CONSISTENCY_CANVAS_LEVELS parsed to an empty list")
    T_alloc = int(T_full.max().item()) if B > 0 else 0
    canvas_pairs = (
        torch.zeros((B, max(T_alloc, 1)), dtype=torch.bool) if _canvas_frac > 0.0 else None
    )
    canvas_mask = (
        torch.zeros((B, Lmax), dtype=torch.bool) if _canvas_frac > 0.0 else None
    )

    def _canvas_uniform(n: int, g):
        if g is not None:
            return torch.randint(low=0, high=_vocab_size, size=(n,), generator=g, dtype=torch.long)
        return torch.randint(low=0, high=_vocab_size, size=(n,), dtype=torch.long)

    def _build_canvas_tile(clean_win: torch.Tensor, valid_n: int, g):
        """Spec §3.2 steps 1-4: input tile = clean window with |R| = round(f*N)
        positions renoised to fresh uniform-random tokens. R is far-weighted
        (P(j ∈ R) ∝ 0.5 + j/N — far positions noisier, approximating the
        decode-time spatial age profile). Pad-tail positions (≥ valid_n, last
        block off-policy only) are filled with noise; the loss masks them out
        via pad_mask. Returns (tile, renoise_index_tensor)."""
        tile = torch.empty(N, dtype=torch.long)
        tile[:valid_n] = clean_win[:valid_n].long()
        if valid_n < N:
            tile[valid_n:] = _canvas_uniform(N - valid_n, g)
        if g is not None:
            f_i = int(torch.randint(low=0, high=len(_canvas_levels), size=(1,), generator=g).item())
        else:
            f_i = int(torch.randint(low=0, high=len(_canvas_levels), size=(1,)).item())
        f = _canvas_levels[f_i]
        n_renoise = min(N, int(round(f * N)))
        if n_renoise <= 0:
            return tile, torch.empty(0, dtype=torch.long)
        w = 0.5 + torch.arange(N, dtype=torch.float32) / float(N)
        if g is not None:
            R = torch.multinomial(w, n_renoise, replacement=False, generator=g)
        else:
            R = torch.multinomial(w, n_renoise, replacement=False)
        tile[R] = _canvas_uniform(n_renoise, g)
        return tile, R

    def _build_canvas_tile_empirical(clean_win: torch.Tensor, valid_n: int,
                                     pools: "list[list[int]] | None", g):
        """v11.1 empirical-state tile (see _canvas_construction docstring).
        Body position j: clean token w.p. p(j) = P_NEAR·(P_FAR/P_NEAR)^(j/(N-1)),
        else a policy-predicted alternative from pools[j] (uniform-noise
        fallback when no pool). Tail of n_new ~ U{0..MAX_COMMIT} positions =
        fresh uniform noise (newly entered at the window's far end)."""
        tile = torch.empty(N, dtype=torch.long)
        if g is not None:
            n_new = int(torch.randint(0, _canvas_max_commit + 1, (1,), generator=g).item())
            u_vec = torch.rand(N, generator=g)
            pick_vec = torch.rand(N, generator=g)
        else:
            n_new = int(torch.randint(0, _canvas_max_commit + 1, (1,)).item())
            u_vec = torch.rand(N)
            pick_vec = torch.rand(N)
        body_end = max(0, N - n_new)
        ratio = _canvas_p_far / max(_canvas_p_near, 1e-9)
        for j in range(N):
            if j >= body_end or j >= valid_n:
                tile[j] = _canvas_uniform(1, g)[0]
                continue
            p_j = _canvas_p_near * (ratio ** (j / max(1, N - 1)))
            if float(u_vec[j]) < p_j:
                tile[j] = clean_win[j]
            elif pools is not None and pools[j]:
                tile[j] = int(pools[j][int(float(pick_vec[j]) * len(pools[j])) % len(pools[j])])
            else:
                tile[j] = _canvas_uniform(1, g)[0]
        return tile

    for b in range(B):
        Pb = int(P[b].item())
        Tb = int(T_full[b].item())
        # prompt section
        input_ids[b, :Pb] = prompt_ids[b]
        position_ids[b, :Pb] = torch.arange(Pb, dtype=torch.long)
        pad_mask[b, :Pb] = True

        pool = torch.cat([prompt_ids[b].long(), response_ids[b].long()])

        def _draw_noisy(g):
            if _noise_source == "uniform":
                return torch.randint(low=0, high=_vocab_size, size=(N,),
                                      generator=g, dtype=torch.long) if g is not None \
                    else torch.randint(low=0, high=_vocab_size, size=(N,), dtype=torch.long)
            elif _noise_source == "mask":
                # dFlash-style: every position gets the SAME mask token id.
                return torch.full((N,), _mask_token_id, dtype=torch.long)
            else:
                idx = torch.randint(low=0, high=int(pool.numel()), size=(N,), generator=g) if g is not None \
                    else torch.randint(low=0, high=int(pool.numel()), size=(N,))
                return pool[idx]

        if not triple_mode:
            # 2-block layout (K=1, historical):
            #   [prompt | noisy | clean | noisy | clean | ...]
            # Multi-noise variant (K>=1):
            #   [prompt | noisy_1 | noisy_2 | ... | noisy_K | clean | noisy_1 | ... ]
            # All K+1 tiles in each group share the same RoPE positions as
            # the clean block they're paired with. Each noisy tile is an
            # INDEPENDENT noise draw (separate _draw_noisy call). Attention
            # masking (in attention.py) prevents sibling noisy tiles from
            # attending to each other.
            #
            # On-policy override: when cascade_drafts is provided, the j-th
            # pair uses the j-th trajectory's (start, end, draft) instead of
            # the regular j*N stride. RoPE position_ids point at the actual
            # response positions [Pb + start, Pb + end).
            #
            # On-policy bridge extension: when consecutive trajectories leave
            # a gap (s_{j+1} > e_j), the gap response tokens are APPENDED to
            # pair j's clean block as a bridge. The clean block's size becomes
            # N + gap (for j < T-1; for j == T-1, just N). This makes pair
            # j+1's noisy attention see the contiguous response prefix instead
            # of jumping over the gap, fixing a training-inference mismatch.
            cursor = Pb  # next packed offset (variable layout for on-policy)
            for j in range(Tb):
                # v11 canvas coin. Drawn ONLY when canvas mode is on, so the
                # legacy RNG stream (and therefore the packed batch) stays
                # bit-identical when CONSISTENCY_CANVAS_FRAC is 0/unset.
                is_canvas = False
                if _canvas_frac > 0.0:
                    if generator is not None:
                        coin = torch.rand((), generator=generator)
                    else:
                        coin = torch.rand(())
                    is_canvas = bool(coin.item() < _canvas_frac)
                if cascade_drafts is not None:
                    # Variable layout: each pair occupies N (noisy) + clean_size_j.
                    clean_size_j = int(onpol_clean_sizes[b][j])
                    group_start = cursor
                    ls = group_start + K * N  # K=1 in on-policy, kept generic
                    cursor = group_start + K * N + clean_size_j  # advance
                else:
                    # Off-policy uniform stride.
                    group_start = Pb + (K + 1) * j * N
                    ls = group_start + K * N
                    clean_size_j = N
                if cascade_drafts is not None:
                    s_traj, e_traj, draft_toks, pml = _unpack_traj(cascade_drafts[b][j])
                    r_start = s_traj
                    r_end = e_traj
                    onpol_noisy = torch.tensor(draft_toks, dtype=torch.long)
                    if onpolicy_prefix_lens is not None:
                        onpolicy_prefix_lens[b, j] = pml
                    # Optional Bernoulli draft corruption: replace each draft
                    # token at position >= pml with a uniform-random vocab token
                    # with probability CONSISTENCY_DRAFT_CORRUPT_PROB. Pushes
                    # the unmatched-draft regime back toward "noise" so the
                    # model trains a stronger ignore-the-noise invariance —
                    # the original CLLM / dFlash setup that achieved high TPF
                    # via uniform-vocab noise. The matched prefix [0, pml)
                    # stays untouched so the trust-the-prefix signal is preserved.
                    _corrupt_p = float(_os.environ.get("CONSISTENCY_DRAFT_CORRUPT_PROB", "0.0"))
                    if _corrupt_p > 0.0 and pml < N and not is_canvas:
                        sub_n = N - pml
                        if generator is not None:
                            keep_or_corrupt = torch.bernoulli(
                                torch.full((sub_n,), float(_corrupt_p)),
                                generator=generator,
                            ).bool()
                            random_toks = torch.randint(
                                low=0, high=_vocab_size, size=(sub_n,),
                                generator=generator, dtype=torch.long,
                            )
                        else:
                            keep_or_corrupt = torch.bernoulli(
                                torch.full((sub_n,), float(_corrupt_p)),
                            ).bool()
                            random_toks = torch.randint(
                                low=0, high=_vocab_size, size=(sub_n,),
                                dtype=torch.long,
                            )
                        if keep_or_corrupt.any():
                            onpol_noisy = onpol_noisy.clone()
                            onpol_noisy[pml:] = torch.where(
                                keep_or_corrupt, random_toks, onpol_noisy[pml:]
                            )
                else:
                    r_start = j * N
                    r_end = min((j + 1) * N, int(Rn[b].item()))
                    onpol_noisy = None
                valid_n = r_end - r_start
                if is_canvas:
                    # Canvas pair: REPLACE the noisy input (cascade draft or
                    # legacy uniform draw) with a corruption of the CLEAN
                    # window. There is no cascade-converged prefix in canvas
                    # mode — zero the pml so the loss treats every position
                    # uniformly.
                    clean_win = response_ids[b][r_start:r_end]
                    # Alt pools (the policy's own rollout-time predictions at
                    # these positions): used by the empirical construction for
                    # wrong-but-plausible tokens and by the ρ plausible-subs.
                    pools = None
                    if cascade_drafts is not None and (
                        _canvas_construction == "empirical" or _canvas_plausible > 0.0
                    ):
                        all_trajs_b = getattr(cascade_drafts, "all_trajs", None)
                        src_list = (
                            all_trajs_b[b]
                            if all_trajs_b is not None and b < len(all_trajs_b)
                            else cascade_drafts[b]
                        )
                        pools = _build_alt_pools(src_list, r_start, N)
                    # Per-pair construction choice: empirical mode keeps a
                    # LEVELS_FRAC fraction of pairs on the f-mixture for
                    # all-noise-regime coverage.
                    use_levels = _canvas_construction == "levels"
                    if not use_levels and _canvas_levels_frac > 0.0:
                        if generator is not None:
                            lc = torch.rand((), generator=generator)
                        else:
                            lc = torch.rand(())
                        use_levels = bool(lc.item() < _canvas_levels_frac)
                    if not use_levels:
                        canvas_tile = _build_canvas_tile_empirical(
                            clean_win, valid_n, pools, generator
                        )
                    else:
                        canvas_tile, renoise_R = _build_canvas_tile(clean_win, valid_n, generator)
                        # Plausible substitutions (ρ): a fraction of the KEPT
                        # positions get a model-believed alternative from the
                        # cascade alt pools instead of ground truth (decode-time
                        # kept tokens are model-chosen, occasionally confident-
                        # wrong). On-policy only; off-policy has no pools (spec
                        # v1 fallback: skip).
                        if _canvas_plausible > 0.0 and pools is not None:
                            in_R = torch.zeros(N, dtype=torch.bool)
                            if renoise_R.numel() > 0:
                                in_R[renoise_R] = True
                            for i_pos in range(min(valid_n, N)):
                                if bool(in_R[i_pos]) or not pools[i_pos]:
                                    continue
                                if generator is not None:
                                    u = torch.rand((), generator=generator)
                                    pick = torch.randint(0, len(pools[i_pos]), (1,), generator=generator)
                                else:
                                    u = torch.rand(())
                                    pick = torch.randint(0, len(pools[i_pos]), (1,))
                                if float(u.item()) < _canvas_plausible:
                                    canvas_tile[i_pos] = int(pools[i_pos][int(pick.item())])
                    onpol_noisy = canvas_tile
                    if onpolicy_prefix_lens is not None:
                        onpolicy_prefix_lens[b, j] = 0
                    canvas_pairs[b, j] = True
                    canvas_mask[b, group_start : group_start + N] = True
                # Position IDs span the actual response positions for this pair.
                # (In on-policy mode, r_start can be arbitrary, not just j*N.)
                shared_pos = torch.arange(Pb + r_start, Pb + r_start + N, dtype=torch.long)
                if valid_n > 0:
                    input_ids[b, ls : ls + valid_n] = response_ids[b][r_start:r_end]
                    pad_mask[b, ls : ls + valid_n] = True
                position_ids[b, ls : ls + N] = shared_pos
                # On-policy bridge: append response[e_j : s_{j+1}) right after
                # the original-N clean block. Bridge positions get their own
                # RoPE (the actual response positions in the gap) so attention
                # sees them as if they were inline in the AR sequence.
                if cascade_drafts is not None and clean_size_j > N:
                    bridge_len = clean_size_j - N
                    bridge_start_resp = r_end  # = e_j
                    bridge_end_resp = bridge_start_resp + bridge_len
                    bridge_ls = ls + N
                    # input_ids: fill response[e_j : s_{j+1})
                    input_ids[b, bridge_ls : bridge_ls + bridge_len] = response_ids[b][
                        bridge_start_resp : bridge_end_resp
                    ]
                    pad_mask[b, bridge_ls : bridge_ls + bridge_len] = True
                    # position_ids: contiguous RoPE matching response positions.
                    position_ids[b, bridge_ls : bridge_ls + bridge_len] = torch.arange(
                        Pb + bridge_start_resp, Pb + bridge_end_resp, dtype=torch.long
                    )
                    # role / pair_idx (bridge is part of pair j's clean region).
                    if role_per_pos is not None:
                        role_per_pos[b, bridge_ls : bridge_ls + bridge_len] = 2
                        pair_idx_per_pos[b, bridge_ls : bridge_ls + bridge_len] = j
                # Build per-position "alternatives" pools from ALL cascade iters
                # that covered any part of this pair's window [r_start, r_end).
                # Source = the full trajectory list (with target_argmax), passed
                # alongside cascade_drafts (selected subset) via the
                # `_ConsDraftListWithAll.all_trajs` side-channel attribute.
                # Falls back to the selected subset if no full list is available.
                tile_alts = None
                if cascade_drafts is not None and K > 1:
                    all_trajs_b = getattr(
                        cascade_drafts, "all_trajs", None,
                    )
                    src_list = (
                        all_trajs_b[b]
                        if all_trajs_b is not None and b < len(all_trajs_b)
                        else cascade_drafts[b]
                    )
                    pool_per_pos = _build_alt_pools(src_list, r_start, N)
                    tile_alts = pool_per_pos
                for k_tile in range(K):
                    ks = group_start + k_tile * N
                    if onpol_noisy is not None and k_tile == 0:
                        # Tile 0: use the cascade-evolved on-policy draft.
                        input_ids[b, ks : ks + N] = onpol_noisy
                    elif onpol_noisy is not None and k_tile >= 1:
                        # Tile k≥1: per-position rank-shifted alternative from
                        # cascade target_argmax pools. Preserves matched prefix
                        # [0, pml). At position offset i past pml, use pool[i]'s
                        # rank ((k_tile - 1) + i) % len(pool) — different rank
                        # at different positions to maximize diversity across
                        # tile sequences (not just position-wise).
                        tile_noise = onpol_noisy.clone()
                        if pml < N and tile_alts is not None:
                            for i in range(pml, N):
                                pool = tile_alts[i]
                                if pool:
                                    rank = ((k_tile - 1) + (i - pml)) % len(pool)
                                    tile_noise[i] = int(pool[rank])
                                # else: keep cascade draft token
                        input_ids[b, ks : ks + N] = tile_noise
                    else:
                        input_ids[b, ks : ks + N] = _draw_noisy(generator)
                    pad_mask[b, ks : ks + N] = True
                    noisy_mask[b, ks : ks + N] = True
                    position_ids[b, ks : ks + N] = shared_pos
                # Populate per-pair start tables + per-position role tensors
                # (only used by the on-policy bridge path; off-policy keeps None).
                if noisy_starts is not None:
                    noisy_starts[b, j] = group_start  # K=1 in on-policy
                    clean_starts[b, j] = ls
                if role_per_pos is not None:
                    role_per_pos[b, group_start : group_start + K * N] = 1     # noisy
                    role_per_pos[b, ls : ls + N] = 2                          # actual clean
                    pair_idx_per_pos[b, group_start : group_start + K * N] = j
                    pair_idx_per_pos[b, ls : ls + N] = j
        else:
            # 3-block layout: [prompt | marked | unmarked | clean | marked | unmarked | clean | ...]
            # Each triple j has 3 sub-blocks at:
            #   marked  : [Pb + 3j*N      , Pb + (3j+1)*N)
            #   unmarked: [Pb + (3j+1)*N  , Pb + (3j+2)*N)
            #   clean   : [Pb + (3j+2)*N  , Pb + (3j+3)*N)
            for j in range(Tb):
                ms = Pb + 3 * j * N
                us = Pb + (3 * j + 1) * N
                ls = Pb + (3 * j + 2) * N
                r_start = j * N
                r_end = min((j + 1) * N, int(Rn[b].item()))
                valid_n = r_end - r_start
                if valid_n > 0:
                    input_ids[b, ls : ls + valid_n] = response_ids[b][r_start:r_end]
                    pad_mask[b, ls : ls + valid_n] = True
                # Same noisy tokens for marked and unmarked copies — only the
                # input-side marker embedding differs between the two slots.
                noisy_toks = _draw_noisy(generator)
                input_ids[b, ms : ms + N] = noisy_toks
                input_ids[b, us : us + N] = noisy_toks
                pad_mask[b, ms : ms + N] = True
                pad_mask[b, us : us + N] = True
                noisy_mask[b, ms : ms + N] = True
                noisy_mask[b, us : us + N] = True
                marked_mask[b, ms : ms + N] = True
                unmarked_mask[b, us : us + N] = True
                shared_pos = torch.arange(Pb + j * N, Pb + (j + 1) * N, dtype=torch.long)
                position_ids[b, ms : ms + N] = shared_pos
                position_ids[b, us : us + N] = shared_pos
                position_ids[b, ls : ls + N] = shared_pos

    return InterleavedBatch(
        input_ids=input_ids,
        position_ids=position_ids,
        prompt_lens=P,
        num_pairs=T_full,
        block_lens=torch.full((B,), N, dtype=torch.long),
        seq_lens=seq_lens,
        pad_mask=pad_mask,
        noisy_mask=noisy_mask,
        num_noisy_tiles=K,
        triple_mode=triple_mode,
        marked_mask=marked_mask,
        unmarked_mask=unmarked_mask,
        onpolicy_prefix_lens=onpolicy_prefix_lens,
        noisy_starts=noisy_starts,
        clean_starts=clean_starts,
        role_per_pos=role_per_pos,
        pair_idx_per_pos=pair_idx_per_pos,
        canvas_pairs=canvas_pairs,
        canvas_mask=canvas_mask,
    )


if __name__ == "__main__":
    # quick self-test
    g = torch.Generator().manual_seed(42)
    prompt_ids = [torch.tensor([1, 2, 3, 4, 5]), torch.tensor([10, 11, 12])]
    response_ids = [torch.tensor([20, 21, 22, 23, 24, 25, 26, 27]),
                    torch.tensor([30, 31, 32, 33])]
    b = build_interleaved_batch(prompt_ids, response_ids, block_size=4, pad_id=0, generator=g)
    print("seq_lens:", b.seq_lens.tolist())
    print("num_pairs:", b.num_pairs.tolist())
    print("input_ids[0]:", b.input_ids[0].tolist())
    print("position_ids[0]:", b.position_ids[0].tolist())
    print("pad_mask[0]:", b.pad_mask[0].tolist())
