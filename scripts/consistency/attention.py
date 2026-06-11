"""SDPA-based attention helpers for the Decode-Learning interleave pattern.

Why SDPA and not flex_attention:
  flex_attention requires a dispatch path that verl's flash_attn varlen
  monkey patch + FSDP2 wrap don't honor cleanly (we tried; it kept routing
  to flash_attention_forward via the class-level dispatch). SDPA, by
  contrast, just consumes a standard 4-D float `attention_mask` and works
  through HF's plain dispatch.

What we provide:
  - `make_mask_mod` (pure-tensor boolean): port of Decode-Learning's
    `_mask_mod_varlen` in Jacobi mode (causal within noisy block).
  - `build_sdpa_attention_mask` materializes the boolean grid into a (B, 1,
    Lmax, Lmax) additive float mask suitable for SDPA. 0.0 = allowed,
    -inf = blocked.
  - `swap_attention_impl` flips `model.config._attn_implementation` to
    "sdpa" for the duration of the consistency forward, then restores.
"""

from __future__ import annotations

import contextlib
import os
from typing import Iterator

import torch


def make_mask_mod(
    prompt_lens: torch.Tensor,
    num_pairs: torch.Tensor,
    block_lens: torch.Tensor,
    pad_mask: torch.Tensor,
    triple_mode: bool = False,
    causal_region_size: int = 0,
    num_noisy_tiles: int = 1,
    canvas_pairs: torch.Tensor | None = None,
):
    """Build a mask_mod closure with batched per-sample metadata baked in.
    Returns a callable `(b, h, q, k) -> bool` (broadcastable).

    - 2-block mode (default): [prompt | noisy | clean | noisy | clean | ...]
      Triple j has 2 sub-blocks; block_idx_q % 2 == 0 → noisy, == 1 → clean.
    - 3-block mode: [prompt | marked | unmarked | clean | marked | unmarked | clean | ...]
      Triple j has 3 sub-blocks; block_idx_q % 3 == 0 → marked-noisy,
      == 1 → unmarked-noisy, == 2 → clean. marked and unmarked do NOT attend
      to each other (they share RoPE positions but are independent slots).

    causal_region_size: hybrid AR/bidirectional mask inside each noisy block
      (the draft block being trained for parallel decoding consistency).
        0  (default) : pure causal within the noisy block — current behavior,
                       bit-identical to the original training recipe.
        N  (0 < N < block_len) : the first N intra-block positions remain causal
                       (AR-equivalent verification region), and the remaining
                       (block_len - N) positions attend bidirectionally to the
                       entire noisy block. Matches the TiDAR-style structured
                       mask used in inference; only the first N positions get
                       committed at decode time so AR-equivalence is preserved.
        -1 : FULLY BIDIRECTIONAL — every position in the noisy block attends
             to every other position in the same noisy block (no causal
             constraint). Matches dFlash / Fast-dLLM v2 within-block attention.
             AR-equivalence is lost; only use when AR equivalence isn't needed.
        N >= block_len : effectively pure causal (no positions in bidir region).

    canvas_pairs (v11): optional (B, T_max) bool. Pairs flagged True are
      CANVAS pairs — their noisy tile is fully bidirectional within itself
      (cr=-1 semantics) regardless of `causal_region_size`, which continues
      to govern the non-canvas (causal) pairs. None → bit-identical legacy
      behavior. 2-block mode only.
    """
    if triple_mode and canvas_pairs is not None:
        raise ValueError("canvas_pairs is incompatible with triple_mode")
    P_t = prompt_lens
    T_t = num_pairs
    N_t = torch.clamp(block_lens, min=1)
    V_t = pad_mask
    cr = int(causal_region_size)
    full_bidir = cr < 0
    hybrid_mask = cr > 0  # if 0 or -1, the per-q-position causal split isn't needed
    K = int(num_noisy_tiles)
    assert K >= 1, f"num_noisy_tiles must be >= 1, got {K}"
    G = K + 1  # tiles per block-group: K noisy + 1 clean

    if not triple_mode:
        def mask_mod(b, h, q, k):
            b = b.long()
            p = P_t[b]
            T = T_t[b]
            N = N_t[b]
            in_range = V_t[b, q] & V_t[b, k]

            is_prompt_q = q < p
            is_prompt_k = k < p
            mask_prompt = is_prompt_q & (k <= q)

            rel_q = q - p
            rel_k = k - p
            block_idx_q = torch.div(rel_q, N, rounding_mode="floor")
            block_idx_k = torch.div(rel_k, N, rounding_mode="floor")

            # Sub-block role: 0..K-1 = noisy tiles, K = clean.
            sub_q = block_idx_q % G
            sub_k = block_idx_k % G
            is_noisy_q = (~is_prompt_q) & (sub_q < K)
            is_clean_q = (~is_prompt_q) & (sub_q == K)
            is_noisy_k = (~is_prompt_k) & (sub_k < K)
            is_clean_k = (~is_prompt_k) & (sub_k == K)

            Tmax = torch.maximum(T - 1, torch.zeros_like(T))
            j_q_unc = block_idx_q // G
            j_q = torch.minimum(torch.maximum(j_q_unc, torch.zeros_like(j_q_unc)), Tmax)

            # Q's own tile start (one of K noisy tiles) and the clean tile of its group.
            ks_ = p + G * j_q * N + sub_q * N        # start of q's own noisy tile (valid only when q is noisy)
            ls_ = p + (G * j_q + K) * N              # start of q's group's clean tile

            # Previous CLEAN tiles are at sub_k == K with block_idx_k < G*j_q.
            clean_in_prev_clean = is_clean_k & (block_idx_k < G * j_q)
            # Sibling noisy tiles in same group must NOT see each other:
            # require same block_idx (not just same group) for noisy↔noisy.
            same_noisy_tile = is_noisy_q & is_noisy_k & (block_idx_q == block_idx_k)
            same_clean_block = is_clean_q & is_clean_k & (block_idx_q == block_idx_k)

            if hybrid_mask:
                q_in_block = q - ks_
                q_in_causal = q_in_block < cr
                allow_in_tile = (q_in_causal & (k <= q)) | (~q_in_causal)
            elif full_bidir:
                allow_in_tile = None  # whole tile already allowed
            else:
                allow_in_tile = k <= q
            if canvas_pairs is not None and allow_in_tile is not None:
                # Canvas pairs: full bidir within their own noisy tile,
                # independent of the global causal_region_size.
                allow_in_tile = allow_in_tile | canvas_pairs[b, j_q]
            if allow_in_tile is None:
                same_noisy_attn = same_noisy_tile & (k >= ks_)
            else:
                same_noisy_attn = same_noisy_tile & (k >= ks_) & allow_in_tile

            mask_noisy = is_noisy_q & (is_prompt_k | clean_in_prev_clean | same_noisy_attn)
            mask_clean = is_clean_q & (
                is_prompt_k | clean_in_prev_clean | (same_clean_block & (k >= ls_) & (k <= q))
            )

            return in_range & (mask_prompt | mask_noisy | mask_clean)

        return mask_mod

    # 3-block (triple) mode.
    def mask_mod(b, h, q, k):
        b = b.long()
        p = P_t[b]
        T = T_t[b]
        N = N_t[b]
        in_range = V_t[b, q] & V_t[b, k]

        is_prompt_q = q < p
        is_prompt_k = k < p
        mask_prompt = is_prompt_q & (k <= q)

        rel_q = q - p
        rel_k = k - p
        block_idx_q = torch.div(rel_q, N, rounding_mode="floor")
        block_idx_k = torch.div(rel_k, N, rounding_mode="floor")

        sub_q = block_idx_q % 3
        sub_k = block_idx_k % 3
        is_marked_q   = (~is_prompt_q) & (sub_q == 0)
        is_unmarked_q = (~is_prompt_q) & (sub_q == 1)
        is_clean_q    = (~is_prompt_q) & (sub_q == 2)
        is_marked_k   = (~is_prompt_k) & (sub_k == 0)
        is_unmarked_k = (~is_prompt_k) & (sub_k == 1)
        is_clean_k    = (~is_prompt_k) & (sub_k == 2)

        Tmax = torch.maximum(T - 1, torch.zeros_like(T))
        j_q_unc = block_idx_q // 3
        j_q = torch.minimum(torch.maximum(j_q_unc, torch.zeros_like(j_q_unc)), Tmax)

        # Per-block start positions (within the packed sequence).
        ms_ = p + 3 * j_q * N
        us_ = p + (3 * j_q + 1) * N
        ls_ = p + (3 * j_q + 2) * N

        # Previous clean blocks are at sub_k == 2 with block_idx_k < 3*j_q.
        clean_in_prev_clean = is_clean_k & (block_idx_k < 3 * j_q)

        same_triple = (j_q_unc == (block_idx_k // 3))
        same_marked = is_marked_q & is_marked_k & same_triple
        same_unmarked = is_unmarked_q & is_unmarked_k & same_triple
        same_clean = is_clean_q & is_clean_k & same_triple

        if hybrid_mask:
            # Hybrid causal + bidirectional inside both noisy sub-blocks
            # (marked and unmarked). First `cr` intra-block positions are
            # causal; the rest are bidirectional within the same sub-block.
            q_in_marked = q - ms_
            q_in_unmarked = q - us_
            q_in_causal_m = q_in_marked < cr
            q_in_causal_u = q_in_unmarked < cr
            same_marked_attn = same_marked & (k >= ms_) & (
                (q_in_causal_m & (k <= q)) | (~q_in_causal_m)
            )
            same_unmarked_attn = same_unmarked & (k >= us_) & (
                (q_in_causal_u & (k <= q)) | (~q_in_causal_u)
            )
        else:
            # Jacobi-style intra-block attention for noisy slots; causal within block.
            same_marked_attn = same_marked & (k >= ms_) & (k <= q)
            same_unmarked_attn = same_unmarked & (k >= us_) & (k <= q)
        # Causal within the clean block (always causal — these are the
        # AR-target supervision positions and must remain AR-faithful).
        same_clean_attn = same_clean & (k >= ls_) & (k <= q)

        mask_marked = is_marked_q & (
            is_prompt_k | clean_in_prev_clean | same_marked_attn
        )
        mask_unmarked = is_unmarked_q & (
            is_prompt_k | clean_in_prev_clean | same_unmarked_attn
        )
        mask_clean = is_clean_q & (
            is_prompt_k | clean_in_prev_clean | same_clean_attn
        )

        return in_range & (mask_prompt | mask_marked | mask_unmarked | mask_clean)

    return mask_mod


def make_mask_mod_variable(
    pair_idx_per_pos: torch.Tensor,   # (B, Lmax) int, -1 for prompt/pad
    role_per_pos: torch.Tensor,        # (B, Lmax) int: 0=prompt 1=noisy 2=clean(+bridge)
    noisy_starts: torch.Tensor,        # (B, T_max) start position of noisy block-0 per pair
    pad_mask: torch.Tensor,            # (B, Lmax) bool
    block_len: int = 32,               # N (block size); needed to compute tile_id for K>1 on-policy
    canvas_pairs: torch.Tensor | None = None,  # (B, T_max) bool — v11 canvas pairs (full bidir in own tile)
):
    """Variable-layout mask: uses per-position role/pair_idx lookups instead
    of modular arithmetic from a uniform stride. Required for on-policy bridge
    mode where pair j's clean block can be longer than N (=N + gap_{j+1}).

    Multi-tile (K>1) safety: when a pair has K noisy tiles laid out
    contiguously starting at noisy_starts[b, j], sibling tiles must NOT see
    each other. We compute tile_id = (pos - noisy_start) // N and gate
    noisy↔noisy attention on tile_q == tile_k.
    """
    PIP = pair_idx_per_pos
    ROL = role_per_pos
    NS = noisy_starts
    PAD = pad_mask
    N = int(block_len)

    def mask_mod(b, h, q, k):
        b = b.long()
        in_range = PAD[b, q] & PAD[b, k]
        role_q = ROL[b, q]
        role_k = ROL[b, k]
        pair_q = PIP[b, q]
        pair_k = PIP[b, k]
        is_prompt_q = role_q == 0
        is_prompt_k = role_k == 0
        is_noisy_q = role_q == 1
        is_noisy_k = role_k == 1
        is_clean_q = role_q == 2
        is_clean_k = role_k == 2
        # Prompt-as-query: causal within prompt only.
        mask_prompt = is_prompt_q & is_prompt_k & (k <= q)
        # Previous-pair clean blocks (includes bridge regions of those pairs).
        clean_in_prev_clean = is_clean_k & (pair_k >= 0) & (pair_q >= 0) & (pair_k < pair_q)
        # Same-pair clean attending to clean (causal within the extended clean).
        same_pair_clean_attn = is_clean_q & is_clean_k & (pair_q == pair_k) & (k <= q)
        # Same-pair noisy attending to its OWN tile only (siblings excluded).
        # Tile id = floor((pos - noisy_start) / N). Pairs are gated above by
        # pair_q == pair_k so we only need the in-tile constraint here.
        pair_q_safe = torch.clamp(pair_q, min=0)
        pair_k_safe = torch.clamp(pair_k, min=0)
        own_noisy_start_q = NS[b, pair_q_safe]
        own_noisy_start_k = NS[b, pair_k_safe]
        tile_q = torch.div(q - own_noisy_start_q, N, rounding_mode="floor")
        tile_k = torch.div(k - own_noisy_start_k, N, rounding_mode="floor")
        # v11 canvas pairs: full bidir within the tile. The upper bound is
        # implied by tile_q == tile_k (k stays inside q's own N-token tile).
        in_tile_allow = k <= q
        if canvas_pairs is not None:
            in_tile_allow = in_tile_allow | canvas_pairs[b, pair_q_safe]
        same_pair_noisy_attn = (
            is_noisy_q & is_noisy_k
            & (pair_q == pair_k)
            & (tile_q == tile_k)
            & (k >= own_noisy_start_q + tile_q * N)
            & in_tile_allow
        )
        mask_noisy = is_noisy_q & (is_prompt_k | clean_in_prev_clean | same_pair_noisy_attn)
        mask_clean = is_clean_q & (is_prompt_k | clean_in_prev_clean | same_pair_clean_attn)
        return in_range & (mask_prompt | mask_noisy | mask_clean)

    return mask_mod


def build_sdpa_attention_mask(
    prompt_lens: torch.Tensor,
    num_pairs: torch.Tensor,
    block_lens: torch.Tensor,
    pad_mask: torch.Tensor,
    device,
    dtype=torch.float32,
    triple_mode: bool = False,
    causal_region_size: int = 0,
    num_noisy_tiles: int = 1,
    role_per_pos: torch.Tensor | None = None,
    pair_idx_per_pos: torch.Tensor | None = None,
    noisy_starts: torch.Tensor | None = None,
    canvas_pairs: torch.Tensor | None = None,
) -> torch.Tensor:
    """Materialize the boolean attention pattern into a (B, 1, Lmax, Lmax)
    additive float mask. 0.0 = allowed, -inf = blocked. SDPA broadcasts
    over the head dimension.

    `causal_region_size`: see make_mask_mod docstring. 0 = pure causal
    (default; bit-identical to original training recipe).

    When `role_per_pos`, `pair_idx_per_pos`, and `noisy_starts` are all
    provided (on-policy bridge mode), uses the variable-layout mask path
    (`make_mask_mod_variable`) which looks up per-position role/pair_idx
    instead of using uniform-stride modular arithmetic. Required because
    pair j's clean block may be longer than N (N + gap_{j+1} for non-last).
    """
    B, Lmax = pad_mask.shape
    use_variable = (
        role_per_pos is not None
        and pair_idx_per_pos is not None
        and noisy_starts is not None
        and not triple_mode
        and causal_region_size == 0
    )
    if use_variable:
        # block_len is uniform across pairs (= N); read from first sample.
        N_for_mask = int(block_lens[0].item()) if block_lens is not None and block_lens.numel() > 0 else 32
        fn = make_mask_mod_variable(
            pair_idx_per_pos=pair_idx_per_pos,
            role_per_pos=role_per_pos,
            noisy_starts=noisy_starts,
            pad_mask=pad_mask,
            block_len=N_for_mask,
            canvas_pairs=canvas_pairs,
        )
    else:
        fn = make_mask_mod(
            prompt_lens, num_pairs, block_lens, pad_mask,
            triple_mode=triple_mode, causal_region_size=causal_region_size,
            num_noisy_tiles=num_noisy_tiles,
            canvas_pairs=canvas_pairs,
        )

    b_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, Lmax, Lmax)
    h_idx = torch.zeros((1,), dtype=torch.long, device=device)  # broadcastable
    qs = torch.arange(Lmax, device=device).view(1, Lmax, 1).expand(B, Lmax, Lmax)
    ks = torch.arange(Lmax, device=device).view(1, 1, Lmax).expand(B, Lmax, Lmax)

    bool_mask = fn(b_idx, h_idx, qs, ks)  # (B, Lmax, Lmax) bool, True=allowed

    # Additive: 0 where allowed, large negative where blocked.
    # Avoid full -inf to keep bf16/half-precision softmax stable.
    NEG = torch.finfo(dtype).min / 2
    additive = torch.zeros((B, 1, Lmax, Lmax), dtype=dtype, device=device)
    additive.masked_fill_(~bool_mask.unsqueeze(1), NEG)
    return additive


@contextlib.contextmanager
def swap_attention_impl(model, target_impl: str = "sdpa") -> Iterator[None]:
    """Temporarily set `model.config._attn_implementation` to `target_impl`.

    Standard AR training path is unchanged — outside this context, the
    original impl is restored.
    """
    target = model
    for _ in range(8):
        inner = getattr(target, "module", None)
        if inner is None:
            break
        target = inner
    cfg_obj = getattr(target, "config", None)
    if cfg_obj is None:
        raise RuntimeError("Could not find .config on model for attention impl swap")

    saved_impl = getattr(cfg_obj, "_attn_implementation", None)

    if os.environ.get("CONSISTENCY_DEBUG", "0") == "1":
        import sys as _sys
        print(
            f"[swap_attention_impl] target_class={type(target).__name__} "
            f"saved_impl={saved_impl} -> {target_impl}",
            flush=True,
            file=_sys.stderr,
        )

    try:
        cfg_obj._attn_implementation = target_impl
        yield
    finally:
        if saved_impl is not None:
            cfg_obj._attn_implementation = saved_impl


if __name__ == "__main__":
    # Self-test: directly evaluate mask_mod on a small grid + visualize.
    P = torch.tensor([2])
    T = torch.tensor([2])
    N = torch.tensor([4])
    Lmax = int((P + 2 * T * N).item())
    pad = torch.ones((1, Lmax), dtype=torch.bool)

    def show(cr, label):
        add_mask = build_sdpa_attention_mask(
            P, T, N, pad, device="cpu", dtype=torch.float32, causal_region_size=cr,
        )
        bool_view = add_mask[0, 0] >= -1e30
        print(f"\n=== {label} (causal_region_size={cr}) ===")
        print("Layout: prompt(2) | k_0(4) last_0(4) | k_1(4) last_1(4)")
        for i in range(Lmax):
            print(f"  q={i:2d}  " + "".join("#" if bool_view[i, j] else "." for j in range(Lmax)))

    show(0, "default — pure causal noisy block (original behavior)")
    show(2, "hybrid 2+2 — first 2 noisy positions causal, last 2 bidirectional")
    show(4, "cr == N — effectively pure causal (degenerate)")
