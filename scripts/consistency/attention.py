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
):
    """Build a mask_mod closure with batched per-sample metadata baked in.
    Returns a callable `(b, h, q, k) -> bool` (broadcastable)."""
    P_t = prompt_lens
    T_t = num_pairs
    N_t = torch.clamp(block_lens, min=1)
    V_t = pad_mask

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

        is_noisy_q = (~is_prompt_q) & (block_idx_q % 2 == 0)
        is_clean_q = (~is_prompt_q) & (block_idx_q % 2 == 1)
        is_noisy_k = (~is_prompt_k) & (block_idx_k % 2 == 0)
        is_clean_k = (~is_prompt_k) & (block_idx_k % 2 == 1)

        Tmax = torch.maximum(T - 1, torch.zeros_like(T))
        j_q_unc = block_idx_q // 2
        j_q = torch.minimum(torch.maximum(j_q_unc, torch.zeros_like(j_q_unc)), Tmax)

        ks_ = p + 2 * j_q * N
        ls_ = p + (2 * j_q + 1) * N

        clean_in_prev_clean = is_clean_k & (block_idx_k < 2 * j_q)
        same_noisy_block = is_noisy_q & is_noisy_k & (block_idx_q == block_idx_k)
        same_clean_block = is_clean_q & is_clean_k & (block_idx_q == block_idx_k)

        # Jacobi mode: causal within the noisy block.
        same_noisy_attn = same_noisy_block & (k >= ks_) & (k <= q)

        mask_noisy = is_noisy_q & (is_prompt_k | clean_in_prev_clean | same_noisy_attn)
        mask_clean = is_clean_q & (
            is_prompt_k | clean_in_prev_clean | (same_clean_block & (k >= ls_) & (k <= q))
        )

        return in_range & (mask_prompt | mask_noisy | mask_clean)

    return mask_mod


def build_sdpa_attention_mask(
    prompt_lens: torch.Tensor,
    num_pairs: torch.Tensor,
    block_lens: torch.Tensor,
    pad_mask: torch.Tensor,
    device,
    dtype=torch.float32,
) -> torch.Tensor:
    """Materialize the boolean attention pattern into a (B, 1, Lmax, Lmax)
    additive float mask. 0.0 = allowed, -inf = blocked. SDPA broadcasts
    over the head dimension.
    """
    B, Lmax = pad_mask.shape
    fn = make_mask_mod(prompt_lens, num_pairs, block_lens, pad_mask)

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
    N = torch.tensor([3])
    Lmax = int((P + 2 * T * N).item())
    pad = torch.ones((1, Lmax), dtype=torch.bool)
    add_mask = build_sdpa_attention_mask(P, T, N, pad, device="cpu", dtype=torch.float32)
    print(f"SDPA additive mask shape: {tuple(add_mask.shape)}  (Lmax={Lmax})")
    bool_view = add_mask[0, 0] >= -1e30  # True = allowed
    print("Layout: prompt(2) | k_0(3) last_0(3) | k_1(3) last_1(3)")
    print("rows = query, cols = key. '#' = attend, '.' = blocked")
    for i in range(Lmax):
        print(f"  q={i:2d}  " + "".join("#" if bool_view[i, j] else "." for j in range(Lmax)))
