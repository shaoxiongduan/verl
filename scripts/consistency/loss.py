"""Consistency loss orchestrator: runs ONE extra forward pass on the
interleaved batch under flex_attention, then computes soft cross-entropy
between student (noisy block) and teacher (clean block, detached) logits.

This is a *separate* forward call from verl's standard AR pass — the AR
training logic is untouched.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F

import contextlib

from .attention import build_sdpa_attention_mask, swap_attention_impl
from .pack import InterleavedBatch, build_interleaved_batch


def _per_block_sinusoidal(T_max: int, H: int, device, dtype) -> torch.Tensor:
    """Standard sinusoidal positional encoding indexed by block number.
    Shape (T_max, H). Same formula as Vaswani et al.'s positional embedding,
    just keyed by block index instead of token position. Plain CUDA tensor
    with own storage on every rank — no FSDP entanglement.
    """
    pe = torch.zeros(T_max, H, device=device, dtype=torch.float32)
    pos = torch.arange(T_max, device=device, dtype=torch.float32).unsqueeze(-1)  # (T_max, 1)
    div_term = torch.exp(
        torch.arange(0, H, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / H)
    )
    pe[:, 0::2] = torch.sin(pos * div_term)
    pe[:, 1::2] = torch.cos(pos * div_term)
    return pe.to(dtype)


def _constant_marker(H: int, device, dtype, seed: int = 42) -> torch.Tensor:
    """Fixed (H,) marker vector — same value for every noise token, every block.
    The block index of the token does NOT affect the marker. Tests whether the
    per-block variation in `_per_block_sinusoidal` carries useful information
    vs. just acting as a static "I am a draft" signal.

    Uses the standard sinusoidal pattern at block_idx=0 (≡ first row of
    `_per_block_sinusoidal`'s table). This keeps the magnitude profile across
    dimensions identical to the per-block variant — only the variation across
    positions is removed. Deterministic; identical on every rank.
    """
    pe = _per_block_sinusoidal(1, H, device, dtype)  # (1, H)
    return pe.squeeze(0)  # (H,)


@contextlib.contextmanager
def _no_gradient_checkpointing(model):
    """Temporarily disable gradient checkpointing on every submodule that
    has the flag. Without this, gradient checkpointing recomputes the
    forward at backward time — and reads `config._attn_implementation`
    *as of backward time*. Our `swap_attention_impl` context has already
    exited by then, so the recompute uses `flash_attention_2` with our
    4D float SDPA mask, which crashes with OOB gather in attention.
    """
    target = model
    for _ in range(8):
        inner = getattr(target, "module", None)
        if inner is None:
            break
        target = inner
    saved = []
    for m in target.modules():
        if getattr(m, "gradient_checkpointing", False):
            saved.append(m)
            m.gradient_checkpointing = False
    try:
        yield len(saved)
    finally:
        for m in saved:
            m.gradient_checkpointing = True


def soft_cross_entropy(predicts: torch.Tensor, targets: torch.Tensor, T_soft: float = 1.0,
                       divergence: str = "forward_kl", reduction: str = "mean") -> torch.Tensor:
    """Soft divergence between student (`predicts`) and teacher (`targets`) logits.

    divergence:
      "forward_kl"  (default) — Hinton-style soft CE: H(softmax(t/T), softmax(s/T)).
                                With detached targets, gradient ≡ forward
                                KL(softmax(t).detach() || softmax(s)). Mass-covering:
                                student tries to cover all modes of teacher.
      "reverse_kl"            — KL(softmax(s) || softmax(t).detach()). Mode-seeking:
                                student picks one mode of teacher rather than covering.
                                Recommended by GKD (Agarwal 2024) / MiniLLM (Gu 2024)
                                when teacher distribution is sharp post-RL.
      "jsd"                   — 0.5*KL(s||m) + 0.5*KL(t||m), m=(s+t)/2. Symmetric
                                middle-ground; less mode-seeking than reverse_kl.

    All variants multiply by T_soft^2 outside if you want gradient-matched scale.
    """
    if predicts.numel() == 0:
        return predicts.sum() * 0.0
    log_q = F.log_softmax(predicts / T_soft, dim=-1)
    log_p = F.log_softmax(targets / T_soft, dim=-1)  # `targets` is already detached upstream
    if divergence == "forward_kl":
        # H(softmax(t), softmax(s)) = -sum( softmax(t) * log_softmax(s) )
        p = log_p.exp()
        per_pos = -(p * log_q).sum(dim=-1)
    elif divergence == "reverse_kl":
        # sum( softmax(s) * (log_softmax(s) - log_softmax(t).detach()) )
        q = log_q.exp()
        per_pos = (q * (log_q - log_p)).sum(dim=-1)
    elif divergence == "jsd":
        # m = (q + p) / 2  (mixture; in log space: logsumexp([log_q, log_p]) - log 2)
        import math
        log_m = torch.logsumexp(torch.stack([log_q, log_p], dim=0), dim=0) - math.log(2.0)
        q = log_q.exp()
        p = log_p.exp()
        per_pos = 0.5 * ((q * (log_q - log_m)).sum(dim=-1) + (p * (log_p - log_m)).sum(dim=-1))
    else:
        raise ValueError(f"Unknown divergence: {divergence!r}")
    if reduction == "none":
        return per_pos
    return per_pos.mean()


@torch.no_grad()
def _identify_block_positions(
    batch: InterleavedBatch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """For each sample b, return (b_idx, k_pos, l_pos, pos_in_block, pair_idx):
    the (sample, position, position-in-block, pair-index) tuples of noisy and
    clean blocks aligned pair-by-pair. Used in 2-block mode.

    `pos_in_block` is the intra-block offset (0..N-1) — needed for dFlash-style
    position-decay weighting w_k = exp(-(k-1)/gamma).
    `pair_idx` is the pair index j ∈ [0, T_b) — needed for on-policy
    `onpolicy_prefix_lens` lookup (per-pair prefix-match length).

    Output shapes: each is a flat int64 tensor of length sum_b T_b * N_b.
    """
    B = batch.input_ids.shape[0]
    device = batch.input_ids.device
    b_list, k_list, l_list, pos_list, pair_list = [], [], [], [], []
    P = batch.prompt_lens
    T = batch.num_pairs
    N = batch.block_lens
    pad = batch.pad_mask
    K = int(getattr(batch, "num_noisy_tiles", 1))
    G = K + 1  # tiles per group
    # On-policy bridge mode: per-pair start tables override the uniform stride.
    noisy_starts = getattr(batch, "noisy_starts", None)
    clean_starts = getattr(batch, "clean_starts", None)
    use_variable = noisy_starts is not None and clean_starts is not None
    for b in range(B):
        Pb = int(P[b].item())
        Tb = int(T[b].item())
        Nb = int(N[b].item())
        if Tb <= 0 or Nb <= 0:
            continue
        for j in range(Tb):
            if use_variable:
                # Variable-layout: read per-pair starts. K must be 1 (enforced
                # by pack.py for on-policy mode).
                group_start = int(noisy_starts[b, j].item())
                ls = int(clean_starts[b, j].item())
            else:
                group_start = Pb + G * j * Nb
                ls = group_start + K * Nb
            offs = torch.arange(Nb, dtype=torch.long, device=device)
            for k_tile in range(K):
                ks = group_start + k_tile * Nb
                # Use ONLY the first N positions of the clean block — the
                # bridge extension (positions N..N+gap-1) doesn't have a noisy
                # counterpart and shouldn't enter the cons loss pairing.
                keep = pad[b, ks : ks + Nb] & pad[b, ls : ls + Nb]
                if not keep.any():
                    continue
                kept_offs = offs[keep]
                b_list.append(torch.full((kept_offs.numel(),), b, dtype=torch.long, device=device))
                k_list.append(ks + kept_offs)
                l_list.append(ls + kept_offs)
                pos_list.append(kept_offs)
                pair_list.append(torch.full((kept_offs.numel(),), j, dtype=torch.long, device=device))
    if not b_list:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty, empty, empty, empty
    return (torch.cat(b_list), torch.cat(k_list), torch.cat(l_list),
            torch.cat(pos_list), torch.cat(pair_list))


@torch.no_grad()
def _identify_block_positions_triple(
    batch: InterleavedBatch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """3-block variant: returns (b_idx, m_pos, u_pos, l_pos) — sample,
    marked-noisy, unmarked-noisy, clean — aligned triple-by-triple.
    """
    B = batch.input_ids.shape[0]
    device = batch.input_ids.device
    b_list, m_list, u_list, l_list = [], [], [], []
    P = batch.prompt_lens
    T = batch.num_pairs
    N = batch.block_lens
    pad = batch.pad_mask
    for b in range(B):
        Pb = int(P[b].item())
        Tb = int(T[b].item())
        Nb = int(N[b].item())
        if Tb <= 0 or Nb <= 0:
            continue
        for j in range(Tb):
            ms = Pb + 3 * j * Nb
            us = Pb + (3 * j + 1) * Nb
            ls = Pb + (3 * j + 2) * Nb
            offs = torch.arange(Nb, dtype=torch.long, device=device)
            keep = pad[b, ms : ms + Nb] & pad[b, us : us + Nb] & pad[b, ls : ls + Nb]
            if not keep.any():
                continue
            kept_offs = offs[keep]
            b_list.append(torch.full((kept_offs.numel(),), b, dtype=torch.long, device=device))
            m_list.append(ms + kept_offs)
            u_list.append(us + kept_offs)
            l_list.append(ls + kept_offs)
    if not b_list:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty, empty, empty
    return torch.cat(b_list), torch.cat(m_list), torch.cat(u_list), torch.cat(l_list)


def _mask_tail_dflash_loss(
    logits: torch.Tensor,
    batch: "InterleavedBatch",
    b_idx: torch.Tensor,
    k_pos: torch.Tensor,
    l_pos: torch.Tensor,
    pos_in_block: torch.Tensor,
    pair_idx: torch.Tensor,
    mask_id: int,
    margin: int,
):
    """Idea-A mask-tail consistency loss (flat-weighted CE).

    For each pair, let ``pml`` be the cascade-converged prefix length
    (``onpolicy_prefix_lens``; 0 off-policy) and ``cut = pml + margin``:
      - block positions ``[pml, cut)``  -> CLEAN target (next rollout token),
      - block positions ``[cut, N)``    -> MASK-token target (`mask_id`),
      - block positions ``[0, pml)``     -> skipped (already converged).

    The dFlash position-decay weight is intentionally DROPPED here: a flat
    weight avoids a strength discontinuity at the clean->mask boundary. The
    model is taught to draft only ``margin`` tokens past what it has already
    committed and to emit a canonical "undecided" mask token beyond that,
    instead of confident-but-wrong tokens.

    Returns (cons_loss, n_clean_positions, n_mask_positions).
    """
    Nb = int(batch.block_lens[0].item())
    next_pos = l_pos + 1
    opl = getattr(batch, "onpolicy_prefix_lens", None)
    if opl is not None and pair_idx is not None and pair_idx.numel() > 0:
        pref = opl[b_idx, pair_idx]
    else:
        pref = torch.zeros_like(pos_in_block)
    cut = pref + int(margin)

    Lmax = batch.pad_mask.shape[-1]
    # Clean region needs a real (non-pad) NEXT token (shift-by-1 AR target) and
    # an in-block successor; the mask region needs neither (target is constant).
    next_real = batch.pad_mask[b_idx, torch.clamp(next_pos, max=Lmax - 1)]
    clean_sel = (
        (pos_in_block >= pref)
        & (pos_in_block < cut)
        & (pos_in_block < (Nb - 1))
        & next_real
    )
    mask_sel = pos_in_block >= cut

    terms = []
    n_clean = int(clean_sel.sum().item())
    n_mask = int(mask_sel.sum().item())
    if n_clean > 0:
        bc, kc, nc = b_idx[clean_sel], k_pos[clean_sel], next_pos[clean_sel]
        terms.append(
            F.cross_entropy(
                logits[bc, kc, :].float(), batch.input_ids[bc, nc], reduction="none"
            )
        )
    if n_mask > 0:
        bm, km = b_idx[mask_sel], k_pos[mask_sel]
        tgt = torch.full((n_mask,), int(mask_id), dtype=torch.long, device=logits.device)
        terms.append(
            F.cross_entropy(logits[bm, km, :].float(), tgt, reduction="none")
        )
    if terms:
        return torch.cat(terms).mean(), n_clean, n_mask
    return logits.sum() * 0.0, 0, 0


def compute_consistency_loss(
    model,
    prompt_ids: list[torch.Tensor],
    response_ids: list[torch.Tensor],
    *,
    block_size: int = 32,
    pad_id: int,
    max_pairs: int | None = None,
    T_soft: float = 1.0,
    seed: int | None = None,
    device=None,
    divergence: str = "forward_kl",
    teacher_model=None,
    compute_anchor: bool = False,
    anchor_mode: str = "clean",
    marker_embed_override: torch.Tensor | None = None,
    cascade_drafts: list[list[tuple[int, int, tuple]]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, dict]:
    """One consistency forward + soft CE loss.

    Args:
      model:          the FSDP-wrapped actor (HF Qwen2-arch)
      prompt_ids:     list[Tensor[Lp_i]] (CPU or device)
      response_ids:   list[Tensor[Lr_i]]
      block_size:     N (Jacobi block size; 32 matches JF Coder)
      pad_id:         the tokenizer's pad token id
      max_pairs:      cap on T per sample (None = no cap)
      T_soft:         softmax temperature for soft CE
      seed:           torch RNG seed for the noisy init
      device:         device for the forward; defaults to model device
      divergence:     "forward_kl" (default, mass-covering — student covers teacher),
                       "reverse_kl" (mode-seeking — student picks one teacher mode),
                       "jsd" (symmetric Jensen-Shannon).

    Returns:
      (loss, metrics) where loss is a scalar tensor on `device`, and
      metrics is a dict with 'cons_loss', 'cons_n_pairs', 'cons_n_pos'.
    """
    if device is None:
        device = next(model.parameters()).device

    gen = torch.Generator()
    if seed is not None:
        gen = gen.manual_seed(int(seed))

    # Build packed batch on CPU then ship to device. Triple_mode activates
    # the (marked, unmarked, clean) 3-block layout for the noisy_unmarked
    # anchor; gives the student both marked and unmarked noisy predictions
    # in a SINGLE forward via block-diagonal attention masking.
    triple_mode = (compute_anchor and anchor_mode == "noisy_unmarked")
    # Multi-noise tiling: K=1 (default) is the historical layout. K>1 emits
    # K independent noisy tiles per clean block (Monte-Carlo over noise).
    num_noisy_tiles = int(os.environ.get("CONSISTENCY_NUM_NOISY_TILES", "1"))
    batch = build_interleaved_batch(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        block_size=block_size,
        pad_id=int(pad_id),
        max_pairs=max_pairs,
        generator=gen,
        triple_mode=triple_mode,
        num_noisy_tiles=num_noisy_tiles,
        cascade_drafts=cascade_drafts,
    ).to(device)

    _diag_once = not getattr(compute_consistency_loss, "_diag_dumped", False)
    if _diag_once:
        compute_consistency_loss._diag_dumped = True
        try:
            _vocab = int(getattr(model.config, "vocab_size", -1))
        except Exception:  # noqa: BLE001
            _vocab = -1
        _ii = batch.input_ids
        _pi = batch.position_ids
        print(
            f"[cons-step] batch.input_ids shape={tuple(_ii.shape)} dtype={_ii.dtype} "
            f"device={_ii.device} min={int(_ii.min().item())} max={int(_ii.max().item())} "
            f"vocab={_vocab}",
            flush=True,
        )
        print(
            f"[cons-step] batch.position_ids shape={tuple(_pi.shape)} dtype={_pi.dtype} "
            f"min={int(_pi.min().item())} max={int(_pi.max().item())}",
            flush=True,
        )
        print(
            f"[cons-step] batch.prompt_lens={batch.prompt_lens.tolist()} "
            f"num_pairs={batch.num_pairs.tolist()} block_lens={batch.block_lens.tolist()} "
            f"seq_lens={batch.seq_lens.tolist()}",
            flush=True,
        )

    # Build the SDPA-compatible additive float mask. Dtype matches the
    # model dtype to avoid promotion in attention.
    #
    # CONSISTENCY_CAUSAL_REGION_SIZE env var (default 0 = pure causal noisy block):
    #   When > 0, the first N intra-block positions of each noisy block use
    #   causal attention (AR-equivalent, committable at inference) and the
    #   remaining (block_len - N) positions attend bidirectionally to the
    #   whole noisy block (refinement context only). This mirrors the
    #   TiDAR-style structured mask used in the Decode-Learning inference
    #   path. Inference must use the same `jacobi_causal_region_size` for
    #   the trained model to behave correctly at decode time.
    model_dtype = next(model.parameters()).dtype
    _causal_region = int(os.environ.get("CONSISTENCY_CAUSAL_REGION_SIZE", "0"))
    sdpa_mask = build_sdpa_attention_mask(
        prompt_lens=batch.prompt_lens,
        num_pairs=batch.num_pairs,
        block_lens=batch.block_lens,
        pad_mask=batch.pad_mask,
        device=device,
        dtype=model_dtype,
        triple_mode=triple_mode,
        causal_region_size=_causal_region,
        num_noisy_tiles=num_noisy_tiles,
        # On-policy bridge mode: variable per-pair clean sizes. Pass the
        # per-position role / pair_idx tensors so the mask uses lookup
        # instead of uniform-stride modular arithmetic. None for off-policy.
        role_per_pos=getattr(batch, "role_per_pos", None),
        pair_idx_per_pos=getattr(batch, "pair_idx_per_pos", None),
        noisy_starts=getattr(batch, "noisy_starts", None),
        # v11: canvas pairs get full-bidir intra-tile attention; causal pairs
        # keep causal_region_size behavior. None when canvas mode is off.
        canvas_pairs=getattr(batch, "canvas_pairs", None),
    )
    if _diag_once:
        print(
            f"[cons-step] sdpa_mask shape={tuple(sdpa_mask.shape)} dtype={sdpa_mask.dtype} "
            f"causal_region_size={_causal_region} "
            f"({'hybrid causal+bidir' if _causal_region > 0 else 'pure causal'})",
            flush=True,
        )
        torch.cuda.synchronize()
        print("[cons-step] pre-model-forward sync ok", flush=True)

    # Run the consistency forward with SDPA. AR path is unaffected — we
    # restore the original `_attn_implementation` on exit. Gradient
    # checkpointing must be off here: otherwise backward recomputes the
    # forward after `_attn_implementation` has been restored, sending our
    # 4D float SDPA mask through flash_attention_2 and crashing.

    # Optional: inject a "draft marker" embedding additively on top of
    # noisy-position token embeddings. Reserved Qwen2.5 row 151665 has
    # never seen gradient on any text (no tokenizer entry maps to it), so
    # it acts as a clean dedicated parameter — like a positional embedding
    # for "this position is a draft, discount preceding in-block tokens".
    # Controlled by CONSISTENCY_USE_DRAFT_MARKER (default off so the existing
    # training path is bit-identical without the env var).
    import os as _os
    use_marker = _os.environ.get("CONSISTENCY_USE_DRAFT_MARKER", "0").lower() in {"1", "true", "yes"}
    marker_id = int(_os.environ.get("CONSISTENCY_MARKER_TOKEN_ID", "151665"))
    marker_scale = float(_os.environ.get("CONSISTENCY_MARKER_INIT_SCALE", "1.0"))
    # marker_type: "embed" (default) = learnable row of the embed table (or
    # disk-loaded via marker_embed_override); "sinusoidal" = fixed per-block
    # sinusoidal encoding, no learnable params, no FSDP entanglement.
    marker_type = _os.environ.get("CONSISTENCY_MARKER_TYPE", "embed").lower()

    inputs_embeds = None
    embed_layer = None
    marker_embed = None
    if use_marker:
        # Find the (FSDP-unwrapped) HF model so we can call get_input_embeddings()
        _emb_owner = model
        for _ in range(8):
            inner = getattr(_emb_owner, "module", None)
            if inner is None:
                break
            _emb_owner = inner
        embed_layer = _emb_owner.get_input_embeddings()
        inputs_embeds = embed_layer(batch.input_ids)
        # In triple_mode we ONLY add the marker at the marked-noisy slot, not
        # at unmarked-noisy. In 2-block mode the marker covers all noisy —
        # EXCEPT in v11 canvas mode, where only canvas-pair noisy positions
        # are marked (causal pairs must stay unmarked: graduation at decode
        # time = dropping the marker, and the AR mode never sees it).
        if triple_mode and batch.marked_mask is not None:
            inject_bool = batch.marked_mask
        elif getattr(batch, "canvas_mask", None) is not None:
            inject_bool = batch.canvas_mask
        else:
            inject_bool = batch.noisy_mask
        inject_mask = inject_bool.to(inputs_embeds.dtype).unsqueeze(-1)

        if marker_type == "sinusoidal":
            # Per-block-index fixed sinusoidal encoding. Each noisy position
            # gets the encoding indexed by which block it belongs to. No
            # learnable params, no FSDP entanglement.
            B_, L_, H_ = inputs_embeds.shape
            T_max = int(batch.num_pairs.max().item()) if int(batch.num_pairs.numel()) else 1
            T_max = max(T_max, 1)
            pe_table = _per_block_sinusoidal(T_max, H_, inputs_embeds.device, inputs_embeds.dtype)  # (T_max, H)
            # Compute per-position block index. Block stride depends on layout:
            #   2-block: each block-pair is 2*N tokens after prompt.
            #   3-block: each block-triple is 3*N tokens after prompt.
            N_ = int(batch.block_lens[0].item())
            stride_per_block = (3 if triple_mode else (num_noisy_tiles + 1)) * N_
            P_ = batch.prompt_lens.unsqueeze(-1).to(inputs_embeds.device)  # (B, 1)
            pos_arange = torch.arange(L_, device=inputs_embeds.device).unsqueeze(0).expand(B_, -1)  # (B, L)
            rel = pos_arange - P_
            # Clamp into valid range; we'll mask off non-noisy positions anyway.
            block_idx = torch.clamp(rel.clamp_min(0) // stride_per_block, max=T_max - 1)  # (B, L)
            encoding = pe_table[block_idx]  # (B, L, H)
            inputs_embeds = inputs_embeds + marker_scale * inject_mask * encoding
            if _diag_once:
                print(
                    f"[cons-step] draft marker ENABLED type=sinusoidal scale={marker_scale} "
                    f"T_max={T_max} stride={stride_per_block} H={H_} "
                    f"triple_mode={triple_mode} n_inject_positions={int(inject_mask.sum().item())}",
                    flush=True,
                )
        elif marker_type == "constant":
            # Single fixed marker vector added to every noise token, regardless of
            # block index. Tests whether per-block variation in the sinusoidal
            # marker carried useful information.
            H_ = inputs_embeds.shape[-1]
            marker_vec = _constant_marker(H_, inputs_embeds.device, inputs_embeds.dtype)  # (H,)
            inputs_embeds = inputs_embeds + marker_scale * inject_mask * marker_vec
            if _diag_once:
                print(
                    f"[cons-step] draft marker ENABLED type=constant scale={marker_scale} "
                    f"marker_norm={float(marker_vec.float().norm().item()):.3f} "
                    f"triple_mode={triple_mode} n_inject_positions={int(inject_mask.sum().item())}",
                    flush=True,
                )
        else:
            # Existing embed-table marker (learnable row, or disk-loaded override).
            if marker_embed_override is not None:
                marker_embed = marker_embed_override.to(inputs_embeds.dtype)
            else:
                marker_embed = embed_layer.weight[marker_id].detach().clone().to(inputs_embeds.dtype)
            inputs_embeds = inputs_embeds + marker_scale * inject_mask * marker_embed
            if _diag_once:
                print(
                    f"[cons-step] draft marker ENABLED type=embed id={marker_id} scale={marker_scale} "
                    f"marker_norm={float(marker_embed.float().norm().item()):.3f} "
                    f"triple_mode={triple_mode} n_inject_positions={int(inject_mask.sum().item())} "
                    f"source={'override' if marker_embed_override is not None else 'embed.weight'}",
                    flush=True,
                )

    # Single student forward through the (possibly 3-block) interleaved batch.
    # In triple_mode the marker is injected only at marked-noisy positions
    # (above), so the model produces independent predictions at marked vs
    # unmarked noisy slots via the block-diagonal attention mask.
    with swap_attention_impl(model, "sdpa"), _no_gradient_checkpointing(model) as _n_gc:
        if _diag_once:
            print(f"[cons-step] disabled GC on {_n_gc} submodules; triple_mode={triple_mode}", flush=True)
        if inputs_embeds is not None:
            out = model(
                inputs_embeds=inputs_embeds,
                position_ids=batch.position_ids,
                attention_mask=sdpa_mask,
                use_cache=False,
            )
        else:
            out = model(
                input_ids=batch.input_ids,
                position_ids=batch.position_ids,
                attention_mask=sdpa_mask,
                use_cache=False,
            )
    if _diag_once:
        torch.cuda.synchronize()
        print("[cons-step] post-model-forward sync ok", flush=True)
    logits = out.logits  # (B, Lmax, V)

    if triple_mode:
        b_idx, m_pos, u_pos, l_pos = _identify_block_positions_triple(batch)
        # In triple mode, cons predictor lives at MARKED-noisy positions,
        # anchor target reads from UNMARKED-noisy positions.
        k_pos = m_pos
        pos_in_block = None  # triple_mode doesn't supply per-position offset yet
        pair_idx = None
    else:
        b_idx, k_pos, l_pos, pos_in_block, pair_idx = _identify_block_positions(batch)
        u_pos = None
    if _diag_once:
        torch.cuda.synchronize()
        print(
            f"[cons-step] post-identify triple_mode={triple_mode} n_pos={int(b_idx.numel())} "
            f"k_pos.max={int(k_pos.max().item()) if k_pos.numel() else -1} "
            f"l_pos.max={int(l_pos.max().item()) if l_pos.numel() else -1} "
            f"u_pos.max={int(u_pos.max().item()) if (u_pos is not None and u_pos.numel()) else -1} "
            f"logits.shape={tuple(logits.shape)}",
            flush=True,
        )
    n_pos = int(b_idx.numel())
    if n_pos == 0:
        zero = logits.sum() * 0.0
        return zero, None, {"cons_loss": 0.0, "cons_n_pairs": 0, "cons_n_pos": 0}

    # v11: per-position canvas flag (True = position belongs to a canvas pair).
    # Canvas pairs take a UNIFORM per-position weight in the decay-weighted
    # paths below — the decay weighting is an AR-zone (prefix-gated Jacobi
    # acceptance) concept; on the canvas all positions matter equally.
    canvas_per_pos = None
    canvas_split_terms = None  # populated by the v11.1 split-objective path
    _cv_tbl = getattr(batch, "canvas_pairs", None)
    if _cv_tbl is not None and pair_idx is not None:
        canvas_per_pos = _cv_tbl[b_idx, pair_idx]

    student_noisy = logits[b_idx, k_pos, :]           # (n_pos, V) — cons predictor (marked-noisy in triple mode)
    student_clean = logits[b_idx, l_pos, :]           # (n_pos, V) — clean-position predictions

    # Diagnostic: silent-drift indicator. Forward KL between the student's
    # CLEAN-position predictions and the teacher's CLEAN-position predictions
    # — both at the same positions, no marker effect. As the RL policy
    # diverges from the teacher (especially with a frozen base teacher), this
    # KL grows even when cons_loss/anchor_loss stay flat (those measure noisy
    # positions where teacher predictions are mostly garbage). Computed only
    # with an external teacher; with self-distill the value is ~0 by
    # construction.
    student_teacher_kl_clean = None

    if teacher_model is None:
        # Self-distillation: teacher = student's own clean-view logits.
        # NOTE: when compute_anchor=True with teacher_model=None the anchor
        # term is vacuous (KL(x || x.detach()) ≈ 0). The dual-KL only makes
        # sense with an *external* teacher (EMA, base, frozen snapshot).
        teacher_clean = logits[b_idx, l_pos, :].detach()
    else:
        # External teacher (frozen base OR EMA student): one no_grad forward
        # on the SAME interleaved batch + mask. Take clean-position logits
        # as the target for BOTH cons (at noisy positions) and anchor (at
        # clean positions).
        with torch.no_grad(), swap_attention_impl(teacher_model, "sdpa"):
            t_out = teacher_model(
                input_ids=batch.input_ids,
                position_ids=batch.position_ids,
                attention_mask=sdpa_mask,
                use_cache=False,
            )
        teacher_clean = t_out.logits[b_idx, l_pos, :].detach()
        # Silent-drift diagnostic: forward KL between student's clean-pos
        # logits (these are what RL is shaping) and teacher's clean-pos logits.
        # Computed with no_grad so it adds zero training signal.
        with torch.no_grad():
            _t_logp = F.log_softmax(teacher_clean.float(), dim=-1)
            _s_logp = F.log_softmax(student_clean.detach().float(), dim=-1)
            student_teacher_kl_clean = float((_t_logp.exp() * (_t_logp - _s_logp)).sum(-1).mean().item())

    # Cons loss form selectable via env. Default is the JF/CLLM-style soft
    # divergence against teacher_clean. "ce" switches to Nemotron-style
    # hard cross-entropy against the ground-truth tokens at the *paired*
    # clean positions (i.e. predict the actual next token from the noisy
    # input). This treats CE as KL(δ_{y_true} || p_student) — a fixed,
    # sharp, non-drifting teacher.
    loss_type = _os.environ.get("CONSISTENCY_LOSS_TYPE", "kl").lower()
    if loss_type == "ce":
        # SHIFT-BY-1 fix (2026-05-30): standard AR convention says logit at
        # position p predicts the token at position p+1. Our layout shares
        # RoPE positions between noisy and clean blocks, so the noisy logit
        # at offset i predicts the clean token at offset i+1 (NOT offset i,
        # which was the previous off-by-one bug that taught the model to use
        # the wrong prediction head).
        if pos_in_block is None:
            gt_tokens = batch.input_ids[b_idx, l_pos]
            cons_loss = F.cross_entropy(student_noisy.float(), gt_tokens, reduction="mean")
        else:
            Nb_loss = int(batch.block_lens[0].item())
            next_pos = l_pos + 1
            valid = (pos_in_block < (Nb_loss - 1))
            if valid.any():
                # Also require the next clean token to be a valid (non-pad) response token.
                np_clamped = torch.clamp(next_pos, max=batch.pad_mask.shape[-1] - 1)
                valid = valid & batch.pad_mask[b_idx, np_clamped]
            if valid.any():
                b_v, k_v, n_v = b_idx[valid], k_pos[valid], next_pos[valid]
                gt_tokens = batch.input_ids[b_v, n_v]
                cons_loss = F.cross_entropy(logits[b_v, k_v, :].float(), gt_tokens, reduction="mean")
            else:
                cons_loss = logits.sum() * 0.0
    elif loss_type == "dflash_ce":
        # dFlash-style CE on noisy-position predictions:
        #   - per-position CE against the NEXT ground-truth rollout token
        #     (shift-by-1, standard AR convention)
        #   - position-decay weight w_k = exp(-(k-1)/gamma), k=1..N intra-block
        #     (gamma=12 default for block_size=32 — extrapolated from paper)
        # Pairs naturally with CONSISTENCY_NOISE_SOURCE=mask + bidir attention.
        #
        # On-policy mode: when `batch.onpolicy_prefix_lens` is provided, each
        # pair's first `prefix_match_len` positions are CASCADE-CONVERGED
        # (= committed spec tokens, draft==response by construction). Mask those
        # out and shift the decay weight to start at position `prefix_match_len`
        # so the FIRST genuinely-noisy position gets weight 1.
        if pos_in_block is None:
            # 3-block / triple_mode fallback: uniform-weight CE w/o shift.
            gt_tokens = batch.input_ids[b_idx, l_pos]
            cons_loss = F.cross_entropy(student_noisy.float(), gt_tokens, reduction="mean")
        elif _os.environ.get("CONSISTENCY_MASK_TAIL", "0").lower() in {"1", "true", "yes"}:
            # Idea A: mask-tail target. Clean CE on [pml, pml+margin), mask-token
            # CE on [pml+margin, N). Flat-weighted (decay dropped by design).
            if canvas_per_pos is not None and bool(canvas_per_pos.any()):
                raise ValueError(
                    "CONSISTENCY_MASK_TAIL is incompatible with canvas pairs "
                    "(margin/mask-token targets are an AR-zone concept)."
                )
            _mask_id = int(_os.environ.get("CONSISTENCY_MASK_TOKEN_ID", "151643"))
            _margin = int(_os.environ.get("CONSISTENCY_MASK_MARGIN", "8"))
            cons_loss, _n_clean, _n_mask = _mask_tail_dflash_loss(
                logits, batch, b_idx, k_pos, l_pos, pos_in_block, pair_idx,
                _mask_id, _margin,
            )
        else:
            Nb_loss = int(batch.block_lens[0].item())
            next_pos = l_pos + 1
            # Drop pos_in_block == N-1 (no in-block "next") and pad-target
            valid = (pos_in_block < (Nb_loss - 1))
            if valid.any():
                np_clamped = torch.clamp(next_pos, max=batch.pad_mask.shape[-1] - 1)
                valid = valid & batch.pad_mask[b_idx, np_clamped]
            # On-policy: filter out positions inside the cascade-converged prefix.
            prefix_per_pos = None
            opl = getattr(batch, "onpolicy_prefix_lens", None)
            if opl is not None and pair_idx is not None:
                prefix_per_pos = opl[b_idx, pair_idx]  # (n_pos,) long
                valid = valid & (pos_in_block >= prefix_per_pos)
            if valid.any():
                b_v = b_idx[valid]
                k_v = k_pos[valid]
                n_v = next_pos[valid]
                pos_v = pos_in_block[valid]
                gt_tokens = batch.input_ids[b_v, n_v]
                student_noisy_v = logits[b_v, k_v, :]
                gamma = float(_os.environ.get("CONSISTENCY_DFLASH_GAMMA", "12.0"))
                student_noisy_v_f = student_noisy_v.float()
                per_pos_losses = F.cross_entropy(
                    student_noisy_v_f, gt_tokens, reduction="none",
                )
                # Decay: shift origin so first NOISY position (= position
                # prefix_match_len) has weight 1. Off-policy: prefix_per_pos
                # is None or zeros → no shift (legacy behavior).
                if prefix_per_pos is not None:
                    pref_v = prefix_per_pos[valid].to(per_pos_losses.dtype)
                    shifted = (pos_v.to(per_pos_losses.dtype) - pref_v).clamp_min(0.0)
                else:
                    shifted = pos_v.to(per_pos_losses.dtype)
                w = torch.exp(-shifted / gamma)
                # v11 canvas pairs: uniform weight (no position decay).
                if canvas_per_pos is not None:
                    w = torch.where(canvas_per_pos[valid], torch.ones_like(w), w)

                # Focal weighting: stack on top of dflash decay.
                # `(1 - p_correct)^focal_gamma` upweights confidently-wrong
                # predictions. focal_gamma=0 disables (recovers vanilla dflash_ce).
                # focal_gamma=2 is the typical focal-loss default.
                focal_gamma = float(_os.environ.get("CONSISTENCY_FOCAL_GAMMA", "0.0"))
                if focal_gamma > 0.0:
                    with torch.no_grad():
                        # p_correct[i] = softmax(logits[i])[gt[i]]
                        log_p = F.log_softmax(student_noisy_v_f, dim=-1)
                        p_correct = log_p.gather(-1, gt_tokens.unsqueeze(-1)).squeeze(-1).exp()
                        focal_floor = float(_os.environ.get("CONSISTENCY_FOCAL_MIN_WEIGHT", "0.0"))
                        focal_w = (1.0 - p_correct).clamp_min(0.0).pow(focal_gamma)
                        focal_w = focal_w.clamp_min(focal_floor)
                    w = w * focal_w
                cons_loss = (per_pos_losses * w).sum() / w.sum().clamp_min(1e-12)
            else:
                cons_loss = logits.sum() * 0.0
    else:
        kl_decay = _os.environ.get("CONSISTENCY_KL_DECAY", "0").lower() in {"1", "true", "yes"}
        # v11.1: split-objective mode. CONSISTENCY_CANVAS_LOSS=ce routes canvas
        # positions to uniform shift-by-1 CE against the CLEAN ROLLOUT TOKENS
        # (the decode-aligned target — the AR zone verifies the policy's own
        # greedy text, so KL-to-base would cap canvas accuracy at base↔policy
        # agreement). Causal positions keep the v9 KL(+decay)-to-teacher path
        # untouched. CONSISTENCY_CANVAS_WEIGHT_MULT scales ONLY the canvas
        # term (the new capability needs more than λ's ~1% gradient share;
        # the causal preservation pressure stays at the proven v9 level).
        canvas_loss_mode = _os.environ.get("CONSISTENCY_CANVAS_LOSS", "").lower()
        canvas_mult = float(_os.environ.get("CONSISTENCY_CANVAS_WEIGHT_MULT", "1.0"))
        if (canvas_loss_mode == "ce" and canvas_per_pos is not None
                and pos_in_block is not None and bool(canvas_per_pos.any())):
            cv = canvas_per_pos
            ncv = ~cv
            if bool(ncv.any()):
                per_pos_c = soft_cross_entropy(
                    student_noisy[ncv], teacher_clean[ncv], T_soft=T_soft,
                    divergence=divergence, reduction="none",
                ) * (T_soft * T_soft)
                if kl_decay:
                    gamma = float(_os.environ.get("CONSISTENCY_DFLASH_GAMMA", "12.0"))
                    opl = getattr(batch, "onpolicy_prefix_lens", None)
                    if opl is not None and pair_idx is not None:
                        pref_c = opl[b_idx, pair_idx][ncv].to(per_pos_c.dtype)
                    else:
                        pref_c = torch.zeros_like(per_pos_c)
                    shifted_c = pos_in_block[ncv].to(per_pos_c.dtype) - pref_c
                    w_c = torch.exp(-shifted_c.clamp_min(0.0) / gamma) * (shifted_c >= 0).to(per_pos_c.dtype)
                    causal_term = (per_pos_c * w_c).sum() / w_c.sum().clamp_min(1e-12)
                else:
                    causal_term = per_pos_c.mean()
            else:
                causal_term = logits.sum() * 0.0
            # Canvas half: uniform-weight shift-by-1 CE to the next clean
            # rollout token (same convention as the dflash_ce path).
            Nb_cv = int(batch.block_lens[0].item())
            next_pos_cv = l_pos + 1
            valid_cv = cv & (pos_in_block < (Nb_cv - 1))
            np_cl = torch.clamp(next_pos_cv, max=batch.pad_mask.shape[-1] - 1)
            valid_cv = valid_cv & batch.pad_mask[b_idx, np_cl]
            if bool(valid_cv.any()):
                b_v = b_idx[valid_cv]
                k_v = k_pos[valid_cv]
                n_v = next_pos_cv[valid_cv]
                canvas_term = F.cross_entropy(
                    logits[b_v, k_v, :].float(), batch.input_ids[b_v, n_v],
                    reduction="mean",
                )
            else:
                canvas_term = logits.sum() * 0.0
            cons_loss = causal_term + canvas_mult * canvas_term
            canvas_split_terms = (
                float(causal_term.detach().item()),
                float(canvas_term.detach().item()),
                canvas_mult,
            )
        elif kl_decay and pos_in_block is not None:
            # dFlash-style position decay on the soft (KL) loss, mirroring the
            # dflash_ce path: (a) mask out cascade-converged prefix positions
            # (pos < pml — their KL is trivially small and dilutes the loss),
            # (b) weight w = exp(-(pos - pml)/gamma) so the first genuinely
            # noisy position carries weight 1 — matching the prefix-gated
            # structure of Jacobi acceptance (position j only matters if
            # 0..j-1 all accepted).
            per_pos = soft_cross_entropy(
                student_noisy, teacher_clean, T_soft=T_soft,
                divergence=divergence, reduction="none",
            ) * (T_soft * T_soft)
            gamma = float(_os.environ.get("CONSISTENCY_DFLASH_GAMMA", "12.0"))
            opl = getattr(batch, "onpolicy_prefix_lens", None)
            if opl is not None and pair_idx is not None:
                pref = opl[b_idx, pair_idx].to(per_pos.dtype)
            else:
                pref = torch.zeros_like(per_pos)
            shifted = pos_in_block.to(per_pos.dtype) - pref
            w = torch.exp(-shifted.clamp_min(0.0) / gamma) * (shifted >= 0).to(per_pos.dtype)
            # v11 canvas pairs: uniform weight (no decay, no prefix gating —
            # canvas pml is zeroed in pack.py; all positions matter equally).
            if canvas_per_pos is not None:
                w = torch.where(canvas_per_pos, torch.ones_like(w), w)
            cons_loss = (per_pos * w).sum() / w.sum().clamp_min(1e-12)
        else:
            cons_loss = soft_cross_entropy(student_noisy, teacher_clean, T_soft=T_soft,
                                           divergence=divergence) * (T_soft * T_soft)

    anchor_loss = None
    if compute_anchor:
        if anchor_mode == "clean":
            # Anchor at CLEAN positions: pull student's clean-position predictions
            # toward teacher's clean-position predictions. Resists drift of AR-mode
            # behavior on tokens we'd actually predict at inference.
            anchor_loss = soft_cross_entropy(student_clean, teacher_clean, T_soft=T_soft,
                                             divergence=divergence) * (T_soft * T_soft)
        elif anchor_mode == "noisy_unmarked":
            # Anchor at UNMARKED-NOISY positions (separate slot in the 3-block
            # layout, sharing RoPE positions with the marked-noisy slot but
            # masked off from it). Target = teacher's prediction at those
            # same positions (teacher sees no marker either way).
            # The marker becomes an explicit gating signal: with marker ->
            # cons predictor pulls toward clean target; without marker ->
            # the model's prediction stays near the teacher's noisy prediction.
            if not triple_mode or u_pos is None:
                # No 3-block layout active -> degenerate; anchor at clean (same as
                # anchor_mode="clean") to avoid silent vacuous KL.
                if teacher_model is None:
                    teacher_clean_target = logits[b_idx, l_pos, :].detach()
                else:
                    teacher_clean_target = t_out.logits[b_idx, l_pos, :].detach()
                anchor_loss = soft_cross_entropy(student_clean, teacher_clean_target,
                                                 T_soft=T_soft, divergence=divergence) * (T_soft * T_soft)
            else:
                student_unmarked_noisy = logits[b_idx, u_pos, :]
                if teacher_model is None:
                    teacher_unmarked_noisy = logits[b_idx, u_pos, :].detach()
                else:
                    teacher_unmarked_noisy = t_out.logits[b_idx, u_pos, :].detach()
                anchor_loss = soft_cross_entropy(student_unmarked_noisy, teacher_unmarked_noisy,
                                                 T_soft=T_soft, divergence=divergence) * (T_soft * T_soft)
        else:
            raise ValueError(f"Unknown anchor_mode: {anchor_mode!r}; expected 'clean' or 'noisy_unmarked'.")

    metrics = {
        "cons_loss": float(cons_loss.detach().item()),
        "cons_n_pairs": int(batch.num_pairs.sum().item()),
        "cons_n_pos": n_pos,
    }
    if canvas_per_pos is not None:
        metrics["cons_canvas_pos_frac"] = float(canvas_per_pos.float().mean().item())
    if canvas_split_terms is not None:
        metrics["cons_causal_term"] = canvas_split_terms[0]
        metrics["cons_canvas_term"] = canvas_split_terms[1]
        metrics["cons_canvas_weight_mult"] = canvas_split_terms[2]
    # Argmax-on-noisy metric: does student.argmax at the noisy position match
    # the NEXT clean response token? This is what actually determines Jacobi
    # acceptance (TPF), independent of which loss variant (CE/KL/forward/reverse)
    # we're using. Tracked separately from cons_loss because CE log_p
    # improvement can decouple from argmax flips (LK-losses paper finding).
    if pos_in_block is not None and n_pos > 0:
        with torch.no_grad():
            Nb_m = int(batch.block_lens[0].item())
            valid_m = (pos_in_block < (Nb_m - 1))
            if valid_m.any():
                next_pos_m = l_pos + 1
                np_clamped_m = torch.clamp(next_pos_m, max=batch.pad_mask.shape[-1] - 1)
                valid_m = valid_m & batch.pad_mask[b_idx, np_clamped_m]
            opl_m = getattr(batch, "onpolicy_prefix_lens", None)
            if opl_m is not None and pair_idx is not None:
                prefix_m = opl_m[b_idx, pair_idx]
                valid_m = valid_m & (pos_in_block >= prefix_m)
            if valid_m.any():
                b_vm = b_idx[valid_m]
                k_vm = k_pos[valid_m]
                n_vm = (l_pos + 1)[valid_m]
                gt_m = batch.input_ids[b_vm, n_vm]
                pred_m = logits[b_vm, k_vm, :].argmax(dim=-1)
                correct_vec = (pred_m == gt_m).float()
                metrics["cons_argmax_correct"] = float(correct_vec.mean().item())
                metrics["cons_argmax_n_valid"] = int(valid_m.sum().item())
                # v11: split by pair mode — canvas argmax_correct is the
                # canvas-training progress signal; causal must hold v9 levels.
                if canvas_per_pos is not None:
                    cv_m = canvas_per_pos[valid_m]
                    if bool(cv_m.any()):
                        metrics["cons_argmax_correct_canvas"] = float(correct_vec[cv_m].mean().item())
                        metrics["cons_argmax_n_canvas"] = int(cv_m.sum().item())
                    if bool((~cv_m).any()):
                        metrics["cons_argmax_correct_causal"] = float(correct_vec[~cv_m].mean().item())
                        metrics["cons_argmax_n_causal"] = int((~cv_m).sum().item())
    if anchor_loss is not None:
        metrics["anchor_loss"] = float(anchor_loss.detach().item())
    if student_teacher_kl_clean is not None:
        metrics["student_teacher_kl_clean"] = student_teacher_kl_clean
    return cons_loss, anchor_loss, metrics


if __name__ == "__main__":
    # Smoke test the soft CE; the full forward smoke is in scripts/test_consistency.py.
    s = torch.randn(64, 1000, requires_grad=True)
    t = torch.randn(64, 1000)
    L = soft_cross_entropy(s, t)
    L.backward()
    print(f"soft_cross_entropy smoke: L={L.item():.4f}  s.grad.norm={s.grad.norm().item():.4f}")
