"""Consistency loss orchestrator: runs ONE extra forward pass on the
interleaved batch under flex_attention, then computes soft cross-entropy
between student (noisy block) and teacher (clean block, detached) logits.

This is a *separate* forward call from verl's standard AR pass — the AR
training logic is untouched.
"""

from __future__ import annotations

import math

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
                       divergence: str = "forward_kl") -> torch.Tensor:
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
        return -(p * log_q).sum(dim=-1).mean()
    elif divergence == "reverse_kl":
        # sum( softmax(s) * (log_softmax(s) - log_softmax(t).detach()) )
        q = log_q.exp()
        return (q * (log_q - log_p)).sum(dim=-1).mean()
    elif divergence == "jsd":
        # m = (q + p) / 2  (mixture; in log space: logsumexp([log_q, log_p]) - log 2)
        import math
        log_m = torch.logsumexp(torch.stack([log_q, log_p], dim=0), dim=0) - math.log(2.0)
        q = log_q.exp()
        p = log_p.exp()
        kl_qm = (q * (log_q - log_m)).sum(dim=-1).mean()
        kl_pm = (p * (log_p - log_m)).sum(dim=-1).mean()
        return 0.5 * (kl_qm + kl_pm)
    else:
        raise ValueError(f"Unknown divergence: {divergence!r}")


@torch.no_grad()
def _identify_block_positions(
    batch: InterleavedBatch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """For each sample b, return (b_idx, k_pos, l_pos): the (sample, position)
    pairs of noisy and clean blocks aligned pair-by-pair. Used in 2-block mode.

    Output shapes: each is a flat int64 tensor of length sum_b T_b * N_b.
    """
    B = batch.input_ids.shape[0]
    device = batch.input_ids.device
    b_list, k_list, l_list = [], [], []
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
            ks = Pb + 2 * j * Nb
            ls = Pb + (2 * j + 1) * Nb
            offs = torch.arange(Nb, dtype=torch.long, device=device)
            keep = pad[b, ks : ks + Nb] & pad[b, ls : ls + Nb]
            if not keep.any():
                continue
            kept_offs = offs[keep]
            b_list.append(torch.full((kept_offs.numel(),), b, dtype=torch.long, device=device))
            k_list.append(ks + kept_offs)
            l_list.append(ls + kept_offs)
    if not b_list:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty, empty
    return torch.cat(b_list), torch.cat(k_list), torch.cat(l_list)


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
    batch = build_interleaved_batch(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        block_size=block_size,
        pad_id=int(pad_id),
        max_pairs=max_pairs,
        generator=gen,
        triple_mode=triple_mode,
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
    model_dtype = next(model.parameters()).dtype
    sdpa_mask = build_sdpa_attention_mask(
        prompt_lens=batch.prompt_lens,
        num_pairs=batch.num_pairs,
        block_lens=batch.block_lens,
        pad_mask=batch.pad_mask,
        device=device,
        dtype=model_dtype,
        triple_mode=triple_mode,
    )
    if _diag_once:
        print(
            f"[cons-step] sdpa_mask shape={tuple(sdpa_mask.shape)} dtype={sdpa_mask.dtype}",
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
        # at unmarked-noisy. In 2-block mode the marker covers all noisy.
        if triple_mode and batch.marked_mask is not None:
            inject_mask = batch.marked_mask.to(inputs_embeds.dtype).unsqueeze(-1)
        else:
            inject_mask = batch.noisy_mask.to(inputs_embeds.dtype).unsqueeze(-1)

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
            stride_per_block = (3 if triple_mode else 2) * N_
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
    else:
        b_idx, k_pos, l_pos = _identify_block_positions(batch)
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

    student_noisy = logits[b_idx, k_pos, :]           # (n_pos, V) — cons predictor (marked-noisy in triple mode)
    student_clean = logits[b_idx, l_pos, :]           # (n_pos, V) — clean-position predictions

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
    if anchor_loss is not None:
        metrics["anchor_loss"] = float(anchor_loss.detach().item())
    return cons_loss, anchor_loss, metrics


if __name__ == "__main__":
    # Smoke test the soft CE; the full forward smoke is in scripts/test_consistency.py.
    s = torch.randn(64, 1000, requires_grad=True)
    t = torch.randn(64, 1000)
    L = soft_cross_entropy(s, t)
    L.backward()
    print(f"soft_cross_entropy smoke: L={L.item():.4f}  s.grad.norm={s.grad.norm().item():.4f}")
