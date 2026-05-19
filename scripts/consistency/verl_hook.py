"""Integration hook for verl's actor `forward_step`.

Call `maybe_add_consistency_loss(self, micro_batch, loss)` from inside
`FSDPEngineWithLMHead.forward_step` right after `loss` is computed, e.g.

    loss, output = self._original_forward_step(...)
    loss = maybe_add_consistency_loss(self, micro_batch, loss)

The standard AR forward path is untouched. The consistency forward is a
SECOND forward call under a flex_attention context manager.

Activation knobs read from `self.config.consistency` (Hydra subtree) or
from the `CONSISTENCY_<NAME>` environment variable:
  enable:           bool       — default False (no-op)
  weight:           float      — λ multiplier (default 1e-2)
  block_size:       int        — N (default 32, matches JF Coder)
  T_soft:           float      — Hinton temperature (default 1.0)
  fraction:         float      — fraction of the micro-batch to use
                                  for consistency (default 0.25, to keep
                                  the extra forward cost <= 25% per step)
  max_pairs:        int|None   — cap T per sample (None = no cap)
  pad_id:           int        — tokenizer pad id

Optional weight scheduling (linear interpolation over training progress):
  schedule:         str        — "constant" (default), "warmup_in", "warmup_out"
                                  warmup_in:  weight 0 → `weight` over ramp window
                                  warmup_out: weight `weight` → 0 over ramp window
  weight_final:     float|None — overrides the schedule preset. If set,
                                  weight goes from `weight` to `weight_final`
                                  over the ramp window regardless of `schedule`.
  ramp_start_frac:  float      — start of ramp as fraction of total_steps (default 0.0)
  ramp_end_frac:    float      — end of ramp as fraction of total_steps (default 1.0)
  total_steps:      int        — total training steps for the schedule (default 300)

The effective weight is logged to `actor/cons_weight_effective` each step.

The micro_batch must provide the raw (un-rolled-shifted) input_ids and a
way to identify (prompt, response) boundaries. We use:
  - micro_batch["input_ids"]  : nested or padded tensor
  - micro_batch["prompts"]    : (B, max_prompt_len) — produced by rollout
  - micro_batch["responses"]  : (B, max_resp_len)   — produced by rollout
"""

from __future__ import annotations

import os
import sys
from typing import Any

import torch

# Make the consistency package importable when this file is sourced from verl.
_SELF_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.dirname(_SELF_DIR)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from consistency.loss import compute_consistency_loss  # noqa: E402


_DIAG_DUMPED = False  # one-shot diagnostic guard (per worker process)
_TEACHER_MODEL = None  # frozen base teacher (lazy-loaded, per worker process)


def _get_base_teacher(student_model, teacher_path: str):
    """Load (once) a frozen HF model from `teacher_path` onto the same device
    as the student. Returned model is in eval mode with requires_grad=False.

    Cached in module-level `_TEACHER_MODEL` — one copy per worker process,
    so each FSDP rank holds its own full (not sharded) copy. For 7B bf16
    that's ~14 GB per H200 rank, which is fine.
    """
    global _TEACHER_MODEL
    if _TEACHER_MODEL is not None:
        return _TEACHER_MODEL
    from transformers import AutoModelForCausalLM  # local import to avoid cost when disabled
    # Find an actual parameter on the (FSDP-wrapped) student to read device/dtype.
    p = next(student_model.parameters())
    device = p.device
    dtype = p.dtype
    print(
        f"[cons-teacher] loading base teacher from {teacher_path!r} "
        f"-> device={device} dtype={dtype}",
        flush=True,
    )
    m = AutoModelForCausalLM.from_pretrained(teacher_path, torch_dtype=dtype)
    m.eval()
    for prm in m.parameters():
        prm.requires_grad = False
    m.to(device)
    _TEACHER_MODEL = m
    print(f"[cons-teacher] base teacher loaded; params={sum(p.numel() for p in m.parameters())/1e9:.2f} B", flush=True)
    return _TEACHER_MODEL


def _dump_micro_batch_structure(micro_batch) -> None:
    """One-shot dump of every micro_batch key's type/shape/dtype.

    Runs only on the first consistency call so logs stay small. Goal:
    confirm whether `prompts`/`responses`/`attention_mask` are dense
    (B, L) or nested/jagged under `use_remove_padding=True` +
    `use_dynamic_bsz=True` (pad_mode=NO_PADDING), which would make the
    naive `attn[:, :Lp]` slicing wrong and produce OOB token ids.
    """
    global _DIAG_DUMPED
    if _DIAG_DUMPED:
        return
    _DIAG_DUMPED = True
    print("[consistency-diag] micro_batch keys + shapes:", flush=True)
    try:
        keys = list(micro_batch.keys())
    except Exception as e:  # noqa: BLE001
        print(f"[consistency-diag] cannot list keys: {e}", flush=True)
        return
    for k in keys:
        try:
            v = micro_batch[k]
        except Exception as e:  # noqa: BLE001
            print(f"[consistency-diag]  {k!r}: <get failed: {e}>", flush=True)
            continue
        if isinstance(v, torch.Tensor):
            is_nested = getattr(v, "is_nested", False)
            try:
                shape = tuple(v.shape) if not is_nested else "<nested>"
            except Exception:  # noqa: BLE001
                shape = "<shape failed>"
            extra = ""
            if isinstance(v, torch.Tensor) and v.dtype in (torch.long, torch.int32, torch.int64):
                try:
                    flat = v.values() if is_nested else v.flatten()
                    extra = f" min={int(flat.min().item())} max={int(flat.max().item())} numel={int(flat.numel())}"
                except Exception as e:  # noqa: BLE001
                    extra = f" <stats failed: {e}>"
            print(
                f"[consistency-diag]  {k!r}: Tensor shape={shape} dtype={v.dtype} "
                f"device={v.device} is_nested={is_nested}{extra}",
                flush=True,
            )
        else:
            print(f"[consistency-diag]  {k!r}: type={type(v).__name__} value={v!r}"[:200], flush=True)


def _extract_prompts_responses(micro_batch) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Recover the (prompt_ids[i], response_ids[i]) lists from verl's batch.

    verl's rollout produces:
      - "prompts": (B, max_prompt_len)   — left-padded, pad_token_id
      - "responses": (B, max_response_len) — right-padded, pad_token_id
      - "attention_mask": (B, prompt+resp_len) with True = valid

    We strip padding using attention_mask to recover real sequences.
    """
    _dump_micro_batch_structure(micro_batch)

    prompts = micro_batch["prompts"]
    responses = micro_batch["responses"]
    attn = micro_batch["attention_mask"]

    # Defensive: nested tensors break the slice math below. If any of the
    # three inputs is nested, bail with a clear message so we can rewrite
    # the extraction for the actual layout.
    for nm, t in (("prompts", prompts), ("responses", responses), ("attention_mask", attn)):
        if isinstance(t, torch.Tensor) and getattr(t, "is_nested", False):
            raise RuntimeError(
                f"consistency hook: micro_batch[{nm!r}] is a nested tensor "
                f"(shape={t.shape}); the current extractor assumes dense "
                f"(B, L). Inspect the [consistency-diag] dump above and "
                f"rewrite _extract_prompts_responses for nested layout."
            )

    Lp = prompts.shape[-1]
    Lr = responses.shape[-1]
    p_mask = attn[:, :Lp].bool()
    r_mask = attn[:, Lp : Lp + Lr].bool()
    prompt_list, resp_list = [], []
    for i in range(prompts.shape[0]):
        prompt_list.append(prompts[i][p_mask[i]].long().cpu())
        resp_list.append(responses[i][r_mask[i]].long().cpu())
    return prompt_list, resp_list


def maybe_add_consistency_loss(
    engine,
    micro_batch,
    loss: torch.Tensor,
    metrics: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Add consistency loss to `loss` if enabled in config. Returns the new
    loss tensor (or the original loss if disabled). Updates `metrics` in
    place if provided.
    """
    # Resolution order: explicit attribute > engine.config.consistency > env.
    # Env-based override lets us toggle via the run script without touching
    # verl's Hydra schema. Set CONSISTENCY_ENABLE=1 to activate.
    cfg = getattr(engine, "_consistency_config", None)
    if cfg is None:
        engine_cfg = getattr(engine, "config", None)
        cfg = getattr(engine_cfg, "consistency", None) if engine_cfg else None

    def _read(name: str, default, cast):
        if cfg is not None and hasattr(cfg, name):
            return cast(getattr(cfg, name))
        env = os.environ.get(f"CONSISTENCY_{name.upper()}")
        return cast(env) if env is not None else default

    enabled = _read("enable", False, lambda v: str(v).lower() in {"1", "true", "yes"})
    if not enabled:
        return loss

    weight_init = _read("weight", 1e-2, float)
    block_size = _read("block_size", 32, int)
    T_soft = _read("T_soft", 1.0, float)
    fraction = _read("fraction", 0.25, float)
    max_pairs = _read("max_pairs", None, lambda v: None if v in (None, "", "None") else int(v))
    pad_id = _read("pad_id", 0, int)
    divergence = _read("divergence", "forward_kl", lambda v: str(v).lower())

    # Optional scheduling. Schedule shape is always linear interpolation
    # between weight_init and weight_final over [ramp_start_frac, ramp_end_frac]
    # of total_steps. Before ramp_start, weight=weight_init; after ramp_end,
    # weight=weight_final.
    #
    # Convenience knobs (CONSISTENCY_SCHEDULE) auto-set init/final for the
    # two named patterns, only if WEIGHT_FINAL is not explicitly set:
    #   "constant"   (default) — weight_final defaults to weight_init.
    #   "warmup_in"  — weight 0 → weight_init over the ramp window.
    #                  (specify weight_init = the target/peak weight.)
    #   "warmup_out" — weight weight_init → 0 over the ramp window.
    #                  (specify weight_init = the starting/initial weight.)
    schedule = _read("schedule", "constant", str).lower()
    weight_final_env = _read("weight_final", None,
                              lambda v: None if v in (None, "", "None") else float(v))
    ramp_start_frac = _read("ramp_start_frac", 0.0, float)
    ramp_end_frac = _read("ramp_end_frac", 1.0, float)
    total_steps = _read("total_steps", 300, int)

    if weight_final_env is not None:
        weight_final = weight_final_env
        # WEIGHT_FINAL overrides the schedule preset.
        actual_init = weight_init
    elif schedule == "warmup_in":
        actual_init = 0.0
        weight_final = weight_init
    elif schedule == "warmup_out":
        actual_init = weight_init
        weight_final = 0.0
    else:  # constant
        actual_init = weight_init
        weight_final = weight_init

    # Current step: prefer engine.global_step, fall back to micro_batch.
    current_step = None
    for attr in ("global_step", "_global_step"):
        if hasattr(engine, attr):
            try:
                current_step = int(getattr(engine, attr))
                break
            except Exception:  # noqa: BLE001
                pass
    if current_step is None:
        try:
            gs = micro_batch["global_steps"]
            if hasattr(gs, "__getitem__"):
                current_step = int(gs[0])
        except Exception:  # noqa: BLE001
            current_step = 0

    if total_steps <= 0 or actual_init == weight_final:
        weight = actual_init
    else:
        progress = float(current_step) / float(total_steps)
        if progress <= ramp_start_frac:
            weight = actual_init
        elif progress >= ramp_end_frac:
            weight = weight_final
        else:
            span = max(1e-12, ramp_end_frac - ramp_start_frac)
            t = (progress - ramp_start_frac) / span
            weight = actual_init + (weight_final - actual_init) * t

    prompt_list, response_list = _extract_prompts_responses(micro_batch)
    B = len(prompt_list)
    if B == 0:
        return loss

    # Validate extracted token ids against the model's vocab. An OOB id
    # here is what crashed previous runs inside the embedding gather
    # (vectorized_gather_kernel index out of bounds, ~1440 hits).
    try:
        _vocab = int(engine.module.config.vocab_size)
    except Exception:  # noqa: BLE001
        _vocab = None
    if _vocab is not None:
        for nm, lst in (("prompt", prompt_list), ("response", response_list)):
            for i, t in enumerate(lst):
                if t.numel() == 0:
                    continue
                mn = int(t.min().item())
                mx = int(t.max().item())
                if mn < 0 or mx >= _vocab:
                    raise RuntimeError(
                        f"consistency hook: extracted {nm}_ids[{i}] has OOB "
                        f"token id (min={mn}, max={mx}, vocab={_vocab}). "
                        f"len={t.numel()} first16={t[:16].tolist()}. This "
                        f"means _extract_prompts_responses produced garbage "
                        f"from the micro_batch — see [consistency-diag] dump."
                    )

    n_use = max(1, int(round(B * fraction)))
    # Random subset for cost control. Use the trainer's global step as seed
    # so all DP ranks pick the same indices (simple sync).
    seed = int(getattr(engine, "_global_step", 0)) if hasattr(engine, "_global_step") else 0
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(B, generator=g)[:n_use].tolist()
    pl = [prompt_list[i] for i in idx]
    rl = [response_list[i] for i in idx]

    # Optional frozen teacher (pre-RL base). When CONSISTENCY_TEACHER=base
    # the cons loss target switches from the live student's clean-view logits
    # to a frozen base model's clean-view logits. Decouples the cons target
    # from the RL-evolving policy, preventing reinforcement of mode collapse.
    teacher_mode = _read("teacher", "self", lambda v: str(v).lower())
    teacher_path = _read("teacher_path", "", str)
    teacher_model = None
    if teacher_mode == "base":
        if not teacher_path:
            raise RuntimeError(
                "CONSISTENCY_TEACHER=base requires CONSISTENCY_TEACHER_PATH "
                "to be set to the HF model dir of the frozen base teacher."
            )
        teacher_model = _get_base_teacher(engine.module, teacher_path)

    cons_loss, cons_metrics = compute_consistency_loss(
        model=engine.module,
        prompt_ids=pl,
        response_ids=rl,
        block_size=block_size,
        pad_id=pad_id,
        max_pairs=max_pairs,
        T_soft=T_soft,
        seed=seed,
        divergence=divergence,
        teacher_model=teacher_model,
    )
    if metrics is not None:
        metrics["actor/cons_loss"] = float(cons_loss.detach().item())
        metrics["actor/cons_weight_effective"] = float(weight)
        for k, v in cons_metrics.items():
            metrics[f"actor/{k}"] = v

    return loss + weight * cons_loss
