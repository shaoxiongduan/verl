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
_TEACHER_MODEL = None  # frozen base teacher OR EMA teacher (lazy-loaded, per worker process)
_TEACHER_MODE = None   # "base" or "ema"; tracks which type of teacher is loaded
_EMA_STEP_COUNT = 0    # legacy counter (no longer used by EMA update guard)
_LAST_EMA_UPDATE_STEP = -1  # only update EMA when current_step crosses to a new value
_MARKER_EMBED = None   # marker row loaded from disk (per worker process)

# Adaptive cons-weight scaling state. Tracks the most-recent |loss| (the
# pre-cons verl loss = PG surrogate + entropy + KL penalty) and rescales
# the cons weight so that, as pg_loss collapses to ~0 mid-training, cons
# collapses proportionally rather than continuing to drive updates on its own.
# Uses the LAST microbatch's |loss| directly — no EMA smoothing — so the
# scale tracks instantaneous PG magnitude per step. Per-rank state; we accept
# rank-local drift since gradients are FSDP-averaged across ranks anyway.
_PG_LOSS_LAST = None        # float | None — last observed |loss| this rank
_PG_LOSS_INITIAL = None     # float | None — locked reference after warmup
_ADAPTIVE_WARMUP_COUNT = 0  # number of microbatches seen so far (warmup gate)

# Per-step accumulator for target_ratio. Within a step we accumulate the
# signed sum of per-microbatch pg loss (matches verl's actor/pg_loss SUM
# aggregation) and the per-microbatch sum of normalized cons loss. At each
# step boundary we either (a) snapshot the totals from the just-completed
# step (CONSISTENCY_TARGET_EMA_DECAY=0, 1-step lookback) or (b) blend them
# into a running EMA (CONSISTENCY_TARGET_EMA_DECAY > 0).
#
# The EMA smooths over per-step pg variance — without it, lambda bounces
# every step depending on which advantage signs cancelled. The trade-off
# is a slower response when pg actually shifts (e.g., DAPO filter fully
# kicking in mid-training).
_TR_LAST_STEP_ID = None
_TR_CURRENT_STEP_PG_SUM = 0.0
_TR_CURRENT_STEP_CONS_SUM = 0.0
_TR_PREV_STEP_PG_ABS = None    # |signed Σ pg_per_mb| over previous step (EMA or raw)
_TR_PREV_STEP_CONS = None      # Σ |cons_loss_norm_per_mb| over previous step (EMA or raw)


def _get_marker_embed_from_disk(model_path: str, marker_id: int, device, dtype):
    """Load row `marker_id` of the HF model's input-embedding from safetensors
    on disk. Returns a regular tensor with stable own storage, identical on all
    ranks. Bypasses FSDP entirely — FSDP-sharded views of `embed.weight[marker_id]`
    have size-0 storage on non-owner ranks, which crashes when used in a (B,L,H)
    broadcast. This caches the result module-globally per worker.

    Frozen: no gradient flow back to the actual embed table. We accept that as
    a trade-off; the marker's *function* (gating) is what matters for the
    experiment, not its in-training learnability.
    """
    global _MARKER_EMBED
    if _MARKER_EMBED is not None:
        return _MARKER_EMBED
    import json
    import os as _os_local
    from safetensors.torch import load_file as _st_load
    # Find which shard has model.embed_tokens.weight
    embed_key_candidates = ("model.embed_tokens.weight",)
    index_path = _os_local.path.join(model_path, "model.safetensors.index.json")
    target_file = None
    target_key = None
    if _os_local.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        wm = index.get("weight_map", {})
        for k in embed_key_candidates:
            if k in wm:
                target_file = _os_local.path.join(model_path, wm[k])
                target_key = k
                break
    if target_file is None:
        # Try a single-file ckpt as fallback
        single = _os_local.path.join(model_path, "model.safetensors")
        if _os_local.path.exists(single):
            target_file = single
            target_key = embed_key_candidates[0]
    if target_file is None:
        raise RuntimeError(f"could not locate embed shard under {model_path!r}")
    print(f"[cons-marker] reading embed shard {target_file!r} key={target_key!r}", flush=True)
    embed_full = _st_load(target_file)[target_key]
    marker = embed_full[marker_id].detach().clone().to(device).to(dtype)
    _MARKER_EMBED = marker
    print(f"[cons-marker] marker loaded id={marker_id} shape={tuple(marker.shape)} "
          f"norm={float(marker.float().norm().item()):.4f}", flush=True)
    return _MARKER_EMBED


def _get_base_teacher(student_model, teacher_path: str):
    """Load (once) a frozen HF model from `teacher_path` onto the same device
    as the student. Returned model is in eval mode with requires_grad=False.

    Cached in module-level `_TEACHER_MODEL` — one copy per worker process,
    so each FSDP rank holds its own full (not sharded) copy. For 7B bf16
    that's ~14 GB per H200 rank, which is fine.
    """
    global _TEACHER_MODEL, _TEACHER_MODE
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
    _TEACHER_MODE = "base"
    print(f"[cons-teacher] base teacher loaded; params={sum(p.numel() for p in m.parameters())/1e9:.2f} B", flush=True)
    return _TEACHER_MODEL


def _get_ema_teacher(student_model, init_path: str):
    """Initialize (once) an EMA-tracking teacher model. Loaded from
    `init_path` (typically the same HF dir as the student's base ckpt; at
    step 0 EMA = student = base). Each rank holds its own full copy and
    updates it in lockstep via `_ema_update_from_student`.
    """
    global _TEACHER_MODEL, _TEACHER_MODE
    if _TEACHER_MODEL is not None:
        return _TEACHER_MODEL
    from transformers import AutoModelForCausalLM
    p = next(student_model.parameters())
    device, dtype = p.device, p.dtype
    print(
        f"[cons-ema] initializing EMA teacher from {init_path!r} "
        f"-> device={device} dtype={dtype}",
        flush=True,
    )
    m = AutoModelForCausalLM.from_pretrained(init_path, torch_dtype=dtype)
    m.eval()
    for prm in m.parameters():
        prm.requires_grad = False
    m.to(device)
    _TEACHER_MODEL = m
    _TEACHER_MODE = "ema"
    print(f"[cons-ema] EMA teacher initialized; params={sum(p.numel() for p in m.parameters())/1e9:.2f} B", flush=True)
    return _TEACHER_MODEL


@torch.no_grad()
def _ema_update_from_student(student_model, ema_model, decay: float) -> None:
    """In-place EMA update: ema = decay*ema + (1-decay)*student.

    Uses FSDP2-compatible `get_state_dict(... full_state_dict=True)` to gather
    full student weights on each rank. Avoids `FSDP.summon_full_params` which
    is an FSDP1 API and corrupts FSDP2 state when called during a forward.

    Call this AT MOST ONCE per training step (use _LAST_EMA_UPDATE_STEP guard
    in the caller) — gathering full state is expensive.
    """
    ema_params = {n: p for n, p in ema_model.named_parameters()}

    student_state = None
    # First try the FSDP2-native checkpoint API (verl uses fsdp2 by default).
    try:
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict
        opts = StateDictOptions(full_state_dict=True, cpu_offload=False, broadcast_from_rank0=False)
        state = get_state_dict(student_model, options=opts)
        # get_state_dict returns (model_state, optim_state); we only need model state.
        student_state = state[0] if isinstance(state, tuple) else state
    except Exception as _e1:  # noqa: BLE001
        # FSDP1 fallback via summon_full_params.
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            student_state = {}
            with FSDP.summon_full_params(student_model, writeback=False, offload_to_cpu=False):
                for n, p in student_model.named_parameters():
                    student_state[n] = p.detach().clone()
        except Exception as _e2:  # noqa: BLE001
            # Last resort: best-effort iteration (will be wrong under FSDP sharding).
            student_state = {n: p for n, p in student_model.named_parameters()}

    if student_state is None:
        return

    for name, sp in student_state.items():
        ema_p = ema_params.get(name)
        if ema_p is None:
            cleaned = name.replace("_fsdp_wrapped_module.", "").replace(".module.", ".")
            ema_p = ema_params.get(cleaned)
        if ema_p is None and name.startswith("module."):
            ema_p = ema_params.get(name[len("module."):])
        if ema_p is None:
            continue
        if ema_p.shape != sp.shape:
            continue
        ema_p.data.mul_(decay).add_(sp.data.to(ema_p.dtype), alpha=1.0 - decay)
    # No second loop now — single fallback chain consolidated above.
    return
    # Dead code below kept only to satisfy the existing try/except indentation
    # contract; never reached.
    for name, sp in student_model.named_parameters():  # pragma: no cover
        ema_p = ema_params.get(name)
        if ema_p is None or ema_p.shape != sp.shape:
            continue


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


def _extract_per_sample_acc(micro_batch, B: int) -> torch.Tensor:
    """Return per-sample binary correctness as a float tensor of length B.

    Prefers `micro_batch["acc"]` (NonTensorStack from the DAPO reward
    manager's `reward_extra_info["acc"]` — set by reward_code_assert to 0/1).
    Falls back to `token_level_scores.sum(-1) > 0.5` (works for the coder
    reward where score = acc - overlong_penalty and overlong_penalty in [0, 1]).
    Raises if neither is available — fail loud rather than silently disable
    the correct-only filter.
    """
    # Try non-tensor stack first.
    try:
        raw = micro_batch["acc"]
    except (KeyError, IndexError):
        raw = None
    if raw is not None:
        vals = []
        try:
            for i in range(B):
                v = raw[i]
                # NonTensorData -> .data
                if hasattr(v, "data"):
                    v = v.data
                vals.append(float(v))
            return torch.tensor(vals, dtype=torch.float32)
        except Exception:  # noqa: BLE001
            pass  # fall through to token_level_scores
    # Fallback: token_level_scores. With code reward + overlong penalty in
    # [-1, 0], `score > 0` is a safe correctness indicator only when
    # overlong_penalty_factor < 1; with factor=1 a long-but-correct response
    # can have score = 0. Threshold at 0.5 to be safe (correct - half penalty).
    if "token_level_scores" in micro_batch.keys():
        tls = micro_batch["token_level_scores"]
        per_sample = tls.sum(-1)
        return (per_sample > 0.5).float().cpu()
    raise RuntimeError(
        "consistency hook: CONSISTENCY_CORRECT_ONLY=1 but micro_batch has no "
        "'acc' nor 'token_level_scores' — cannot determine per-sample "
        "correctness. Check that the reward manager populates one of these."
    )


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

    # Adaptive cons-weight scaling against the pre-cons loss magnitude. When
    # enabled, the effective cons weight is multiplied by
    #   max(floor, min(ceiling, |loss|_last / |loss|_initial)),
    # so cons fades out alongside PG when DAPO's dynamic-sampling filter
    # collapses pg_loss to ~0 mid-training. Uses the LAST microbatch's |loss|
    # directly (no EMA) — adaptive_decay is accepted for backward compat but
    # ignored. State is always updated so the diagnostic metric is logged
    # even when scaling is off.
    adaptive_scale = _read("adaptive_scale", False, lambda v: str(v).lower() in {"1", "true", "yes"})
    adaptive_warmup_steps = _read("adaptive_warmup_steps", 10, int)
    adaptive_floor = _read("adaptive_floor", 0.05, float)
    adaptive_ceiling = _read("adaptive_ceiling", 1.0, float)

    global _PG_LOSS_LAST, _PG_LOSS_INITIAL, _ADAPTIVE_WARMUP_COUNT
    pg_loss_value = float(loss.detach().abs().item())
    _PG_LOSS_LAST = pg_loss_value
    if _PG_LOSS_INITIAL is None:
        _ADAPTIVE_WARMUP_COUNT += 1
        if _ADAPTIVE_WARMUP_COUNT >= adaptive_warmup_steps:
            _PG_LOSS_INITIAL = _PG_LOSS_LAST

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

    # Optional correct-only filter. When CONSISTENCY_CORRECT_ONLY=1, restrict
    # the cons-loss pool to samples whose binary correctness `acc` meets
    # CONSISTENCY_CORRECT_THRESHOLD (default 1.0). The fraction subsample
    # below then runs over the filtered pool, so cons gradient is computed
    # only from traces the policy got right.
    #
    # IMPORTANT: a per-rank `return loss` here would deadlock NCCL. If one DP
    # rank has zero correct samples and skips the cons forward, but other
    # ranks' filtered pools are non-empty, only the non-empty ranks call the
    # FSDP-collective backward (reduce_scatter on cons grads). The empty-pool
    # rank never joins the collective → 30 min watchdog timeout → SIGABRT.
    # First-attempt run on 2026-05-25 (commits before this fix) crashed
    # exactly this way at training step 1 on both nodes.
    #
    # Instead: on empty-pool ranks, keep ONE placeholder sample (so the cons
    # forward runs and FSDP collectives stay symmetric across DP ranks) and
    # set this rank's local cons weight to 0 — gradient contribution is zero,
    # so the empty rank effectively abstains while collectives still fire.
    correct_only = _read("correct_only", False, lambda v: str(v).lower() in {"1", "true", "yes"})
    correct_threshold = _read("correct_threshold", 1.0, float)
    if correct_only:
        acc_vec = _extract_per_sample_acc(micro_batch, B)
        correct_idx = (acc_vec >= correct_threshold).nonzero().flatten().tolist()
        n_correct = len(correct_idx)
        if metrics is not None:
            metrics["actor/cons_n_correct_in_batch"] = int(n_correct)
            metrics["actor/cons_correct_fraction"] = float(n_correct) / float(max(1, B))
        if n_correct == 0:
            # Empty pool on this rank → keep collectives alive with a 1-sample
            # placeholder and zero out this rank's cons weight.
            prompt_list = [prompt_list[0]]
            response_list = [response_list[0]]
            B = 1
            weight = 0.0
        else:
            prompt_list = [prompt_list[i] for i in correct_idx]
            response_list = [response_list[i] for i in correct_idx]
            B = n_correct

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

    # Teacher selection.
    #   "self" (default): cons target = student's own clean-view logits (CLLM-style).
    #   "base": cons target = frozen pre-RL base model's clean-view logits.
    #   "ema":  cons target = EMA of student's params (DMD2 fake_score analog;
    #           tracks current student so no "drag back to base"). Requires
    #           CONSISTENCY_TEACHER_PATH for the EMA's initial weights (the
    #           same path as the student's base ckpt is the natural choice).
    teacher_mode = _read("teacher", "self", lambda v: str(v).lower())
    teacher_path = _read("teacher_path", "", str)
    ema_decay = _read("ema_decay", 0.999, float)
    anchor_weight_init = _read("anchor_weight", 0.0, float)
    anchor_weight_final_env = _read("anchor_weight_final", None,
                                    lambda v: None if v in (None, "", "None") else float(v))
    anchor_mode = _read("anchor_mode", "clean", lambda v: str(v).lower())

    # Apply the SAME schedule (warmup_in/warmup_out/explicit weight_final) to
    # the anchor weight as to the cons weight. Uses the same current_step,
    # total_steps, ramp_start_frac, ramp_end_frac computed above for cons.
    if anchor_weight_final_env is not None:
        a_final = anchor_weight_final_env
        a_init = anchor_weight_init
    elif schedule == "warmup_in":
        a_init = 0.0
        a_final = anchor_weight_init
    elif schedule == "warmup_out":
        a_init = anchor_weight_init
        a_final = 0.0
    else:
        a_init = anchor_weight_init
        a_final = anchor_weight_init
    if total_steps <= 0 or a_init == a_final:
        anchor_weight = a_init
    else:
        progress = float(current_step) / float(total_steps)
        if progress <= ramp_start_frac:
            anchor_weight = a_init
        elif progress >= ramp_end_frac:
            anchor_weight = a_final
        else:
            span = max(1e-12, ramp_end_frac - ramp_start_frac)
            t = (progress - ramp_start_frac) / span
            anchor_weight = a_init + (a_final - a_init) * t
    teacher_model = None
    if teacher_mode == "base":
        if not teacher_path:
            raise RuntimeError(
                "CONSISTENCY_TEACHER=base requires CONSISTENCY_TEACHER_PATH "
                "to be set to the HF model dir of the frozen base teacher."
            )
        teacher_model = _get_base_teacher(engine.module, teacher_path)
    elif teacher_mode == "ema":
        if not teacher_path:
            raise RuntimeError(
                "CONSISTENCY_TEACHER=ema requires CONSISTENCY_TEACHER_PATH "
                "(used as the EMA's initial weights — typically the base ckpt)."
            )
        teacher_model = _get_ema_teacher(engine.module, teacher_path)
        # Update EMA AT MOST ONCE per training step. With use_dynamic_bsz=True
        # the cons hook fires multiple times per step (one per micro-chunk);
        # multiple state-dict gathers per step is what was corrupting FSDP2's
        # forward state and crashing 3-block runs.
        global _LAST_EMA_UPDATE_STEP
        if current_step > 0 and current_step != _LAST_EMA_UPDATE_STEP:
            _ema_update_from_student(engine.module, teacher_model, ema_decay)
            _LAST_EMA_UPDATE_STEP = current_step

    compute_anchor = anchor_weight > 0.0 and teacher_model is not None

    # Load the marker embed from disk if marker is enabled (avoids the
    # FSDP-sharded `embed.weight[marker_id]` size-0 storage problem).
    use_marker = _read("use_draft_marker", False, lambda v: str(v).lower() in {"1", "true", "yes"})
    marker_embed_override = None
    if use_marker:
        marker_path = _read("marker_path", "", str) or _read("teacher_path", "", str)
        marker_id_local = _read("marker_token_id", 151665, int)
        if marker_path:
            try:
                p = next(engine.module.parameters())
                marker_embed_override = _get_marker_embed_from_disk(
                    marker_path, marker_id_local, p.device, p.dtype,
                )
            except Exception as _e:  # noqa: BLE001
                print(f"[cons-marker] failed to load marker from disk: {_e}", flush=True)
                marker_embed_override = None

    cons_loss, anchor_loss, cons_metrics = compute_consistency_loss(
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
        compute_anchor=compute_anchor,
        anchor_mode=anchor_mode,
        marker_embed_override=marker_embed_override,
    )

    # === Global per-position normalization ============================
    # verl's PG loss is normalized as `masked_sum / batch_num_tokens * dp_size`
    # (workers/utils/losses.py:49, with batch_num_tokens all-reduced across DP
    # at workers/engine/fsdp/transformer_impl.py:612). When this per-chunk PG
    # is summed across all N micro-batches per step, it yields the global
    # per-token-mean PG gradient.
    #
    # `cons_loss` and `anchor_loss` come back from compute_consistency_loss
    # as PER-CHUNK token-mean values with no global denominator. Adding them
    # to each micro-batch's loss and backward'ing N times accumulates an
    # N× over-counted cons gradient relative to PG. With our dynamic_bsz
    # config N≈10-15, so the cons gradient was effectively ~10-15× its
    # nominal `CONSISTENCY_WEIGHT`.
    #
    # Fix: scale each per-chunk cons by (chunk_tokens / batch_num_tokens * dp_size).
    # Per-chunk response_tokens is a clean proxy for "this chunk's share of the
    # global step's work" (cons positions scale ~ linearly with response tokens
    # given fixed block_size and fixed `fraction`). Summed across micro-batches
    # this ratio collapses to 1.0 (modulo balanced DP), so the cons gradient
    # ends at ~per-chunk-position-mean × dp_size (FSDP averaging then gives
    # per-position-mean) — exactly the PG normalization shape.
    cons_loss_scale = 1.0
    try:
        # batch_num_tokens / dp_size are stored as `NonTensorData` on the
        # FULL batch via tensordict_utils.assign_non_tensor; they propagate
        # to every micro_batch chunk. Plain `.get()` returns the wrapper —
        # use verl's canonical unwrap helper so we get the raw float / int.
        from verl.utils.tensordict_utils import get_non_tensor_data as _get_nt
        _btn = _get_nt(micro_batch, "batch_num_tokens", None)
        _dp = _get_nt(micro_batch, "dp_size", None)
        if _btn is None or _dp is None:
            # Fall back to engine accessors if non-tensor data wasn't propagated.
            if _dp is None and hasattr(engine, "get_data_parallel_size"):
                _dp = int(engine.get_data_parallel_size())
        if _btn is not None and float(_btn) > 0:
            _resp_mask = None
            for _k in ("response_mask", "loss_mask"):
                if _k in micro_batch.keys():
                    _resp_mask = micro_batch[_k]
                    break
            if _resp_mask is not None and _dp is not None:
                _chunk_tokens = float(_resp_mask.to(torch.bool).sum().item())
                cons_loss_scale = (_chunk_tokens / float(_btn)) * float(_dp)
        if not getattr(maybe_add_consistency_loss, "_scale_announced", False):
            print(
                f"[cons-norm] global per-position normalization ACTIVE: "
                f"batch_num_tokens={_btn} dp_size={_dp} "
                f"chunk_tokens={float(_resp_mask.to(torch.bool).sum().item()) if _resp_mask is not None else 'N/A'} "
                f"-> cons_loss_scale={cons_loss_scale:.6f}",
                flush=True,
            )
            maybe_add_consistency_loss._scale_announced = True
    except Exception as _scale_exc:  # noqa: BLE001
        if not getattr(maybe_add_consistency_loss, "_scale_warn", False):
            print(f"[cons-norm] could not compute scale, using 1.0: {_scale_exc}", flush=True)
            maybe_add_consistency_loss._scale_warn = True
        cons_loss_scale = 1.0

    cons_loss_norm = cons_loss * cons_loss_scale
    anchor_loss_norm = (anchor_loss * cons_loss_scale) if anchor_loss is not None else None
    # ===================================================================

    # Apply adaptive scale to BOTH cons and anchor weights. Both are
    # auxiliary losses that should fade alongside PG; scaling them together
    # preserves the cons:anchor ratio chosen at config time.
    adaptive_ratio = 1.0
    if adaptive_scale and _PG_LOSS_INITIAL is not None and _PG_LOSS_INITIAL > 0.0:
        adaptive_ratio = _PG_LOSS_LAST / _PG_LOSS_INITIAL
        adaptive_ratio = max(adaptive_floor, min(adaptive_ceiling, adaptive_ratio))
        weight = weight * adaptive_ratio
        anchor_weight = anchor_weight * adaptive_ratio

    # Target-ratio mode (Nemotron-Diffusion style), per-STEP edition.
    #
    # `target_lambda` is computed from the totals of the LAST COMPLETED step:
    #   target_lambda = target_ratio × |Σ_mb signed pg_per_mb| / Σ_mb cons_per_mb
    # All microbatches in the current step use the SAME `weight` (snapshot of
    # the previous step's totals). Within step s, the resulting per-step ratio
    # Σ_mb (weight × cons_per_mb) / |Σ_mb pg_per_mb| ≈ target_ratio (with a
    # 1-step lag from the snapshot).
    #
    # Why this differs from the per-microbatch version: per-mb |pg| over-counts
    # the per-step pg magnitude (signed sums cancel across micro-batches in DAPO).
    # Per-mb enforcement made cons ~N× too strong at the per-step level.
    #
    # Note: this is intentionally NOT an EMA — just a 1-step lookback.
    target_ratio = _read("target_ratio", 0.0, float)
    target_lambda_min = _read("target_lambda_min", 1e-5, float)
    target_lambda_max = _read("target_lambda_max", 1e-2, float)
    target_active = target_ratio > 0.0
    target_lambda = None

    if target_active:
        global _TR_LAST_STEP_ID, _TR_CURRENT_STEP_PG_SUM, _TR_CURRENT_STEP_CONS_SUM
        global _TR_PREV_STEP_PG_ABS, _TR_PREV_STEP_CONS

        # EMA decay across step snapshots. 0 = single-step lookback (original
        # behavior); 0.9 = blend last step at 10% weight, history at 90% — much
        # smoother lambda trajectory. Without EMA, per-step pg variance from
        # advantage cancellation bounces lambda every step.
        target_ema_decay = _read("target_ema_decay", 0.0, float)

        # Step boundary: blend (or snapshot) the just-completed step's totals.
        if current_step != _TR_LAST_STEP_ID:
            if _TR_LAST_STEP_ID is not None and _TR_CURRENT_STEP_CONS_SUM > 0:
                new_pg_abs = abs(_TR_CURRENT_STEP_PG_SUM)
                new_cons = _TR_CURRENT_STEP_CONS_SUM
                if target_ema_decay > 0.0 and _TR_PREV_STEP_PG_ABS is not None:
                    a = target_ema_decay
                    _TR_PREV_STEP_PG_ABS = a * _TR_PREV_STEP_PG_ABS + (1.0 - a) * new_pg_abs
                    _TR_PREV_STEP_CONS = a * _TR_PREV_STEP_CONS + (1.0 - a) * new_cons
                else:
                    _TR_PREV_STEP_PG_ABS = new_pg_abs
                    _TR_PREV_STEP_CONS = new_cons
            _TR_CURRENT_STEP_PG_SUM = 0.0
            _TR_CURRENT_STEP_CONS_SUM = 0.0
            _TR_LAST_STEP_ID = current_step

        # Accumulate this microbatch into the current step's totals.
        _TR_CURRENT_STEP_PG_SUM += float(loss.detach().item())              # SIGNED
        _TR_CURRENT_STEP_CONS_SUM += float(cons_loss_norm.detach().abs().item())  # positive

        # Set this microbatch's weight from the previous step's snapshot.
        if (
            _TR_PREV_STEP_PG_ABS is not None
            and _TR_PREV_STEP_CONS is not None
            and _TR_PREV_STEP_CONS > 1e-12
        ):
            target_lambda = target_ratio * _TR_PREV_STEP_PG_ABS / _TR_PREV_STEP_CONS
        else:
            # Bootstrap (first step): fall back to per-mb estimate.
            pg_value = float(loss.detach().abs().item())
            cons_value = float(cons_loss_norm.detach().abs().item())
            if cons_value > 1e-12:
                target_lambda = target_ratio * pg_value / cons_value
            else:
                target_lambda = target_lambda_min

        target_lambda = max(target_lambda_min, min(target_lambda_max, target_lambda))

        # Apply the schedule MULTIPLIER on top of target_lambda. The schedule
        # block above computed `weight` from CONSISTENCY_WEIGHT and
        # CONSISTENCY_WEIGHT_FINAL — when target_ratio overrides weight to
        # target_lambda, we still want the schedule's decay shape (e.g.
        # warmup_out → ratio decays 1.0 → 0.0) to fade cons out late. Use the
        # ratio between the scheduled weight and its starting value.
        schedule_mult = 1.0
        if actual_init > 0:
            schedule_mult = float(weight) / float(actual_init)
        elif weight_final > 0 and actual_init == 0:
            # warmup_in case: schedule_mult ramps 0 → 1 over the window
            schedule_mult = float(weight) / float(weight_final)
        schedule_mult = max(0.0, min(1.0, schedule_mult))

        weight = target_lambda * schedule_mult
        # Only lock anchor to cons weight when anchor is actually active.
        # `compute_anchor` (set above) requires anchor_weight_init > 0 AND a
        # teacher. Without this guard, target_ratio overrides the default
        # anchor_weight=0 to target_lambda, which doesn't add an anchor term
        # to the loss (anchor_loss is None) but produces a misleading non-zero
        # `actor/anchor_weight_effective` metric.
        if compute_anchor:
            anchor_weight = weight

    # NOTE: We previously had a `CONSISTENCY_LOG_GRAD_NORMS=1` knob that called
    # `torch.autograd.grad(weight * cons_loss_norm, model.parameters(), ...)`
    # to log component grad norms. Under FSDP2 (verl's default), `parameters()`
    # returns sharded DTensors that are NOT leaves of the forward graph (the
    # graph holds the all-gathered shadow params), so `allow_unused=True`
    # returned None for every param and the resulting norm was always 0.0.
    # Both runs from 2026-05-28 showed `actor/{cons,pg}_grad_norm_local=0.0`.
    # Dropping that path. The cons:PG balance is observable from the much
    # cheaper loss-magnitude proxies:
    #   actor/cons_loss_effective  = weight * |cons_loss_normalized|  (per chunk)
    #   actor/pg_loss              = verl's already-globally-normalized PG
    # Their per-step values (after verl's SUM aggregation across micro-batches
    # for pg, MEAN for cons_loss_effective) approximate the relative gradient
    # contribution to the optimizer step.

    if metrics is not None:
        metrics["actor/cons_loss"] = float(cons_loss.detach().item())
        # cons_loss_normalized = the per-chunk value entering backward.
        # cons_loss_effective = weight * |normalized| = per-chunk loss-magnitude
        # contribution; cons_loss_effective summed across all chunks per step
        # approximates the per-step weighted cons loss magnitude that drives
        # the gradient.
        metrics["actor/cons_loss_normalized"] = float(cons_loss_norm.detach().item())
        metrics["actor/cons_loss_scale"] = float(cons_loss_scale)
        metrics["actor/cons_loss_effective"] = float(weight * float(cons_loss_norm.detach().abs().item()))
        metrics["actor/cons_weight_effective"] = float(weight)
        metrics["actor/anchor_weight_effective"] = float(anchor_weight)
        metrics["actor/cons_pg_loss_last"] = float(_PG_LOSS_LAST if _PG_LOSS_LAST is not None else 0.0)
        metrics["actor/cons_pg_loss_initial"] = float(_PG_LOSS_INITIAL or 0.0)
        metrics["actor/cons_adaptive_ratio"] = float(adaptive_ratio)
        if target_active:
            metrics["actor/cons_target_ratio"] = float(target_ratio)
            metrics["actor/cons_target_lambda"] = float(target_lambda if target_lambda is not None else 0.0)
        if correct_only:
            metrics["actor/cons_n_correct_used"] = int(n_use)
        if anchor_loss is not None:
            metrics["actor/anchor_loss"] = float(anchor_loss.detach().item())
            metrics["actor/anchor_loss_normalized"] = float(anchor_loss_norm.detach().item())
        for k, v in cons_metrics.items():
            metrics[f"actor/{k}"] = v

    total = loss + weight * cons_loss_norm
    if anchor_loss_norm is not None:
        total = total + anchor_weight * anchor_loss_norm
    return total
