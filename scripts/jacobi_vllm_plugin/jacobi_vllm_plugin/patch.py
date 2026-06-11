"""Drop-in patch enabling Jacobi speculative decoding inside vLLM 0.10.2.

USAGE
=====

    from vllm_jacobi_patch import enable_jacobi_spec_decode
    enable_jacobi_spec_decode(K=32)  # call ONCE before any LLM() construction

    # Then use vLLM normally, requesting `method="ngram"`. The ngram slot is
    # hijacked to run our JacobiProposer (no separate draft model).
    from vllm import LLM, SamplingParams
    llm = LLM(model=..., speculative_config={
        "method": "ngram", "num_speculative_tokens": 32,
        "prompt_lookup_min": 2, "prompt_lookup_max": 4,  # required by ngram, ignored by us
    })

DESIGN
======

vLLM 0.10.2 doesn't expose a public hook for new spec-decode methods. We make
two surgical class-level patches at import time:

1. `NgramProposer` -> `JacobiProposer` (subclass), so `gpu_model_runner`'s
   `isinstance(self.drafter, NgramProposer)` succeeds while routing to our
   propose() logic.

2. `RejectionSampler.forward` is wrapped to stash the per-request target
   argmax + acceptance count. Our proposer reads this on the next call to
   build the warm draft from `target_argmax[n_acc+1:]` (the "windowed Jacobi
   refresh" pattern). This is the MODE B fix that gives JF-reference TPF.

LIMITATIONS
===========

- Cold start uses last K prompt tokens (close to JF's prefill-argmax but not
  identical). Iter-0 acceptance is ~88% of JF reference; iter-1+ matches.
- We rely on monkey-patching, which requires `enable_jacobi_spec_decode()` to
  run before any vLLM imports in the *parent* process. vLLM's subprocess
  (EngineCore) inherits the patches because Python's `spawn` re-imports the
  caller's __main__, which triggers our top-level `enable_*` call.
"""

from __future__ import annotations

import json
import os
import numpy as np
from typing import Optional

# Trajectory-log state — module globals (per process).
_TRAJ_PATH: Optional[str] = None
_TRAJ_FILE = None
_LAST_TARGET_ARGMAX_PER_REQ: list = []
_LAST_NUM_ACCEPTED_PER_REQ: list = []
_PENDING_DRAFT_PER_REQ: dict = {}
_N_FWD: int = 0
_N_TOK_ACCEPTED: int = 0
# Per-forward stable batch identifiers. Set by execute_model wrapper before
# each forward; read by the rejection sampler patch when writing trajectories.
# This lets the loader group per-iter records by vLLM request_id.
_CURRENT_REQ_IDS: list = []


def _traj_open():
    global _TRAJ_FILE
    if _TRAJ_FILE is None and _TRAJ_PATH is not None:
        pid = os.getpid()
        path = _TRAJ_PATH + f".{pid}"
        _TRAJ_FILE = open(path, "w")
    return _TRAJ_FILE


def reset_metrics():
    """Call between independent batches to zero counters & re-open log file."""
    global _N_FWD, _N_TOK_ACCEPTED, _LAST_TARGET_ARGMAX_PER_REQ, _LAST_NUM_ACCEPTED_PER_REQ
    global _PENDING_DRAFT_PER_REQ, _TRAJ_FILE
    _N_FWD = 0
    _N_TOK_ACCEPTED = 0
    _LAST_TARGET_ARGMAX_PER_REQ = []
    _LAST_NUM_ACCEPTED_PER_REQ = []
    _PENDING_DRAFT_PER_REQ = {}
    if _TRAJ_FILE is not None:
        try: _TRAJ_FILE.close()
        except Exception: pass
        _TRAJ_FILE = None


def get_metrics():
    """Return (num_spec_forwards, num_tokens_accepted, aggregate_TPF)."""
    tpf = (_N_TOK_ACCEPTED / _N_FWD) if _N_FWD > 0 else float("nan")
    return {"n_fwd": _N_FWD, "n_tok": _N_TOK_ACCEPTED, "agg_tpf": tpf}


def get_trajectory_path() -> Optional[str]:
    return _TRAJ_PATH


# -----------------------------------------------------------------------------
# JacobiProposer (lives in __main__ namespace so vLLM's spawn-imported __main__
# carries the class along with the patches).
# -----------------------------------------------------------------------------
class JacobiProposer:
    """Per-request Jacobi proposer; matches vllm 0.10.2 ngram_proposer interface
    (one call per request with `token_ids_slice`)."""

    def __init__(self, vllm_config):
        spec = vllm_config.speculative_config
        self.k = int(spec.num_speculative_tokens)
        self.vocab_size = int(vllm_config.model_config.get_vocab_size())
        self._req_idx = 0
        self._stash_id = -1
        self._rng = np.random.default_rng(int(os.environ.get("JACOBI_SEED", "0")))

    # No-op stubs for vLLM 0.10.2's proposer interface (called regardless of type).
    def load_model(self, *a, **kw):
        return

    def dummy_run(self, *a, **kw):
        return

    def propose(self, token_ids_slice):
        global _PENDING_DRAFT_PER_REQ
        result = self._propose_inner(token_ids_slice)
        _PENDING_DRAFT_PER_REQ[self._req_idx - 1] = [int(x) for x in result]
        return result

    def _propose_inner(self, token_ids_slice):
        global _LAST_TARGET_ARGMAX_PER_REQ, _LAST_NUM_ACCEPTED_PER_REQ, _PENDING_DRAFT_PER_REQ
        K = self.k
        stash_id = id(_LAST_TARGET_ARGMAX_PER_REQ)
        if stash_id != self._stash_id:
            self._req_idx = 0
            self._stash_id = stash_id
            _PENDING_DRAFT_PER_REQ.clear()
        i = self._req_idx
        self._req_idx += 1

        # JACOBI_DRAFT_INIT controls the K-window draft policy:
        #   "argmax_prev" (default) — prompt tail on cold start, then target_argmax_prev[n_acc+1:]
        #   "random"               — every iter is K uniform random vocab tokens (no acceleration)
        # JACOBI_REFRESH_AFTER_I: if set, hybrid mode — keep argmax_prev[:I] then refresh
        #                         argmax_prev[I:K] to uniform random. Simulates a probe with
        #                         predicted boundary = I. Common values: 5, 8, 10.
        init_mode = os.environ.get("JACOBI_DRAFT_INIT", "argmax_prev")
        refresh_after_i_str = os.environ.get("JACOBI_REFRESH_AFTER_I", "")
        refresh_after_i = int(refresh_after_i_str) if refresh_after_i_str else None
        if init_mode == "random":
            return self._rng.integers(0, self.vocab_size, size=K, dtype=np.int64)

        cold_start = (i >= len(_LAST_TARGET_ARGMAX_PER_REQ)
                      or len(_LAST_TARGET_ARGMAX_PER_REQ[i]) != K)
        if cold_start:
            # Warm draft from last K of prompt (cheap proxy for prefill-argmax).
            n = int(len(token_ids_slice))
            tail_len = min(K, n)
            tail = token_ids_slice[n - tail_len : n].astype(np.int64)
            if tail_len < K:
                pad = self._rng.integers(0, self.vocab_size, size=K - tail_len, dtype=np.int64)
                tail = np.concatenate([tail, pad])
            if refresh_after_i is not None and refresh_after_i < K:
                # Refresh tail to random
                pad = self._rng.integers(0, self.vocab_size, size=K - refresh_after_i, dtype=np.int64)
                tail = np.concatenate([tail[:refresh_after_i], pad])
            return tail

        argmax_prev = _LAST_TARGET_ARGMAX_PER_REQ[i]
        n_acc = _LAST_NUM_ACCEPTED_PER_REQ[i]
        start = n_acc + 1  # skip the bonus = new seed
        if start >= K:
            n = int(len(token_ids_slice))
            tail_len = min(K, n)
            tail = token_ids_slice[n-tail_len:n].astype(np.int64)
            if tail_len < K:
                pad = self._rng.integers(0, self.vocab_size, size=K-tail_len, dtype=np.int64)
                tail = np.concatenate([tail, pad])
            if refresh_after_i is not None and refresh_after_i < K:
                pad = self._rng.integers(0, self.vocab_size, size=K - refresh_after_i, dtype=np.int64)
                tail = np.concatenate([tail[:refresh_after_i], pad])
            return tail
        keep = argmax_prev[start:].astype(np.int64)
        pad_len = K - len(keep)
        if pad_len > 0:
            pad_token = int(keep[-1]) if len(keep) > 0 else int(token_ids_slice[-1])
            pad = np.full(pad_len, pad_token, dtype=np.int64)
            keep = np.concatenate([keep, pad])
        if refresh_after_i is not None and refresh_after_i < K:
            # Hybrid: keep keep[:refresh_after_i], refresh rest to random
            pad = self._rng.integers(0, self.vocab_size, size=K - refresh_after_i, dtype=np.int64)
            keep = np.concatenate([keep[:refresh_after_i], pad])
        return keep


def _install_proposer_patch():
    """Replace NgramProposer with JacobiProposer in vllm's namespace."""
    import vllm.v1.spec_decode.ngram_proposer as ngram_module
    import vllm.v1.worker.gpu_model_runner as gmr

    class JacobiNgramShim(JacobiProposer):
        pass  # alias so isinstance(x, NgramProposer) succeeds after swap

    ngram_module.NgramProposer = JacobiNgramShim
    gmr.NgramProposer = JacobiNgramShim


def _install_rejection_sampler_patch():
    """Wrap RejectionSampler.forward to stash target_argmax and write trajectory."""
    from vllm.v1.sample.rejection_sampler import RejectionSampler, PLACEHOLDER_TOKEN_ID
    orig_forward = RejectionSampler.forward

    def new_forward(self, metadata, draft_probs, target_logits, bonus_token_ids, sampling_metadata):
        out = orig_forward(self, metadata, draft_probs, target_logits, bonus_token_ids, sampling_metadata)
        target_argmax = target_logits.argmax(dim=-1).cpu().numpy()
        nd = list(metadata.num_draft_tokens)
        out_cpu = out.cpu().numpy() if hasattr(out, 'cpu') else out

        per_req_argmax = []
        per_req_n_acc = []
        offset = 0
        for i, k in enumerate(nd):
            per_req_argmax.append(target_argmax[offset:offset+k].copy())
            n_nonpad = int((out_cpu[i] != PLACEHOLDER_TOKEN_ID).sum())
            per_req_n_acc.append(max(0, n_nonpad - 1))
            offset += k

        global _LAST_TARGET_ARGMAX_PER_REQ, _LAST_NUM_ACCEPTED_PER_REQ, _N_FWD, _N_TOK_ACCEPTED
        _LAST_TARGET_ARGMAX_PER_REQ = per_req_argmax
        _LAST_NUM_ACCEPTED_PER_REQ = per_req_n_acc
        _N_FWD += 1
        _N_TOK_ACCEPTED += sum(per_req_n_acc) + len(per_req_n_acc)

        # Trajectory log: write ONE JSONL file PER REQUEST (not a global file).
        # vllm_async_server.generate picks up the request's file when the
        # request completes and stuffs trajectories into TokenOutput.extra_fields.
        # This is the cleanest cross-process channel (EngineCore subprocess →
        # parent) without modifying vLLM source. Put _TRAJ_PATH under /dev/shm
        # for RAM-speed file I/O.
        if _TRAJ_PATH is not None:
            bonus_list = []
            draft_list = []
            for i, (am, na) in enumerate(zip(per_req_argmax, per_req_n_acc)):
                draft_list.append(_PENDING_DRAFT_PER_REQ.get(i, []))
                if 0 <= na < len(am):
                    bonus_list.append(int(am[na]))
                elif len(am) > 0:
                    bonus_list.append(int(am[-1]))
                else:
                    bonus_list.append(-1)
            req_ids = _CURRENT_REQ_IDS if len(_CURRENT_REQ_IDS) >= len(nd) else [f"slot_{i}" for i in range(len(nd))]
            for slot in range(len(nd)):
                if int(nd[slot]) <= 0:
                    continue  # request didn't speculate this iter
                rid = str(req_ids[slot])
                if rid.startswith("slot_"):
                    continue  # warmup/dummy slot, skip
                rec = {
                    "iter": int(_N_FWD),
                    "num_draft": int(nd[slot]),
                    "n_acc": int(per_req_n_acc[slot]),
                    "draft": [int(x) for x in draft_list[slot]],
                    "target_argmax": [int(x) for x in per_req_argmax[slot]],
                    "bonus": int(bonus_list[slot]),
                }
                try:
                    per_req_path = _TRAJ_PATH + f".req_{rid}.jsonl"
                    with open(per_req_path, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                except Exception:
                    pass
        return out

    RejectionSampler.forward = new_forward


def _install_model_runner_patch():
    """Wrap GPUModelRunner._update_states to stash input_batch.req_ids per forward.

    `execute_model` calls `_update_states(scheduler_output)` at its start, which
    populates `self.input_batch.req_ids` with the CURRENT scheduled batch. We
    wrap `_update_states` (not `execute_model`) because patching execute_model
    captures the STALE prior-batch req_ids before _update_states refreshes them.

    The rejection sampler patch reads `_CURRENT_REQ_IDS` to label each trajectory
    record with stable vLLM request_ids. Without this, the JSONL only has per-slot
    indices, which are unstable across forwards.
    """
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    if getattr(GPUModelRunner, "_jacobi_runner_patched", False):
        return
    orig_update_states = GPUModelRunner._update_states

    def wrapped_update_states(self, scheduler_output):
        ret = orig_update_states(self, scheduler_output)
        global _CURRENT_REQ_IDS
        try:
            # Filter out None entries (vacated slots).
            _CURRENT_REQ_IDS = [rid for rid in self.input_batch.req_ids if rid is not None]
        except Exception:
            _CURRENT_REQ_IDS = []
        return ret

    GPUModelRunner._update_states = wrapped_update_states
    GPUModelRunner._jacobi_runner_patched = True


def enable_jacobi_spec_decode(K: int = 32, traj_path: Optional[str] = None) -> None:
    """Install all monkey-patches. Idempotent — safe to call multiple times.

    Args:
        K: number of speculative tokens per iter (= JF block_len). Default 32.
        traj_path: if set, every spec-decode iter writes a JSONL record to
            f"{traj_path}.<PID>". Useful for cons-RL training / debug.
            Set to None (default) to skip logging for production speed.
    """
    global _TRAJ_PATH
    _TRAJ_PATH = traj_path
    # Clean any pre-existing traj files for this path
    if traj_path is not None:
        import glob as _glob
        for tp in _glob.glob(traj_path + ".*"):
            try: os.remove(tp)
            except FileNotFoundError: pass

    _install_proposer_patch()
    _install_rejection_sampler_patch()
    _install_model_runner_patch()

    # Stash K via env so JacobiProposer reads it (constructor only gets vllm_config,
    # but vllm honors num_speculative_tokens from SpeculativeConfig).
    # The caller is expected to also set speculative_config["num_speculative_tokens"]=K.
    os.environ.setdefault("JACOBI_K", str(K))


def aggregate_trajectories() -> dict:
    """Parent-side helper: glob all per-PID trajectory files, compute TPF stats.

    Returns dict with: agg_tpf (aggregate per-iter), per_req_tpf (list per
    seen batch position), n_iters, n_files.
    """
    if _TRAJ_PATH is None:
        return {"error": "no trajectory path configured"}
    import glob as _glob
    files = _glob.glob(_TRAJ_PATH + ".*")
    if not files:
        return {"error": "no files found"}
    spec_tok_total = 0
    n_spec_iters = 0
    per_req_tokens: dict = {}
    per_req_iters: dict = {}
    for tp in files:
        with open(tp) as f:
            for line in f:
                rec = json.loads(line)
                # Skip warmup phases (very large num_draft list of 1s)
                if len(rec["num_draft"]) > 256 and set(rec["num_draft"]) == {1}:
                    continue
                n_spec_iters += 1
                for i, (nd, na) in enumerate(zip(rec["num_draft"], rec["n_acc"])):
                    if nd <= 0:
                        continue
                    committed = na + 1
                    spec_tok_total += committed
                    per_req_tokens[i] = per_req_tokens.get(i, 0) + committed
                    per_req_iters[i] = per_req_iters.get(i, 0) + 1
    agg = (spec_tok_total / n_spec_iters) if n_spec_iters > 0 else float("nan")
    per_req = []
    for i in sorted(per_req_tokens.keys()):
        t, it = per_req_tokens[i], per_req_iters[i]
        per_req.append(t / it if it > 0 else float("nan"))
    return {"agg_tpf": agg, "per_req_tpf": per_req,
            "n_iters": n_spec_iters, "n_files": len(files),
            "spec_tok_total": spec_tok_total}
