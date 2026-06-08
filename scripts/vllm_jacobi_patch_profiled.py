"""Instrumented variant of vllm_jacobi_patch.py.

Adds per-segment timing inside the spec-decode hot path so we can attribute the
~4.2 ms/forward Jacobi overhead at BS=1 to specific code paths.

Segments timed (per spec-decode forward, AFTER warmup):
  T_RS_INNER      orig RejectionSampler.forward (triton greedy+compute_probs)
  T_AM_CPU_SYNC   target_logits.argmax(dim=-1).cpu().numpy()  <-- suspected hot spot
  T_AM_SPLIT      per-req slicing + .copy() + n_acc compute
  T_PROPOSE_CALL  JacobiProposer.propose() body (numpy reads, draft assembly)
  T_PROPOSE_LIST  drafter_output.tolist() in propose_ngram_draft_token_ids
  T_PROPOSE_WALL  full propose_ngram_draft_token_ids wall (all reqs)
  T_REST_PER_FWD  (next RS.forward enter) - (prev RS.forward exit) - (above wraps)
                  approximates: scheduler + model fwd + sampling + bookkeeping
  T_PER_FWD       wall time between successive RS.forward enters (= ms/forward)

Run: python scripts/_diag_jacobi_profiled.py (uses this patch under the hood)
"""
from __future__ import annotations

import os
import time
import numpy as np
import statistics as stats
from typing import Optional

# Buckets: list of ms per spec-decode forward (after WARMUP iters discarded)
_WARMUP = int(os.environ.get("JACOBI_PROF_WARMUP", "20"))
_BUCKETS: dict[str, list[float]] = {}
_N_FWD = 0
_LAST_RS_EXIT_PERF: Optional[float] = None
_LAST_RS_ENTER_PERF: Optional[float] = None
_CUR_PROPOSE_LIST_MS = 0.0   # filled by the gpu_model_runner wrap

_LAST_TARGET_ARGMAX_PER_REQ: list = []
_LAST_NUM_ACCEPTED_PER_REQ: list = []


def _bucket(name: str, ms: float):
    _BUCKETS.setdefault(name, []).append(ms)


def reset_metrics():
    global _BUCKETS, _N_FWD, _LAST_RS_EXIT_PERF, _LAST_RS_ENTER_PERF
    global _LAST_TARGET_ARGMAX_PER_REQ, _LAST_NUM_ACCEPTED_PER_REQ
    _BUCKETS = {}
    _N_FWD = 0
    _LAST_RS_EXIT_PERF = None
    _LAST_RS_ENTER_PERF = None
    _LAST_TARGET_ARGMAX_PER_REQ = []
    _LAST_NUM_ACCEPTED_PER_REQ = []


def report_metrics() -> dict:
    out = {}
    for k, v in _BUCKETS.items():
        if not v:
            continue
        out[k] = {
            "n": len(v),
            "mean_ms": stats.mean(v),
            "median_ms": stats.median(v),
            "p90_ms": sorted(v)[int(0.9 * len(v))] if len(v) > 10 else max(v),
        }
    out["_n_fwd_recorded"] = _N_FWD
    return out


# -----------------------------------------------------------------------------
# JacobiProposer — same logic, with timing.
# -----------------------------------------------------------------------------
class JacobiProposer:
    def __init__(self, vllm_config):
        spec = vllm_config.speculative_config
        self.k = int(spec.num_speculative_tokens)
        self.vocab_size = int(vllm_config.model_config.get_vocab_size())
        self._req_idx = 0
        self._stash_id = -1
        self._rng = np.random.default_rng(int(os.environ.get("JACOBI_SEED", "0")))

    def load_model(self, *a, **kw): return
    def dummy_run(self, *a, **kw): return

    def propose(self, token_ids_slice):
        t0 = time.perf_counter()
        result = self._propose_inner(token_ids_slice)
        t1 = time.perf_counter()
        # Only record after warmup
        if _N_FWD >= _WARMUP:
            _bucket("T_PROPOSE_CALL_us", (t1 - t0) * 1e6)
        return result

    def _propose_inner(self, token_ids_slice):
        global _LAST_TARGET_ARGMAX_PER_REQ, _LAST_NUM_ACCEPTED_PER_REQ
        K = self.k
        stash_id = id(_LAST_TARGET_ARGMAX_PER_REQ)
        if stash_id != self._stash_id:
            self._req_idx = 0
            self._stash_id = stash_id
        i = self._req_idx
        self._req_idx += 1

        cold = (i >= len(_LAST_TARGET_ARGMAX_PER_REQ)
                or len(_LAST_TARGET_ARGMAX_PER_REQ[i]) != K)
        if cold:
            n = int(len(token_ids_slice))
            tl = min(K, n)
            tail = token_ids_slice[n - tl: n].astype(np.int64)
            if tl < K:
                pad = self._rng.integers(0, self.vocab_size, size=K - tl, dtype=np.int64)
                tail = np.concatenate([tail, pad])
            return tail

        argmax_prev = _LAST_TARGET_ARGMAX_PER_REQ[i]
        n_acc = _LAST_NUM_ACCEPTED_PER_REQ[i]
        start = n_acc + 1
        if start >= K:
            n = int(len(token_ids_slice))
            tl = min(K, n)
            tail = token_ids_slice[n - tl: n].astype(np.int64)
            if tl < K:
                pad = self._rng.integers(0, self.vocab_size, size=K - tl, dtype=np.int64)
                tail = np.concatenate([tail, pad])
            return tail
        keep = argmax_prev[start:].astype(np.int64)
        pad_len = K - len(keep)
        if pad_len > 0:
            pad_token = int(keep[-1]) if len(keep) > 0 else int(token_ids_slice[-1])
            pad = np.full(pad_len, pad_token, dtype=np.int64)
            keep = np.concatenate([keep, pad])
        return keep


def _install_proposer_patch():
    import vllm.v1.spec_decode.ngram_proposer as ngram_module
    import vllm.v1.worker.gpu_model_runner as gmr

    class JacobiNgramShim(JacobiProposer):
        pass
    ngram_module.NgramProposer = JacobiNgramShim
    gmr.NgramProposer = JacobiNgramShim


def _install_propose_wall_patch():
    """Wrap propose_ngram_draft_token_ids to time the wall + the .tolist() phase."""
    import vllm.v1.worker.gpu_model_runner as gmr
    orig = gmr.GPUModelRunner.propose_ngram_draft_token_ids

    def wrapped(self, sampled_token_ids):
        global _CUR_PROPOSE_LIST_MS
        t0 = time.perf_counter()
        # Re-implement orig with extra timing of drafter_output.tolist()
        # We mirror the original logic but bucket tolist() time.
        req_ids = self.input_batch.req_ids
        out = []
        list_us = 0.0
        for i, sampled_ids in enumerate(sampled_token_ids):
            if not sampled_ids:
                out.append([]); continue
            req_id = req_ids[i]
            if req_id in self.input_batch.spec_decode_unsupported_reqs:
                out.append([]); continue
            ntok = self.input_batch.num_tokens_no_spec[i]
            if ntok >= self.max_model_len:
                out.append([]); continue
            drafter_output = self.drafter.propose(
                self.input_batch.token_ids_cpu[i, :ntok])
            if drafter_output is None or len(drafter_output) == 0:
                out.append([])
            else:
                tA = time.perf_counter()
                lst = drafter_output.tolist()
                tB = time.perf_counter()
                list_us += (tB - tA) * 1e6
                out.append(lst)
        t1 = time.perf_counter()
        if _N_FWD >= _WARMUP:
            _bucket("T_PROPOSE_WALL_us", (t1 - t0) * 1e6)
            _bucket("T_PROPOSE_LIST_us", list_us)
        return out

    gmr.GPUModelRunner.propose_ngram_draft_token_ids = wrapped


def _install_rejection_sampler_patch():
    import torch
    from vllm.v1.sample.rejection_sampler import RejectionSampler, PLACEHOLDER_TOKEN_ID
    orig_forward = RejectionSampler.forward

    def new_forward(self, metadata, draft_probs, target_logits, bonus_token_ids, sampling_metadata):
        global _N_FWD, _LAST_RS_EXIT_PERF, _LAST_RS_ENTER_PERF
        global _LAST_TARGET_ARGMAX_PER_REQ, _LAST_NUM_ACCEPTED_PER_REQ

        t_enter = time.perf_counter()
        if _LAST_RS_ENTER_PERF is not None and _N_FWD > _WARMUP:
            _bucket("T_PER_FWD_us", (t_enter - _LAST_RS_ENTER_PERF) * 1e6)
        if _LAST_RS_EXIT_PERF is not None and _N_FWD > _WARMUP:
            _bucket("T_BETWEEN_RS_us", (t_enter - _LAST_RS_EXIT_PERF) * 1e6)
        _LAST_RS_ENTER_PERF = t_enter

        # --- segment: original RejectionSampler.forward (triton) ---
        torch.cuda.synchronize()
        t_a = time.perf_counter()
        out = orig_forward(self, metadata, draft_probs, target_logits, bonus_token_ids, sampling_metadata)
        torch.cuda.synchronize()
        t_b = time.perf_counter()

        # --- segment: argmax + .cpu().numpy() (the suspected hot spot) ---
        target_argmax_gpu = target_logits.argmax(dim=-1)
        torch.cuda.synchronize()
        t_c = time.perf_counter()
        target_argmax = target_argmax_gpu.cpu().numpy()
        # .cpu() implicitly syncs but be safe
        torch.cuda.synchronize()
        t_d = time.perf_counter()

        # --- segment: per-req split/copy + n_acc compute ---
        nd = list(metadata.num_draft_tokens)
        out_cpu = out.cpu().numpy()  # already cheap, small tensor
        per_req_argmax = []
        per_req_n_acc = []
        offset = 0
        for i, k in enumerate(nd):
            per_req_argmax.append(target_argmax[offset:offset + k].copy())
            n_nonpad = int((out_cpu[i] != PLACEHOLDER_TOKEN_ID).sum())
            per_req_n_acc.append(max(0, n_nonpad - 1))
            offset += k
        t_e = time.perf_counter()

        _LAST_TARGET_ARGMAX_PER_REQ = per_req_argmax
        _LAST_NUM_ACCEPTED_PER_REQ = per_req_n_acc
        _N_FWD += 1

        if _N_FWD > _WARMUP:
            _bucket("T_RS_INNER_us", (t_b - t_a) * 1e6)
            _bucket("T_AM_KERNEL_us", (t_c - t_b) * 1e6)
            _bucket("T_AM_CPU_SYNC_us", (t_d - t_c) * 1e6)
            _bucket("T_AM_SPLIT_us", (t_e - t_d) * 1e6)

        _LAST_RS_EXIT_PERF = time.perf_counter()
        return out

    RejectionSampler.forward = new_forward


def _atexit_dump():
    import atexit, json, os
    def dump():
        out = report_metrics()
        if not out or out.get("_n_fwd_recorded", 0) < 5:
            return
        out["_pid"] = os.getpid()
        path = os.environ.get("JACOBI_PROF_DUMP", "/tmp/jacobi_prof.json") + f".{os.getpid()}"
        try:
            with open(path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"[jacobi_prof] dumped {path}", flush=True)
        except Exception as e:
            print(f"[jacobi_prof] dump failed: {e}", flush=True)
    atexit.register(dump)


def enable_jacobi_spec_decode(K: int = 32) -> None:
    os.environ.setdefault("JACOBI_K", str(K))
    _install_proposer_patch()
    _install_propose_wall_patch()
    _install_rejection_sampler_patch()
    _atexit_dump()
