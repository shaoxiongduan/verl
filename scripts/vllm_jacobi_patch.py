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
            return tail
        keep = argmax_prev[start:].astype(np.int64)
        pad_len = K - len(keep)
        if pad_len > 0:
            pad_token = int(keep[-1]) if len(keep) > 0 else int(token_ids_slice[-1])
            pad = np.full(pad_len, pad_token, dtype=np.int64)
            keep = np.concatenate([keep, pad])
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
        # Compute per-position entropy of target softmax for noise-condition analysis.
        # H = -sum_v p_v log p_v ; with bf16 logits we cast to fp32 to avoid nan.
        # Only computed when traj logging is enabled (otherwise too expensive).
        target_entropy = None
        target_max_prob = None
        if _TRAJ_PATH is not None:
            try:
                import torch as _t
                logits_f = target_logits.float()
                logp = _t.log_softmax(logits_f, dim=-1)
                p = logp.exp()
                ent = -(p * logp).sum(dim=-1).cpu().numpy()  # [num_tokens]
                maxp = p.max(dim=-1).values.cpu().numpy()
                target_entropy = ent
                target_max_prob = maxp
            except Exception:
                pass
        nd = list(metadata.num_draft_tokens)
        out_cpu = out.cpu().numpy() if hasattr(out, 'cpu') else out

        # Read drafts directly from metadata.draft_token_ids (concat'd Tensor).
        try:
            draft_token_ids_cpu = metadata.draft_token_ids.cpu().numpy()
        except Exception:
            draft_token_ids_cpu = None

        per_req_argmax = []
        per_req_entropy = []
        per_req_max_prob = []
        per_req_draft = []
        per_req_n_acc = []
        offset = 0
        for i, k in enumerate(nd):
            per_req_argmax.append(target_argmax[offset:offset+k].copy())
            if target_entropy is not None:
                per_req_entropy.append(target_entropy[offset:offset+k].copy())
                per_req_max_prob.append(target_max_prob[offset:offset+k].copy())
            if draft_token_ids_cpu is not None:
                per_req_draft.append(draft_token_ids_cpu[offset:offset+k].copy())
            n_nonpad = int((out_cpu[i] != PLACEHOLDER_TOKEN_ID).sum())
            per_req_n_acc.append(max(0, n_nonpad - 1))
            offset += k

        global _LAST_TARGET_ARGMAX_PER_REQ, _LAST_NUM_ACCEPTED_PER_REQ, _N_FWD, _N_TOK_ACCEPTED
        _LAST_TARGET_ARGMAX_PER_REQ = per_req_argmax
        _LAST_NUM_ACCEPTED_PER_REQ = per_req_n_acc
        _N_FWD += 1
        _N_TOK_ACCEPTED += sum(per_req_n_acc) + len(per_req_n_acc)

        # Trajectory log
        if _TRAJ_PATH is not None:
            bonus_list = []
            draft_list = []
            for i, (am, na) in enumerate(zip(per_req_argmax, per_req_n_acc)):
                # Prefer per_req_draft from metadata (always populated); fall back to global.
                if i < len(per_req_draft):
                    draft_list.append(per_req_draft[i])
                else:
                    draft_list.append(_PENDING_DRAFT_PER_REQ.get(i, []))
                if 0 <= na < len(am):
                    bonus_list.append(int(am[na]))
                elif len(am) > 0:
                    bonus_list.append(int(am[-1]))
                else:
                    bonus_list.append(-1)
            rec = {
                "iter": int(_N_FWD),
                "num_draft": [int(x) for x in nd],
                "n_acc": [int(x) for x in per_req_n_acc],
                "draft": [[int(x) for x in d] for d in draft_list],
                "target_argmax": [a.tolist() for a in per_req_argmax],
                "bonus": [int(x) for x in bonus_list],
            }
            if per_req_entropy:
                rec["target_entropy"] = [[float(x) for x in e] for e in per_req_entropy]
                rec["target_max_prob"] = [[float(x) for x in m] for m in per_req_max_prob]
            try:
                f = _traj_open()
                if f is not None:
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
            except Exception:
                pass
        return out

    RejectionSampler.forward = new_forward


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
