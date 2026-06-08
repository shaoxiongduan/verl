"""FIX B prototype: GPU-native target_argmax stash, batched GPU draft pack.

Differences from vllm_jacobi_patch.py:
  - RejectionSampler wrap keeps target_argmax + n_acc as GPU tensors (no .cpu()).
  - propose_ngram_draft_token_ids is replaced with a batched GPU draft builder
    that emits a single [B, K] int64 tensor (one cuda kernel + one .tolist()).
  - Cold-start path remains CPU/numpy (executes once per request lifetime).

Phase profile expected vs phase1 vllm_jacobi_patch_phase (CG@33 enabled):
  S_AM_CPU_TGT + S_OUT_CPU + S_AM_SPLIT (~0.097 ms) -> ~0.005 ms
  D_PROPOSE_BODY (~0.015 ms) -> ~0.005 ms
  Predicted savings: ~0.10 ms/forward
"""
from __future__ import annotations
import os, time, json, statistics as stats, atexit
import numpy as np

_MODE = os.environ.get("VLLM_JACOBI_MODE", "jacobi")
_WARMUP = int(os.environ.get("JACOBI_PROF_WARMUP", "30"))
_BUCKETS: dict[str, list[float]] = {}
_N = 0

# GPU-side stash from prev RejectionSampler.forward
_STASH = {
    "target_argmax_gpu": None,  # [N_tok] int32/int64
    "cu_num_draft_tokens": None,  # [B] int32, exclusive cumsum-ish
    "n_acc_gpu": None,          # [B] int64
    "B": 0, "K": 32,
}

# CPU fallback cache for cold start (first-time per request slot)
_COLD_RNG = np.random.default_rng(0)


def _b(name: str, ms: float):
    if _N >= _WARMUP:
        _BUCKETS.setdefault(name, []).append(ms)


def report():
    out = {}
    for k, v in _BUCKETS.items():
        if not v: continue
        s = sorted(v)
        out[k] = {
            "n": len(v), "mean_ms": stats.mean(v),
            "median_ms": s[len(s)//2],
            "p90_ms": s[int(0.9*len(s))] if len(s) > 10 else max(v),
        }
    out["_n_total"] = _N
    out["_mode"] = _MODE + "_v2"
    return out


def _atexit_dump():
    def d():
        rep = report()
        if rep["_n_total"] < 5: return
        path = os.environ.get("JACOBI_PHASE_DUMP", "/tmp/jacobi_phase_v2.json") + f".{os.getpid()}"
        try:
            with open(path, "w") as f:
                json.dump(rep, f, indent=2)
            print(f"[phase_v2] dumped {path} ({rep['_n_total']} iters)", flush=True)
        except Exception as e:
            print(f"[phase_v2] dump_err: {e}", flush=True)
    atexit.register(d)


class JacobiProposer:
    """Used only on cold start (numpy path, called once per request slot)."""
    def __init__(self, vllm_config):
        spec = vllm_config.speculative_config
        self.k = int(spec.num_speculative_tokens)
        self.vocab_size = int(vllm_config.model_config.get_vocab_size())
        _STASH["K"] = self.k
    def load_model(self, *a, **kw): return
    def dummy_run(self, *a, **kw): return
    def propose(self, token_ids_slice):
        # COLD START path. Called only when stash unavailable for this slot.
        K = self.k
        n = int(len(token_ids_slice))
        tl = min(K, n)
        tail = token_ids_slice[n-tl:n].astype(np.int64)
        if tl < K:
            pad = _COLD_RNG.integers(0, self.vocab_size, size=K-tl, dtype=np.int64)
            tail = np.concatenate([tail, pad])
        return tail


def enable():
    import torch
    sync = torch.cuda.synchronize
    perf = time.perf_counter

    import vllm.v1.spec_decode.ngram_proposer as ngm
    import vllm.v1.worker.gpu_model_runner as gmr
    class Shim(JacobiProposer): pass
    ngm.NgramProposer = Shim
    gmr.NgramProposer = Shim

    from vllm.v1.sample.rejection_sampler import RejectionSampler, PLACEHOLDER_TOKEN_ID
    orig_rs = RejectionSampler.forward

    def new_rs_forward(self, metadata, draft_probs, target_logits, bonus_token_ids, sampling_metadata):
        sync(); ta = perf()
        out = orig_rs(self, metadata, draft_probs, target_logits, bonus_token_ids, sampling_metadata)
        sync(); tb = perf()
        # GPU-side stash: keep argmax + n_acc on device.
        target_argmax_gpu = target_logits.argmax(dim=-1)  # [N_tok]
        # n_acc = #non-placeholder per req, minus 1 (bonus accounted)
        n_acc_gpu = (out != PLACEHOLDER_TOKEN_ID).sum(dim=-1).sub_(1).clamp_min_(0)
        sync(); tc = perf()
        _STASH["target_argmax_gpu"] = target_argmax_gpu
        _STASH["cu_num_draft_tokens"] = metadata.cu_num_draft_tokens
        _STASH["n_acc_gpu"] = n_acc_gpu
        _STASH["B"] = int(out.shape[0])
        td = perf()
        _b("S_RS_INNER", (tb-ta)*1e3)
        _b("S_AM_GPU_STASH", (tc-tb)*1e3)
        _b("S_STASH_ASSIGN", (td-tc)*1e3)
        return out
    RejectionSampler.forward = new_rs_forward

    # Replace propose_ngram_draft_token_ids with a single-tensor batched path.
    # Strategy:
    # - If stash is set and the per-req `nd` is K (full draft expected), do a
    #   single GPU gather: per req b, draft[b] = target_argmax[start_b+n_acc+1:
    #     start_b+K] padded by last element to length K.
    # - Else fall through to per-req propose (cold start).
    # Returns List[List[int]] (one final cpu().tolist() at end).
    GR = gmr.GPUModelRunner
    orig_ngram = GR.propose_ngram_draft_token_ids

    def w_ngram_batched(self, sampled_token_ids):
        sync(); t0 = perf()
        req_ids = self.input_batch.req_ids
        B = len(sampled_token_ids)
        K = int(_STASH["K"])

        # Decide per-req cold vs warm
        warm_mask = []
        skip_mask = []
        for i, sampled_ids in enumerate(sampled_token_ids):
            if not sampled_ids:
                skip_mask.append(True); warm_mask.append(False); continue
            req_id = req_ids[i]
            if req_id in self.input_batch.spec_decode_unsupported_reqs:
                skip_mask.append(True); warm_mask.append(False); continue
            ntok = self.input_batch.num_tokens_no_spec[i]
            if ntok >= self.max_model_len:
                skip_mask.append(True); warm_mask.append(False); continue
            skip_mask.append(False)
            warm_mask.append(_STASH["target_argmax_gpu"] is not None
                             and _STASH["B"] == B
                             and _STASH["n_acc_gpu"] is not None)

        all_warm = all(w for w, s in zip(warm_mask, skip_mask) if not s)
        out = [None] * B

        if all_warm and any(not s for s in skip_mask):
            # Batched GPU path. Build [B, K] draft tensor on GPU then .tolist().
            target_am = _STASH["target_argmax_gpu"]          # [N_tok]
            cu_nd = _STASH["cu_num_draft_tokens"]            # [B] int32, exclusive cumsum end-points
            n_acc = _STASH["n_acc_gpu"]                      # [B] int64
            device = target_am.device
            # start_b = cu_nd[b-1] (with cu_nd[-1]=0). We can compute via shift.
            B_t = cu_nd.shape[0]
            start = torch.empty_like(cu_nd)
            start[0] = 0
            if B_t > 1:
                start[1:] = cu_nd[:-1]
            # nd_b = cu_nd[b] - start[b]; expected == K for warm
            # offset_b = start[b] + n_acc[b] + 1
            offset = start + n_acc.to(start.dtype) + 1   # [B]
            # Build draft via gather: draft[b, k] = target_am[offset_b + k] if (offset_b+k) < start_b+nd_b
            # else pad with target_am[start_b+nd_b - 1] (last valid).
            k_range = torch.arange(K, device=device, dtype=start.dtype)
            idx = offset.unsqueeze(1) + k_range.unsqueeze(0)  # [B, K]
            limit = cu_nd.unsqueeze(1)                         # [B, 1]
            last_valid = (cu_nd - 1).unsqueeze(1)              # [B, 1]
            idx_clamped = torch.where(idx < limit, idx, last_valid.expand_as(idx))
            draft_gpu = target_am[idx_clamped.long()]         # [B, K] int*
            draft_lists = draft_gpu.tolist()                  # single sync
            j = 0
            for i in range(B):
                if skip_mask[i]:
                    out[i] = []
                else:
                    out[i] = draft_lists[j]
                    j += 1
            sync(); _b("T_DRAFT_NGRAM", (perf()-t0)*1e3)
            _b("D_BATCHED_PATH", 1.0)
            return out
        # COLD path: per-req via JacobiProposer (numpy)
        for i, sampled_ids in enumerate(sampled_token_ids):
            if skip_mask[i]:
                out[i] = []; continue
            ntok = self.input_batch.num_tokens_no_spec[i]
            drafter_output = self.drafter.propose(
                self.input_batch.token_ids_cpu[i, :ntok])
            if drafter_output is None or len(drafter_output) == 0:
                out[i] = []
            else:
                out[i] = drafter_output.tolist()
        sync(); _b("T_DRAFT_NGRAM", (perf()-t0)*1e3)
        _b("D_COLD_PATH", 1.0)
        return out

    GR.propose_ngram_draft_token_ids = w_ngram_batched

    # Plus the standard phase wraps so we can compare to phase profile.
    orig_prep = GR._prepare_inputs
    orig_preproc = GR._preprocess
    orig_sample = GR._sample
    orig_book = GR._bookkeeping_sync
    orig_draft = GR.propose_draft_token_ids
    orig_execute = GR.execute_model
    orig_load = GR.load_model

    def w_prep(self, *a, **kw):
        sync(); t0 = perf()
        r = orig_prep(self, *a, **kw)
        sync(); _b("T_PREPARE", (perf()-t0)*1e3)
        return r
    def w_preproc(self, *a, **kw):
        sync(); t0 = perf()
        r = orig_preproc(self, *a, **kw)
        sync(); _b("T_PREPROC", (perf()-t0)*1e3)
        return r
    def w_sample(self, *a, **kw):
        sync(); t0 = perf()
        r = orig_sample(self, *a, **kw)
        sync(); _b("T_SAMPLE", (perf()-t0)*1e3)
        return r
    def w_book(self, *a, **kw):
        sync(); t0 = perf()
        r = orig_book(self, *a, **kw)
        sync(); _b("T_BOOKKEEP", (perf()-t0)*1e3)
        return r
    def w_draft(self, *a, **kw):
        sync(); t0 = perf()
        r = orig_draft(self, *a, **kw)
        sync(); _b("T_DRAFT", (perf()-t0)*1e3)
        return r
    def w_execute(self, *a, **kw):
        global _N
        sync(); t0 = perf()
        r = orig_execute(self, *a, **kw)
        sync(); _b("T_OUTER", (perf()-t0)*1e3)
        _N += 1
        return r

    GR._prepare_inputs = w_prep
    GR._preprocess = w_preproc
    GR._sample = w_sample
    GR._bookkeeping_sync = w_book
    GR.propose_draft_token_ids = w_draft
    GR.execute_model = w_execute

    def w_load(self, *a, **kw):
        r = orig_load(self, *a, **kw)
        m = self.model
        if hasattr(m, 'forward'):
            orig_fwd = m.forward
            def w_fwd(*args, **kwargs):
                sync(); t0 = perf()
                out = orig_fwd(*args, **kwargs)
                sync(); _b("T_MODEL", (perf()-t0)*1e3)
                return out
            m.forward = w_fwd
        if hasattr(m, 'compute_logits'):
            orig_cl = m.compute_logits
            def w_cl(*args, **kwargs):
                sync(); t0 = perf()
                out = orig_cl(*args, **kwargs)
                sync(); _b("T_LOGITS", (perf()-t0)*1e3)
                return out
            m.compute_logits = w_cl
        return r
    GR.load_model = w_load

    _atexit_dump()
