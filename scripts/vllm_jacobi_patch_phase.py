"""Phase-level instrumentation for vLLM v1 spec-decode hot loop.

Wraps inner methods of GPUModelRunner.execute_model with cuda-sync + perf_counter
brackets so we get a per-execute_model breakdown:
  T_PREPARE      _prepare_inputs
  T_PREPROC      _preprocess
  T_MODEL        model forward (self.model.__call__)
  T_LOGITS       compute_logits
  T_SAMPLE       _sample
  T_RS_INNER       (inside sample, only Jacobi) orig RejectionSampler.forward
  T_AM_GPU         argmax kernel
  T_AM_CPU         .cpu().numpy() sync
  T_AM_SPLIT       per-req split/copy
  T_BOOKKEEP     _bookkeeping_sync
  T_DRAFT        propose_draft_token_ids
  T_OUTER        wall(execute_model)  (== ms/forward)

Bracketing uses torch.cuda.synchronize() so values are CPU-side ms; we accept the
extra sync overhead in exchange for clean attribution.

Both AR and Jacobi modes work. Set VLLM_JACOBI_MODE=ar to skip the proposer patch.
"""
from __future__ import annotations
import os, time, json, statistics as stats, atexit
import numpy as np

_MODE = os.environ.get("VLLM_JACOBI_MODE", "jacobi")  # ar | jacobi
_WARMUP = int(os.environ.get("JACOBI_PROF_WARMUP", "30"))
_BUCKETS: dict[str, list[float]] = {}
_N = 0
_LAST_TARGET_ARGMAX: list = []
_LAST_N_ACC: list = []


def _b(name: str, ms: float):
    if _N >= _WARMUP:
        _BUCKETS.setdefault(name, []).append(ms)


def report():
    out = {}
    for k, v in _BUCKETS.items():
        if not v: continue
        s = sorted(v)
        out[k] = {
            "n": len(v),
            "mean_ms": stats.mean(v),
            "median_ms": s[len(s)//2],
            "p90_ms": s[int(0.9*len(s))] if len(s) > 10 else max(v),
        }
    out["_n_total"] = _N
    out["_mode"] = _MODE
    return out


def _atexit_dump():
    def d():
        rep = report()
        if rep["_n_total"] < 5: return
        path = os.environ.get("JACOBI_PHASE_DUMP", "/tmp/jacobi_phase.json") + f".{os.getpid()}"
        try:
            with open(path, "w") as f:
                json.dump(rep, f, indent=2)
            print(f"[phase] dumped {path} ({rep['_n_total']} iters)", flush=True)
        except Exception as e:
            print(f"[phase] dump_err: {e}", flush=True)
    atexit.register(d)


# ----- Jacobi proposer -----
class JacobiProposer:
    def __init__(self, vllm_config):
        spec = vllm_config.speculative_config
        self.k = int(spec.num_speculative_tokens)
        self.vocab_size = int(vllm_config.model_config.get_vocab_size())
        self._req_idx = 0
        self._stash_id = -1
        self._rng = np.random.default_rng(0)
    def load_model(self, *a, **kw): return
    def dummy_run(self, *a, **kw): return
    def propose(self, token_ids_slice):
        global _LAST_TARGET_ARGMAX, _LAST_N_ACC
        K = self.k
        sid = id(_LAST_TARGET_ARGMAX)
        if sid != self._stash_id:
            self._req_idx = 0; self._stash_id = sid
        i = self._req_idx; self._req_idx += 1
        cold = (i >= len(_LAST_TARGET_ARGMAX)
                or len(_LAST_TARGET_ARGMAX[i]) != K)
        if cold:
            n = int(len(token_ids_slice))
            tl = min(K, n)
            tail = token_ids_slice[n-tl:n].astype(np.int64)
            if tl < K:
                pad = self._rng.integers(0, self.vocab_size, size=K-tl, dtype=np.int64)
                tail = np.concatenate([tail, pad])
            return tail
        argmax_prev = _LAST_TARGET_ARGMAX[i]
        n_acc = _LAST_N_ACC[i]
        start = n_acc + 1
        if start >= K:
            n = int(len(token_ids_slice))
            tl = min(K, n)
            tail = token_ids_slice[n-tl:n].astype(np.int64)
            if tl < K:
                pad = self._rng.integers(0, self.vocab_size, size=K-tl, dtype=np.int64)
                tail = np.concatenate([tail, pad])
            return tail
        keep = argmax_prev[start:].astype(np.int64)
        pad_len = K - len(keep)
        if pad_len > 0:
            pad_token = int(keep[-1]) if len(keep) > 0 else int(token_ids_slice[-1])
            pad = np.full(pad_len, pad_token, dtype=np.int64)
            keep = np.concatenate([keep, pad])
        return keep


def enable():
    import torch
    # 1) Install Jacobi proposer in vllm namespace (if Jacobi mode)
    if _MODE == "jacobi":
        import vllm.v1.spec_decode.ngram_proposer as ngm
        import vllm.v1.worker.gpu_model_runner as gmr
        class Shim(JacobiProposer): pass
        ngm.NgramProposer = Shim
        gmr.NgramProposer = Shim

        from vllm.v1.sample.rejection_sampler import RejectionSampler, PLACEHOLDER_TOKEN_ID
        orig_rs = RejectionSampler.forward

        def new_rs_forward(self, metadata, draft_probs, target_logits, bonus_token_ids, sampling_metadata):
            global _LAST_TARGET_ARGMAX, _LAST_N_ACC
            torch.cuda.synchronize(); ta = time.perf_counter()
            out = orig_rs(self, metadata, draft_probs, target_logits, bonus_token_ids, sampling_metadata)
            torch.cuda.synchronize(); tb = time.perf_counter()
            target_argmax_gpu = target_logits.argmax(dim=-1)
            torch.cuda.synchronize(); tc = time.perf_counter()
            target_argmax = target_argmax_gpu.cpu().numpy()
            torch.cuda.synchronize(); td = time.perf_counter()
            out_cpu = out.cpu().numpy()
            nd = list(metadata.num_draft_tokens)
            per_a = []; per_n = []
            off = 0
            for i, k in enumerate(nd):
                per_a.append(target_argmax[off:off+k].copy())
                n_nonpad = int((out_cpu[i] != PLACEHOLDER_TOKEN_ID).sum())
                per_n.append(max(0, n_nonpad - 1))
                off += k
            te = time.perf_counter()
            _LAST_TARGET_ARGMAX = per_a; _LAST_N_ACC = per_n
            _b("T_RS_INNER", (tb-ta)*1e3)
            _b("T_AM_GPU",   (tc-tb)*1e3)
            _b("T_AM_CPU",   (td-tc)*1e3)
            _b("T_AM_SPLIT", (te-td)*1e3)
            return out
        RejectionSampler.forward = new_rs_forward

    # 2) Wrap GPUModelRunner phases
    import vllm.v1.worker.gpu_model_runner as gmr
    GR = gmr.GPUModelRunner

    orig_prep = GR._prepare_inputs
    orig_preproc = GR._preprocess
    orig_sample = GR._sample
    orig_book = GR._bookkeeping_sync
    orig_draft = GR.propose_draft_token_ids if hasattr(GR, 'propose_draft_token_ids') else None
    orig_ngram = GR.propose_ngram_draft_token_ids if hasattr(GR, 'propose_ngram_draft_token_ids') else None
    orig_execute = GR.execute_model
    orig_load = GR.load_model

    def w_prep(self, *a, **kw):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        r = orig_prep(self, *a, **kw)
        torch.cuda.synchronize(); _b("T_PREPARE", (time.perf_counter()-t0)*1e3)
        return r

    def w_preproc(self, *a, **kw):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        r = orig_preproc(self, *a, **kw)
        torch.cuda.synchronize(); _b("T_PREPROC", (time.perf_counter()-t0)*1e3)
        return r

    def w_sample(self, *a, **kw):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        r = orig_sample(self, *a, **kw)
        torch.cuda.synchronize(); _b("T_SAMPLE", (time.perf_counter()-t0)*1e3)
        return r

    def w_book(self, *a, **kw):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        r = orig_book(self, *a, **kw)
        torch.cuda.synchronize(); _b("T_BOOKKEEP", (time.perf_counter()-t0)*1e3)
        return r

    def w_draft(self, *a, **kw):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        r = orig_draft(self, *a, **kw)
        torch.cuda.synchronize(); _b("T_DRAFT", (time.perf_counter()-t0)*1e3)
        return r

    def w_ngram(self, sampled_token_ids):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        r = orig_ngram(self, sampled_token_ids)
        torch.cuda.synchronize(); _b("T_DRAFT_NGRAM", (time.perf_counter()-t0)*1e3)
        return r

    def w_execute(self, *a, **kw):
        global _N
        torch.cuda.synchronize(); t0 = time.perf_counter()
        r = orig_execute(self, *a, **kw)
        torch.cuda.synchronize(); _b("T_OUTER", (time.perf_counter()-t0)*1e3)
        _N += 1
        return r

    GR._prepare_inputs = w_prep
    GR._preprocess = w_preproc
    GR._sample = w_sample
    GR._bookkeeping_sync = w_book
    if orig_draft is not None:
        GR.propose_draft_token_ids = w_draft
    if orig_ngram is not None:
        GR.propose_ngram_draft_token_ids = w_ngram
    GR.execute_model = w_execute

    # Wrap model.forward and compute_logits after model load
    def w_load(self, *a, **kw):
        r = orig_load(self, *a, **kw)
        # self.model is set now
        m = self.model
        if hasattr(m, 'forward'):
            orig_fwd = m.forward
            def w_fwd(*args, **kwargs):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                out = orig_fwd(*args, **kwargs)
                torch.cuda.synchronize(); _b("T_MODEL", (time.perf_counter()-t0)*1e3)
                return out
            m.forward = w_fwd
        if hasattr(m, 'compute_logits'):
            orig_cl = m.compute_logits
            def w_cl(*args, **kwargs):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                out = orig_cl(*args, **kwargs)
                torch.cuda.synchronize(); _b("T_LOGITS", (time.perf_counter()-t0)*1e3)
                return out
            m.compute_logits = w_cl
        return r
    GR.load_model = w_load

    _atexit_dump()
