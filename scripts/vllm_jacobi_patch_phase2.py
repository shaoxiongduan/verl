"""Sub-phase instrumentation focused on the residual gap after FIX A.

Adds finer brackets on top of vllm_jacobi_patch_phase:
  Inside _prepare_inputs -> times _calc_spec_decode_metadata.
  Inside _sample -> times bonus_logits index + sampler + target index + RS.
  Inside our RS wrap -> times each piece (RS, AM_GPU, AM_CPU, AM_SPLIT).
  Wraps propose_ngram tolist segment + Jacobi propose body separately.
"""
from __future__ import annotations
import os, time, json, statistics as stats, atexit
import numpy as np

_MODE = os.environ.get("VLLM_JACOBI_MODE", "jacobi")
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
            "n": len(v), "mean_ms": stats.mean(v),
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
        path = os.environ.get("JACOBI_PHASE_DUMP", "/tmp/jacobi_phase2.json") + f".{os.getpid()}"
        try:
            with open(path, "w") as f:
                json.dump(rep, f, indent=2)
            print(f"[phase2] dumped {path} ({rep['_n_total']} iters)", flush=True)
        except Exception as e:
            print(f"[phase2] dump_err: {e}", flush=True)
    atexit.register(d)


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
    sync = torch.cuda.synchronize
    perf = time.perf_counter

    if _MODE == "jacobi":
        import vllm.v1.spec_decode.ngram_proposer as ngm
        import vllm.v1.worker.gpu_model_runner as gmr_mod
        class Shim(JacobiProposer): pass
        ngm.NgramProposer = Shim
        gmr_mod.NgramProposer = Shim

        from vllm.v1.sample.rejection_sampler import RejectionSampler, PLACEHOLDER_TOKEN_ID
        orig_rs = RejectionSampler.forward

        def new_rs_forward(self, metadata, draft_probs, target_logits, bonus_token_ids, sampling_metadata):
            global _LAST_TARGET_ARGMAX, _LAST_N_ACC
            sync(); ta = perf()
            out = orig_rs(self, metadata, draft_probs, target_logits, bonus_token_ids, sampling_metadata)
            sync(); tb = perf()
            target_argmax_gpu = target_logits.argmax(dim=-1)
            sync(); tc = perf()
            target_argmax = target_argmax_gpu.cpu().numpy()
            sync(); td = perf()
            out_cpu = out.cpu().numpy()
            te = perf()
            nd = list(metadata.num_draft_tokens)
            per_a = []; per_n = []
            off = 0
            for i, k in enumerate(nd):
                per_a.append(target_argmax[off:off+k].copy())
                n_nonpad = int((out_cpu[i] != PLACEHOLDER_TOKEN_ID).sum())
                per_n.append(max(0, n_nonpad - 1))
                off += k
            tf = perf()
            _LAST_TARGET_ARGMAX = per_a; _LAST_N_ACC = per_n
            _b("S_RS_INNER", (tb-ta)*1e3)
            _b("S_AM_GPU",   (tc-tb)*1e3)
            _b("S_AM_CPU_TGT", (td-tc)*1e3)
            _b("S_OUT_CPU",  (te-td)*1e3)
            _b("S_AM_SPLIT", (tf-te)*1e3)
            return out
        RejectionSampler.forward = new_rs_forward

    import vllm.v1.worker.gpu_model_runner as gmr
    GR = gmr.GPUModelRunner

    orig_prep = GR._prepare_inputs
    orig_calc = GR._calc_spec_decode_metadata if hasattr(GR, '_calc_spec_decode_metadata') else None
    orig_preproc = GR._preprocess
    orig_sample = GR._sample
    orig_book = GR._bookkeeping_sync
    orig_draft = GR.propose_draft_token_ids if hasattr(GR, 'propose_draft_token_ids') else None
    orig_ngram = GR.propose_ngram_draft_token_ids if hasattr(GR, 'propose_ngram_draft_token_ids') else None
    orig_execute = GR.execute_model
    orig_load = GR.load_model

    def w_prep(self, *a, **kw):
        sync(); t0 = perf()
        r = orig_prep(self, *a, **kw)
        sync(); _b("T_PREPARE", (perf()-t0)*1e3)
        return r

    if orig_calc is not None:
        def w_calc(self, *a, **kw):
            sync(); t0 = perf()
            r = orig_calc(self, *a, **kw)
            sync(); _b("T_CALC_SPEC_META", (perf()-t0)*1e3)
            return r
        GR._calc_spec_decode_metadata = w_calc

    def w_preproc(self, *a, **kw):
        sync(); t0 = perf()
        r = orig_preproc(self, *a, **kw)
        sync(); _b("T_PREPROC", (perf()-t0)*1e3)
        return r

    # Re-implement _sample so we can time sub-segments without breaking semantics.
    def w_sample(self, logits, spec_decode_metadata):
        sampling_metadata = self.input_batch.sampling_metadata
        sync(); t0 = perf()
        if spec_decode_metadata is None:
            sampler_output = self.sampler(logits=logits, sampling_metadata=sampling_metadata)
            sync(); _b("T_SAMPLE", (perf()-t0)*1e3)
            return sampler_output
        sync(); ti = perf()
        bonus_logits = logits[spec_decode_metadata.bonus_logits_indices]
        sync(); tj = perf()
        sampler_output = self.sampler(logits=bonus_logits, sampling_metadata=sampling_metadata)
        sync(); tk = perf()
        bonus_token_ids = sampler_output.sampled_token_ids
        target_logits = logits[spec_decode_metadata.target_logits_indices]
        sync(); tl = perf()
        output_token_ids = self.rejection_sampler(
            spec_decode_metadata, None, target_logits, bonus_token_ids, sampling_metadata,
        )
        sync(); tm = perf()
        sampler_output.sampled_token_ids = output_token_ids
        self._update_states_after_model_execute(output_token_ids)
        sync(); tn = perf()
        _b("S_BONUS_IDX",   (tj-ti)*1e3)
        _b("S_BONUS_SAMPLER",(tk-tj)*1e3)
        _b("S_TGT_IDX",     (tl-tk)*1e3)
        _b("S_RS_FULL",     (tm-tl)*1e3)
        _b("S_UPDATE_STATE",(tn-tm)*1e3)
        _b("T_SAMPLE",      (tn-t0)*1e3)
        return sampler_output

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

    def w_ngram(self, sampled_token_ids):
        # Mirror orig logic with .tolist() timing split out
        sync(); t0 = perf()
        req_ids = self.input_batch.req_ids
        out = []
        propose_us = 0.0
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
            tA = perf()
            drafter_output = self.drafter.propose(
                self.input_batch.token_ids_cpu[i, :ntok])
            tB = perf()
            propose_us += (tB - tA) * 1e3
            if drafter_output is None or len(drafter_output) == 0:
                out.append([])
            else:
                tC = perf()
                lst = drafter_output.tolist()
                tD = perf()
                list_us += (tD - tC) * 1e3
                out.append(lst)
        sync(); _b("T_DRAFT_NGRAM", (perf()-t0)*1e3)
        _b("D_PROPOSE_BODY", propose_us)
        _b("D_TOLIST", list_us)
        return out

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
    if orig_draft is not None:
        GR.propose_draft_token_ids = w_draft
    if orig_ngram is not None and _MODE == "jacobi":
        GR.propose_ngram_draft_token_ids = w_ngram
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
