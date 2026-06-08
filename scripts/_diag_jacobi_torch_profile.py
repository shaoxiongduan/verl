"""torch.profiler-based phase breakdown for vLLM AR + Jacobi at BS=1.

vLLM's execute_model wraps phases in `record_function_or_nullcontext("Preprocess"|
"Forward"|"Postprocess"|"Sample"|"Bookkeep"|"Draft"|"EPLB")` — once we activate a
torch.profiler, those become labeled sections we can sum per name.

We also add labels for the inner pieces of our patch ("JF_RS_INNER", "JF_AM_GPU",
"JF_AM_CPU", "JF_AM_SPLIT", "JF_PROPOSE").

Outputs: per-section mean ms/forward over the steady-state portion (skipping the
first JACOBI_PROF_WARMUP forwards).
"""
from __future__ import annotations
import argparse, json, os, sys, time
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODE = os.environ.get("VLLM_DIAG_MODE", "jacobi")  # ar | jacobi


# ---- Patch (labeled with profiler ranges) ----
if MODE == "jacobi":
    import numpy as np
    import torch
    from torch.profiler import record_function

    _LAST_TARGET_ARGMAX: list = []
    _LAST_N_ACC: list = []

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
            with record_function("JF_PROPOSE"):
                K = self.k
                stash_id = id(_LAST_TARGET_ARGMAX)
                if stash_id != self._stash_id:
                    self._req_idx = 0; self._stash_id = stash_id
                i = self._req_idx
                self._req_idx += 1
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

    def _patch_proposer():
        import vllm.v1.spec_decode.ngram_proposer as ngm
        import vllm.v1.worker.gpu_model_runner as gmr
        class Shim(JacobiProposer): pass
        ngm.NgramProposer = Shim
        gmr.NgramProposer = Shim

    def _patch_rs():
        from vllm.v1.sample.rejection_sampler import RejectionSampler, PLACEHOLDER_TOKEN_ID
        orig = RejectionSampler.forward
        def new_forward(self, metadata, draft_probs, target_logits, bonus_token_ids, sampling_metadata):
            global _LAST_TARGET_ARGMAX, _LAST_N_ACC
            with record_function("JF_RS_INNER"):
                out = orig(self, metadata, draft_probs, target_logits, bonus_token_ids, sampling_metadata)
            with record_function("JF_AM_GPU"):
                target_argmax_gpu = target_logits.argmax(dim=-1)
            with record_function("JF_AM_CPU"):
                target_argmax = target_argmax_gpu.cpu().numpy()
                out_cpu = out.cpu().numpy()
            with record_function("JF_AM_SPLIT"):
                nd = list(metadata.num_draft_tokens)
                per_req_a = []; per_req_n = []
                off = 0
                for i, k in enumerate(nd):
                    per_req_a.append(target_argmax[off:off+k].copy())
                    n_nonpad = int((out_cpu[i] != PLACEHOLDER_TOKEN_ID).sum())
                    per_req_n.append(max(0, n_nonpad - 1))
                    off += k
                _LAST_TARGET_ARGMAX = per_req_a
                _LAST_N_ACC = per_req_n
            return out
        RejectionSampler.forward = new_forward

    _patch_proposer()
    _patch_rs()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompts_jsonl", required=True)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--max_num_seqs", type=int, default=1)
    p.add_argument("--n_prompts", type=int, default=8)
    p.add_argument("--gpu_mem_util", type=float, default=0.6)
    args = p.parse_args()

    from vllm import LLM, SamplingParams
    import torch

    prompts_raw = [json.loads(l) for l in open(args.prompts_jsonl)][:args.n_prompts]
    K = int(os.environ.get("JACOBI_K", "32"))
    kw = dict(model=args.model, dtype="bfloat16",
              max_model_len=4096, gpu_memory_utilization=args.gpu_mem_util,
              enforce_eager=False, max_num_seqs=args.max_num_seqs)
    if MODE == "jacobi":
        kw["speculative_config"] = {
            "method": "ngram", "num_speculative_tokens": K,
            "prompt_lookup_min": 2, "prompt_lookup_max": 4,
        }
    t0 = time.time()
    llm = LLM(**kw)
    print(f"[diag] load_s={time.time()-t0:.1f}", flush=True)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    chat_texts = [
        tok.apply_chat_template([{"role": "user", "content": p["input"]}],
                                tokenize=False, add_generation_prompt=True)
        for p in prompts_raw
    ]
    # warmup
    _ = llm.generate(chat_texts[:1], SamplingParams(max_tokens=16), use_tqdm=False)

    out_dir = os.environ.get("PROF_TRACE_DIR", "/tmp/jacobi_traces")
    os.makedirs(out_dir, exist_ok=True)
    trace_path = f"{out_dir}/{MODE}_K{K}_bs{args.max_num_seqs}.pt.trace.json"

    # Profile a SHORTER bench so trace file stays small but covers steady-state
    short_max_tok = 96
    sp2 = SamplingParams(temperature=0.0, max_tokens=short_max_tok)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False, with_stack=False, profile_memory=False,
    ) as prof:
        t0 = time.time()
        outs = llm.generate(chat_texts, sp2, use_tqdm=False)
        dt = time.time() - t0
    n_tok = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"[diag] short_wall={dt:.2f}s tokens={n_tok} TPS={n_tok/dt:.1f}", flush=True)

    # Aggregate by name
    ev = prof.key_averages()
    # The names of interest
    names = ["Preprocess", "Forward", "Postprocess", "Sample", "Bookkeep",
             "Draft", "EPLB",
             "JF_PROPOSE", "JF_RS_INNER", "JF_AM_GPU", "JF_AM_CPU", "JF_AM_SPLIT"]
    rows = []
    for e in ev:
        if e.key in names:
            # In recent torch versions, cpu_time_total / cuda_time_total are in microseconds.
            rows.append({
                "name": e.key,
                "count": e.count,
                "cpu_time_us": getattr(e, "cpu_time_total", 0),
                "cuda_time_us": getattr(e, "cuda_time_total",
                                        getattr(e, "device_time_total", 0)),
            })
    print(f"[diag-phase] mode={MODE} K={K} BS={args.max_num_seqs}", flush=True)
    print(json.dumps(rows, indent=2), flush=True)

    # Save raw trace too
    try:
        prof.export_chrome_trace(trace_path)
        print(f"[diag] trace_saved {trace_path}", flush=True)
    except Exception as e:
        print(f"[diag] trace_save_err {e}", flush=True)


if __name__ == "__main__":
    main()
