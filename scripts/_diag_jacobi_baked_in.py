"""Test FIX A + FIX D + FIX E baked into vllm core (no CompilationConfig override).

Runs vLLM with the Jacobi monkey-patch from vllm_jacobi_patch.py (so the
proposer + RS wrap are active) but DOES NOT pass cudagraph_capture_sizes or
cudagraph_mode. The fixes in vllm/config/__init__.py and gpu_model_runner.py
should auto-apply: cg_mode=FULL, capture_sizes ⊇ K+1=33, greedy bonus fast path.

Expected vs baseline (no fixes):
  TPS at BS=1 K=32 max_new=128: 297 -> ~470
  TPS at BS=1 K=32 max_new=512: ~? -> ~640
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

os.environ["VLLM_PLUGINS"] = ""
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---- minimal Jacobi monkey-patch (no CompilationConfig touching) ----
_LAST_TARGET_ARGMAX = []
_LAST_N_ACC = []

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
        K = self.k
        sid = id(_LAST_TARGET_ARGMAX)
        if sid != self._stash_id:
            self._req_idx = 0; self._stash_id = sid
        i = self._req_idx; self._req_idx += 1
        cold = (i >= len(_LAST_TARGET_ARGMAX) or len(_LAST_TARGET_ARGMAX[i]) != K)
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
    import vllm.v1.spec_decode.ngram_proposer as ngm
    import vllm.v1.worker.gpu_model_runner as gmr
    class Shim(JacobiProposer): pass
    ngm.NgramProposer = Shim
    gmr.NgramProposer = Shim
    from vllm.v1.sample.rejection_sampler import RejectionSampler, PLACEHOLDER_TOKEN_ID
    orig_rs = RejectionSampler.forward
    def new_forward(self, metadata, draft_probs, target_logits, bonus_token_ids, sampling_metadata):
        global _LAST_TARGET_ARGMAX, _LAST_N_ACC
        out = orig_rs(self, metadata, draft_probs, target_logits, bonus_token_ids, sampling_metadata)
        target_argmax = target_logits.argmax(dim=-1).cpu().numpy()
        out_cpu = out.cpu().numpy()
        nd = list(metadata.num_draft_tokens)
        per_a = []; per_n = []
        off = 0
        for i, k in enumerate(nd):
            per_a.append(target_argmax[off:off+k].copy())
            n_nonpad = int((out_cpu[i] != PLACEHOLDER_TOKEN_ID).sum())
            per_n.append(max(0, n_nonpad - 1))
            off += k
        _LAST_TARGET_ARGMAX = per_a; _LAST_N_ACC = per_n
        return out
    RejectionSampler.forward = new_forward
enable()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompts_jsonl", required=True)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--max_num_seqs", type=int, default=1)
    p.add_argument("--n_prompts", type=int, default=8)
    p.add_argument("--gpu_mem_util", type=float, default=0.6)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()

    from vllm import LLM, SamplingParams
    prompts_raw = [json.loads(l) for l in open(args.prompts_jsonl)][:args.n_prompts]
    K = int(os.environ.get("JACOBI_K", "32"))
    # NOTE: NO compilation_config override -> rely on the fixes baked into
    # vllm/config/__init__.py
    kw = dict(model=args.model, dtype="bfloat16",
              max_model_len=4096, gpu_memory_utilization=args.gpu_mem_util,
              enforce_eager=False, max_num_seqs=args.max_num_seqs,
              speculative_config={
                  "method": "ngram", "num_speculative_tokens": K,
                  "prompt_lookup_min": 2, "prompt_lookup_max": 4,
              })
    llm = LLM(**kw)
    # Verify the fixes auto-applied:
    cc = llm.llm_engine.vllm_config.compilation_config
    print(f"[diag-fixes] cudagraph_mode={cc.cudagraph_mode!r} "
          f"capture_sizes={cc.cudagraph_capture_sizes}", flush=True)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    chat_texts = [
        tok.apply_chat_template([{"role": "user", "content": p["input"]}],
                                tokenize=False, add_generation_prompt=True)
        for p in prompts_raw
    ]
    _ = llm.generate(chat_texts[:1], SamplingParams(max_tokens=16), use_tqdm=False)
    for r in range(args.repeats):
        t0 = time.time()
        outs = llm.generate(chat_texts, sp, use_tqdm=False)
        dt = time.time() - t0
        n_tok = sum(len(o.outputs[0].token_ids) for o in outs)
        print(f"[baked] r={r} max={args.max_new_tokens} wall={dt:.3f}s "
              f"tok={n_tok} TPS={n_tok/dt:.1f} ms/tok={1000*dt/n_tok:.3f}",
              flush=True)
    print(f"[baked] sample_out_0_first40={list(outs[0].outputs[0].token_ids[:40])}",
          flush=True)


if __name__ == "__main__":
    main()
