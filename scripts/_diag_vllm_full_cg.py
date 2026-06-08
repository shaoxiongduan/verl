"""A/B: vLLM Jacobi BS=1 K=32 with cudagraph_mode={PIECEWISE, FULL, FULL_AND_PIECEWISE}.

PIECEWISE is the default; attention layers run eagerly outside the cuda graph.
FULL captures the entire forward as a single graph (no attention boundary).
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

os.environ["VLLM_PLUGINS"] = ""
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODE = os.environ.get("CG_MODE", "1")  # "1"=PIECEWISE, "2"=FULL, "3"=FULL_AND_PIECEWISE

# ---- standard JF patch ----
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
    cap = sorted({1, 2, 4, 8, 16, 32, K+1, 2*(K+1), 4*(K+1)})
    cg_mode_int = int(MODE)
    comp = {"level": 3, "use_inductor": True,
            "cudagraph_mode": cg_mode_int, "use_cudagraph": True,
            "cudagraph_capture_sizes": cap}
    # NOTE: full_cuda_graph is deprecated; cudagraph_mode handles it.
    kw = dict(model=args.model, dtype="bfloat16",
              max_model_len=4096, gpu_memory_utilization=args.gpu_mem_util,
              enforce_eager=False, max_num_seqs=args.max_num_seqs,
              compilation_config=comp,
              speculative_config={
                  "method": "ngram", "num_speculative_tokens": K,
                  "prompt_lookup_min": 2, "prompt_lookup_max": 4,
              })
    llm = LLM(**kw)
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
        print(f"[cg={MODE}] r={r} wall={dt:.3f}s tok={n_tok} "
              f"TPS={n_tok/dt:.1f} ms/tok={1000*dt/n_tok:.3f}", flush=True)


if __name__ == "__main__":
    main()
