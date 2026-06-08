#!/usr/bin/env python3
"""
End-to-end test of JacobiProposer in vLLM via runtime monkey-patch.

Strategy: vllm 0.10.2 doesn't have custom_class proposer dispatch. We
instead hijack the "ngram" method slot: instantiate JacobiProposer where
vllm would normally instantiate NgramProposer. Same `propose()` signature.

What this validates:
  - vllm can dispatch through our proposer
  - the proposer is invoked on real forwards
  - draft acceptance is plumbed through vllm's RejectionSampler correctly
  - the output token sequence is still valid

What this does NOT yet validate (needs MODE B core hook):
  - JF-parity refresh from target_argmax (we use prompt-tail drafts instead)
  - real Jacobi TPF — our drafts here are weak, so acceptance will be modest

Comparison baseline: vllm AR (no spec decode) on the same prompts.
"""
from __future__ import annotations
import os, sys, time, re, json
from pathlib import Path
import pyarrow.parquet as pq

# Import vllm before patching.
import vllm
from vllm import LLM, SamplingParams
from vllm.config import SpeculativeConfig
import vllm.v1.spec_decode.ngram_proposer as ngram_module

# Inject our JacobiProposer file into vllm's spec_decode subpackage.
# We can't `import` it normally because vllm 0.10.2 doesn't ship jacobi.py.
import importlib.util
JACOBI_PATH = "/mnt/weka/home/hao.zhang/shao/research/vllm/vllm/v1/spec_decode/jacobi.py"
spec = importlib.util.spec_from_file_location("vllm.v1.spec_decode.jacobi", JACOBI_PATH)
jacobi_module = importlib.util.module_from_spec(spec)
sys.modules["vllm.v1.spec_decode.jacobi"] = jacobi_module
spec.loader.exec_module(jacobi_module)
JacobiProposer = jacobi_module.JacobiProposer

# Hijack: replace NgramProposer in vllm's namespace.
_OrigNgram = ngram_module.NgramProposer

import numpy as np
import torch as _torch

# Module-global stash: written by patched RejectionSampler.forward,
# read by JacobiNgramShim.propose on the next iter.
#   _LAST_TARGET_ARGMAX_PER_REQ: list[np.ndarray] indexed by batch position
#   _LAST_NUM_ACCEPTED_PER_REQ:  list[int]
# Reset between batches.
_LAST_TARGET_ARGMAX_PER_REQ: list = []
_LAST_NUM_ACCEPTED_PER_REQ: list = []
_N_FWD: int = 0
_N_TOK_ACCEPTED: int = 0

# Trajectory log: written by subprocess hook → consumed by parent after gen.
# JSONL: one line per spec-decode iter, all reqs at that iter together.
# Schema per line:
#   {"iter": int, "num_draft": [int per req], "n_acc": [int per req],
#    "draft": [[K ints per req]], "target_argmax": [[K ints per req]],
#    "bonus": [int per req]}
TRAJ_PATH = os.environ.get("TRAJ_PATH", "/tmp/vllm_jacobi_traj.jsonl")
_TRAJ_FILE = None  # opened lazily in subprocess

def _traj_open():
    global _TRAJ_FILE
    if _TRAJ_FILE is None:
        # Use process-PID-tagged file because vLLM spawns subprocesses and we
        # may have multiple writers. Parent aggregates after run.
        pid = os.getpid()
        path = TRAJ_PATH + f".{pid}"
        _TRAJ_FILE = open(path, "w")
        print(f"[TRAJ-OPEN] pid={pid} path={path} exists_after={os.path.exists(path)}", flush=True)
    return _TRAJ_FILE

# Per-request running drafts (subprocess-side): we capture the drafts BEFORE
# forward via the proposer call, then pair them with target_argmax after
# forward via the rejection_sampler call.
_PENDING_DRAFT_PER_REQ: dict = {}  # batch_idx -> last draft handed back from propose()

def _patch_rejection_sampler():
    """Hook RejectionSampler.forward to expose target_argmax per request."""
    import json
    from vllm.v1.sample.rejection_sampler import RejectionSampler, PLACEHOLDER_TOKEN_ID
    globals()['json'] = json  # ensure visible inside new_forward closure
    orig_forward = RejectionSampler.forward

    def new_forward(self, metadata, draft_probs, target_logits, bonus_token_ids, sampling_metadata):
        out = orig_forward(self, metadata, draft_probs, target_logits, bonus_token_ids, sampling_metadata)
        # Compute target_argmax over speculative positions [num_tokens, vocab]
        target_argmax = target_logits.argmax(dim=-1).cpu().numpy()
        # num_draft_tokens[i] = K_i positions for request i
        cu = metadata.cu_num_draft_tokens.cpu().numpy() if hasattr(metadata.cu_num_draft_tokens, 'cpu') else metadata.cu_num_draft_tokens
        nd = list(metadata.num_draft_tokens)
        # out: [batch_size, max_spec_len + 1]; count non-placeholder per row = num_accepted+1 (bonus)
        out_cpu = out.cpu().numpy() if hasattr(out, 'cpu') else out
        per_req_argmax = []
        per_req_n_acc = []
        offset = 0
        for i, k in enumerate(nd):
            per_req_argmax.append(target_argmax[offset:offset+k].copy())
            # num accepted = count of non-PLACEHOLDER tokens minus the bonus
            n_nonpad = int((out_cpu[i] != PLACEHOLDER_TOKEN_ID).sum())
            per_req_n_acc.append(max(0, n_nonpad - 1))
            offset += k
        global _LAST_TARGET_ARGMAX_PER_REQ, _LAST_NUM_ACCEPTED_PER_REQ, _N_FWD, _N_TOK_ACCEPTED
        _LAST_TARGET_ARGMAX_PER_REQ = per_req_argmax
        _LAST_NUM_ACCEPTED_PER_REQ = per_req_n_acc
        _N_FWD += 1
        _N_TOK_ACCEPTED += sum(per_req_n_acc) + len(per_req_n_acc)

        # ---- trajectory recording (for cons-RL training) ----
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
        rec = {
            "iter": int(_N_FWD),
            "num_draft": [int(x) for x in nd],
            "n_acc": [int(x) for x in per_req_n_acc],
            "draft": [[int(x) for x in d] for d in draft_list],
            "target_argmax": [a.tolist() if hasattr(a, 'tolist') else list(a) for a in per_req_argmax],
            "bonus": [int(x) for x in bonus_list],
        }
        try:
            f = _traj_open()
            f.write(json.dumps(rec) + "\n")
            f.flush()
        except Exception as e:
            if _N_FWD <= 5:
                print(f"[TRAJ-ERR] iter#{_N_FWD}: {type(e).__name__}: {e}", flush=True)
        if _N_FWD <= 3 or _N_FWD % 100 == 0:
            pid = os.getpid()
            real_path = TRAJ_PATH + f".{pid}"
            print(f"[TRAJ-OK] iter#{_N_FWD} path={real_path} exists={os.path.exists(real_path)} size={os.path.getsize(real_path) if os.path.exists(real_path) else 'NA'}", flush=True)
        return out

    RejectionSampler.forward = new_forward
    print("[PATCH] RejectionSampler.forward hooked for target_argmax stash", flush=True)

_patch_rejection_sampler()


class JacobiNgramShim(JacobiProposer):
    """
    MODE B: use stashed target_argmax from previous iter's rejection sampler
    to build the next iter's draft (JF semantics).

    propose(token_ids_cpu_slice) is called PER request in vllm 0.10.2.
    We track which call index corresponds to which batch index via a counter
    that resets when the stashed list changes length.
    """
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._req_idx = 0
        self._stash_id = -1
        self._rng_np = np.random.default_rng(0)

    def propose(self, token_ids_slice):
        result = self._propose_inner(token_ids_slice)
        # Stash for trajectory logging (key by batch index used in _req_idx).
        _PENDING_DRAFT_PER_REQ[self._req_idx - 1] = [int(x) for x in result]
        return result

    def _propose_inner(self, token_ids_slice):
        K = self.k
        global _LAST_TARGET_ARGMAX_PER_REQ, _LAST_NUM_ACCEPTED_PER_REQ, _PENDING_DRAFT_PER_REQ
        stash_id = id(_LAST_TARGET_ARGMAX_PER_REQ)
        if stash_id != self._stash_id:
            self._req_idx = 0
            self._stash_id = stash_id
            _PENDING_DRAFT_PER_REQ.clear()

        i = self._req_idx
        self._req_idx += 1

        # Cold start: no prior stash for this req
        if i >= len(_LAST_TARGET_ARGMAX_PER_REQ) or len(_LAST_TARGET_ARGMAX_PER_REQ[i]) != K:
            # MODE A fallback: last K of prompt
            n = int(len(token_ids_slice))
            tail_len = min(K, n)
            tail = token_ids_slice[n-tail_len:n].astype(np.int64)
            if tail_len < K:
                pad = self._rng_np.integers(0, self.vocab_size, size=K-tail_len, dtype=np.int64)
                tail = np.concatenate([tail, pad])
            return tail

        argmax_prev = _LAST_TARGET_ARGMAX_PER_REQ[i]
        n_acc = _LAST_NUM_ACCEPTED_PER_REQ[i]
        # vLLM commits target_argmax[n_acc] as the bonus (new seed). So the
        # "useful" predictions remaining are target_argmax[n_acc+1 : K], which
        # are conditioned on the (wrong) rejected draft. We use them anyway —
        # they may match if the model is robust to small perturbations. Pad
        # with last-known token to fill K slots (avoids random which has 0
        # match probability).
        start = n_acc + 1
        if start >= K:
            # No "useful" predictions left — fall back to MODE A (prompt tail)
            n = int(len(token_ids_slice))
            tail_len = min(K, n)
            tail = token_ids_slice[n-tail_len:n].astype(np.int64)
            if tail_len < K:
                pad = self._rng_np.integers(0, self.vocab_size, size=K-tail_len, dtype=np.int64)
                tail = np.concatenate([tail, pad])
            return tail
        keep = argmax_prev[start:].astype(np.int64)
        # Pad with repeats of last useful prediction (not random) so the
        # remaining slots at least don't kill cascades.
        pad_len = K - len(keep)
        if pad_len > 0:
            pad_token = int(keep[-1]) if len(keep) > 0 else int(token_ids_slice[-1])
            pad = np.full(pad_len, pad_token, dtype=np.int64)
            keep = np.concatenate([keep, pad])
        return keep

# Make isinstance(x, NgramProposer) pass for our shim:
ngram_module.NgramProposer = JacobiNgramShim

# Also patch the import in the model_runner module (it does `from ... import NgramProposer`)
import vllm.v1.worker.gpu_model_runner as gmr
gmr.NgramProposer = JacobiNgramShim

print(f"[PATCH] vllm version: {vllm.__version__}")
print(f"[PATCH] JacobiProposer at: {JACOBI_PATH}")
print(f"[PATCH] NgramProposer -> JacobiNgramShim", flush=True)


# -----------------------------------------------------------------------
MODEL_PATH = "/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/math_k3_ds_step_300"
DATA_PATH = "/mnt/weka/home/hao.zhang/shao/verl/data/openmathinstruct2/val.parquet"
NUM_PROMPTS = int(os.environ.get("NUM_PROMPTS", "4"))
MAX_NEW = int(os.environ.get("MAX_NEW", "256"))
K = int(os.environ.get("JACOBI_K", "8"))  # spec window size

_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
def extract_boxed(s):
    if not s: return None
    m = _BOXED.findall(s)
    return m[-1].strip() if m else None

def grade(p, g):
    pb = extract_boxed(p)
    if pb is None: return False
    p_n = pb.replace(" ", "").replace("\\,", "").replace("$", "")
    g_n = str(g).replace(" ", "").replace("\\,", "").replace("$", "")
    if p_n == g_n: return True
    try: return float(p_n) == float(g_n)
    except: return False


def load_prompts(tokenizer, n):
    rows = pq.read_table(DATA_PATH).to_pylist()[:n]
    out = []
    for r in rows:
        text = tokenizer.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True)
        out.append({"text": text, "gt": r["reward_model"]["ground_truth"]})
    return out


def run(spec_method: str | None):
    print(f"\n{'='*60}\n{'JACOBI (via ngram hijack)' if spec_method else 'NO SPEC (vanilla AR)'}\n{'='*60}", flush=True)

    # Reset trajectory logs BEFORE LLM() spawns subprocess — otherwise we'd
    # delete the subprocess's open file mid-warmup.
    if spec_method:
        import glob
        for tp in glob.glob(TRAJ_PATH + ".*"):
            try: os.remove(tp)
            except FileNotFoundError: pass

    kw = dict(
        model=MODEL_PATH, dtype="bfloat16",
        max_model_len=4096, gpu_memory_utilization=0.6,
        enforce_eager=(os.environ.get("ENFORCE_EAGER","1")=="1"),
    )
    if spec_method == "ngram":
        kw["speculative_config"] = {
            "method": "ngram",
            "num_speculative_tokens": K,
            "prompt_lookup_min": 2,
            "prompt_lookup_max": 4,
        }

    t0 = time.time()
    llm = LLM(**kw)
    load_s = time.time() - t0
    print(f"[load] {load_s:.1f}s", flush=True)

    tok = llm.get_tokenizer()
    prompts = load_prompts(tok, NUM_PROMPTS)
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_NEW)

    global _N_FWD, _N_TOK_ACCEPTED
    _N_FWD = 0
    _N_TOK_ACCEPTED = 0
    t0 = time.time()
    outs = llm.generate([p["text"] for p in prompts], sp, use_tqdm=False)
    dt = time.time() - t0
    n_tok = sum(len(o.outputs[0].token_ids) for o in outs)
    n_correct = sum(1 for p, o in zip(prompts, outs) if grade(o.outputs[0].text, p["gt"]))

    # Read trajectory log (written by subprocess) and compute TPF stats.
    agg_tpf = None
    per_req_tpf = []
    n_spec_iters = 0
    if spec_method:
        import json as _json, glob as _glob
        spec_tok_total = 0
        per_req_tokens = {}    # batch_idx -> total tokens
        per_req_iters = {}     # batch_idx -> num iters this req participated
        # Aggregate across per-PID trajectory files written by subprocesses.
        traj_files = _glob.glob(TRAJ_PATH + ".*")
        if not traj_files:
            print(f"[WARN] no trajectory files found at {TRAJ_PATH}.*", flush=True)
        all_lines = []
        for tp in traj_files:
            with open(tp) as f:
                all_lines.extend(f.readlines())
        for line in all_lines:
                rec = _json.loads(line)
                # skip warmup phase: huge batches of num_draft=1
                if len(rec["num_draft"]) > len(prompts) * 4:
                    continue
                n_spec_iters += 1
                for i, (nd, na) in enumerate(zip(rec["num_draft"], rec["n_acc"])):
                    if nd <= 0:
                        continue
                    committed = na + 1  # accepted spec + bonus
                    spec_tok_total += committed
                    per_req_tokens[i] = per_req_tokens.get(i, 0) + committed
                    per_req_iters[i] = per_req_iters.get(i, 0) + 1
        agg_tpf = (spec_tok_total / n_spec_iters) if n_spec_iters > 0 else float("nan")
        for i in sorted(per_req_tokens.keys()):
            t = per_req_tokens[i]
            it = per_req_iters[i]
            per_req_tpf.append(t / it if it > 0 else float("nan"))

    tpf_str = f"agg_tpf={agg_tpf:.2f}" if agg_tpf is not None else "agg_tpf=N/A"
    per_req_str = ""
    if per_req_tpf:
        avg_per_req = sum(per_req_tpf)/len(per_req_tpf)
        per_req_str = f"  per_req_tpf={[round(x,2) for x in per_req_tpf]}  avg_per_req={avg_per_req:.2f}"
    print(f"[gen] {dt:.2f}s  tokens={n_tok}  tok/s={n_tok/dt:.1f}  acc={n_correct}/{len(prompts)}  "
          f"spec_iters={n_spec_iters}  {tpf_str}{per_req_str}", flush=True)
    return {"wall": dt, "tokens": n_tok, "tps": n_tok/dt, "acc": n_correct/len(prompts),
            "n_spec_iters": n_spec_iters, "agg_tpf": agg_tpf,
            "per_req_tpf": per_req_tpf,
            "traj_path": TRAJ_PATH if spec_method else None}


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "spec"
    if mode == "spec":
        r = run("ngram")  # hijacked to Jacobi
    else:
        r = run(None)
    print(f"\nresult: {r}")
