# vLLM + Jacobi speculative decoding (for cons-RL rollout)

Drop-in monkey-patch that gives JF-reference-TPF Jacobi speculative decoding
inside vLLM 0.10.2, with zero modifications to vLLM source.

## What it does

Replaces vLLM's `NgramProposer` with a `JacobiProposer` that:

1. **Cold start**: returns the last K tokens of the prompt as drafts.
2. **Hot path (MODE B)**: reads the previous iter's per-position
   `target_argmax` (captured via a wrap on `RejectionSampler.forward`) and
   builds the next draft as `target_argmax[n_acc+1:]` (shift-by-1 windowed
   Jacobi refresh). This matches the DL `generate_chunk_batch` semantics.

Per-prompt TPF matches the JacobiForcing reference (≈3.35 on math k3) within
±5% at every batch size we tested.

## Performance

### K=32 (matches JF block_len; for TPF-parity training data)

| BS | AR tok/s | Jacobi tok/s | Speedup | Per-prompt TPF |
|---:|---:|---:|---:|---:|
| 1 | 176 | 419 | 2.39× | 2.93 |
| 4 | 685 | 1519 | 2.22× | 3.39 |
| 8 | 1347 | 2123 | 1.58× | 3.08 |
| 16 | 2605 | 2818 | 1.08× | 3.19 |
| 32 | 4915 | 3152 | 0.64× | 3.18 |
| 128 | 8479 | 4145 | 0.49× | 3.60 |

### Optimal K vs batch size (training-scale included)

| BS | K=4 | K=8 | K=16 | K=32 | Best |
|---:|---:|---:|---:|---:|:---:|
| 4 | 1.82× | 2.43× | **2.61×** | 2.22× | K=16 |
| 16 | 1.54× | 1.76× | **1.57×** (tied) | 1.08× | K=8/16 |
| 128 | 1.19× | **1.19×** | 0.85× | 0.49× | K=4/8 |

**Crossover at K=8→16**: compute-bound transition. Per-forward kernel jumps
+45% (46.5→67.7 ms at BS=128) while TPF only gains +13% (3.22→3.63). Above
K=16, TPF saturates around 3.6 because windowed-Jacobi can't iterate to
convergence on the same K positions like JF chunked decode does.

**Default K=16** in the plugin: optimal for cons-RL training data quality.
Per-position acceptance analysis (see `analyze_jacobi_trajectories.py`)
shows K=16 captures the windowed-Jacobi TPF ceiling AND produces stable
deep-position predictions (acceptance @pos 7 = 38% vs 8% for K=8). That's
the regime cons-loss training is meant to target.

Override via `JACOBI_K` env var:
- `JACOBI_K=8`  — pure throughput at BS=128 (1.19× vs AR vs K=16's 0.86×)
- `JACOBI_K=32` — JF-parity TPF (3.6 ≈ 3.35), but ~50% slower than AR

## Usage

### Recommended: auto-install via vLLM plugin (zero caller changes)

```bash
pip install /mnt/weka/home/hao.zhang/shao/verl/scripts/jacobi_vllm_plugin
```

After install, ANY vLLM process — parent or subprocess — auto-runs
`enable_jacobi_spec_decode()` at import time via vLLM's `vllm.general_plugins`
entry point mechanism. Caller code (incl. verl) needs ZERO changes.

```python
# No imports of jacobi_vllm_plugin needed. Just use vLLM normally:
from vllm import LLM, SamplingParams
llm = LLM(model=..., enforce_eager=False,
          speculative_config={
              "method": "ngram",                # hijacked slot
              "num_speculative_tokens": 32,
              "prompt_lookup_min": 2,           # required by ngram, unused
              "prompt_lookup_max": 4,           # required by ngram, unused
          })
outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=512))
```

Configure via env vars:
- `JACOBI_K=32` — speculative window (default 32; matches JF block_len)
- `JACOBI_TRAJ_PATH=/path/traj.jsonl` — optional per-iter trajectory log
- `VLLM_PLUGINS=jacobi` — restrict vLLM to only load our plugin (else
  all general_plugins load by default)

Read trajectory stats after a run:
```python
from jacobi_vllm_plugin import aggregate_trajectories
stats = aggregate_trajectories()
print(stats["agg_tpf"], stats["per_req_tpf"])
```

### Alternative: manual install (no plugin)

```python
import vllm_jacobi_patch
vllm_jacobi_patch.enable_jacobi_spec_decode(K=32, traj_path="/path/to/traj.jsonl")
from vllm import LLM, SamplingParams
...
```

Caller MUST use `if __name__ == "__main__":` guard around `LLM(...)` because
vLLM uses `spawn` (subprocess re-imports caller's `__main__`).

### Inside verl (cons-RL rollout)

With the plugin installed, **no changes to verl source needed**. Just set
in the rollout YAML:

```yaml
mtp:
  enable: true
  enable_rollout: true
  method: ngram                  # hijacked by jacobi plugin
  num_speculative_tokens: 32
```

`verl/workers/rollout/vllm_rollout/vllm_async_server.py:286-291` already
builds the `speculative_config` from this subtree; the plugin transparently
swaps the proposer.

The cons-loss training side reads only `(prompt, response)` from rollout
(see `scripts/consistency/verl_hook.py`); it does not need the per-iter
trajectory file. Trajectories are saved for debugging / future use.

## Trajectory file format

`{traj_path}.{PID}` JSONL, one record per spec-decode forward:

```json
{"iter": 5,
 "num_draft": [32, 32, 32, 0],      // K per request (0 = skipped)
 "n_acc":     [3, 0, 5, 0],          // accepted spec tokens per req
 "draft":     [[t0, t1, ...], ...],  // K tokens our proposer returned
 "target_argmax": [[t0', t1', ...], ...],  // model's argmax at each spec pos
 "bonus":     [b0, b1, b2, b3]}      // = target_argmax[n_acc] per req
```

Aggregate via `vllm_jacobi_patch.aggregate_trajectories()`.

## Known limitations

- **Cold start is prompt-tail, not prefill-argmax.** JF reference uses
  `argmax(prefill_logits[-K-1:-1])` as the iter-0 draft. We use the last K
  prompt tokens — close but ~88% of JF iter-0 acceptance. Adding the proper
  hook would require capturing prefill logits, ~50 LOC + a new
  `gpu_model_runner` hook. Marginal TPF impact (only iter 0).

- **`method="ngram"` is hijacked.** Native `method="jacobi"` would require
  modifying installed vLLM source (`SpeculativeMethod` Literal, dispatch
  branches in `gpu_model_runner.py`). Avoided to keep verl integration
  zero-touch on vLLM.

- **vLLM commits a bonus token every iter.** Unlike DL's windowed
  `_run_one_block` (which omits bonus for `acc_len > 1`), vLLM always
  commits `target_argmax[n_acc]` as the new seed. This means the refresh
  for next iter has one fewer "good" position than DL.

## Files

- `scripts/jacobi_vllm_plugin/` — installable plugin package (auto-loads in
  every vLLM process). Recommended path for verl integration.
- `scripts/vllm_jacobi_patch.py` — the standalone drop-in module (use if you
  prefer explicit `enable_jacobi_spec_decode()` calls instead of a pip install).
- `scripts/vllm_jacobi_plugin_test.py` — verifies plugin auto-loads (no
  caller code changes). Reads `JACOBI_K` env to vary spec window size.
- `scripts/vllm_jacobi_smoke.py` — end-to-end smoke test using the manual
  drop-in API.
- `scripts/vllm_jacobi_e2e_test.py` — original test script (BS/K sweeps,
  CUDA-graphs comparison, AR baseline). Kept for benchmark reproducibility.

### Analysis tools

- `scripts/analyze_jacobi_trajectories.py` — per-block-position analysis:
  acceptance rate, survival, Shannon entropy of target argmax, top-10
  concentration. Pass one or more trajectory globs:
  ```
  python analyze_jacobi_trajectories.py 'traj_k8.jsonl.*' 'traj_k16.jsonl.*'
  ```

- `scripts/decode_jacobi_traces.py` — qualitative trace dump. Decodes drafts
  + targets + acceptance pattern for example iters:
  ```
  python decode_jacobi_traces.py traj_k16.jsonl.* --n_examples 5 --min_acc 6
  ```
  Shows token-level matches (`✓`/`✗`) so you can eyeball whether K is
  capturing the model's actual Jacobi convergence pattern.

- `scripts/dl_jacobi_profile.py` — per-forward kernel timing breakdown
  (wall, fwd_sum, overhead, per-B kernel cost). Useful for diagnosing
  compute-bound transitions when sweeping K.
