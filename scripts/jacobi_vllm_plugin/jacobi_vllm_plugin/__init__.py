"""vLLM general plugin: install Jacobi spec-decode patch on every vLLM process.

Triggered by vllm.plugins.load_general_plugins() which runs in BOTH parent
and subprocess. Set env JACOBI_K (default 16 — see below), JACOBI_TRAJ_PATH
(optional).

Default K=16 chosen for cons-RL training data quality: it captures the
windowed-Jacobi TPF ceiling (3.70 vs 3.26 for K=8) and produces stable
per-position prediction distributions at deep window positions (acceptance
@pos 7 = 38% vs 8% for K=8 — meaning the model is doing real bidirectional
refinement at K=16 that K=8 cuts off).

Throughput tradeoff at BS=128/GPU (training scale):
  K=8 : 10241 tok/s (1.19× vs AR)  — pure-throughput pick
  K=16: 7249  tok/s (0.86× vs AR)  — training-data-quality pick (DEFAULT)
  K=32: 4145  tok/s (0.49× vs AR)  — TPF saturated, compute waste

At smaller serving batch sizes (BS≤16), K=16 also wins on throughput
(BS=4: 2.61× vs AR; BS=16: 1.57× vs AR). Set JACOBI_K=8 explicitly only
when (a) BS≥128 and (b) you don't need the deep-position training signal.

Enable via: VLLM_PLUGINS=jacobi   (or leave unset to auto-load).
Skip via:   VLLM_PLUGINS=          (empty list).
"""
import os
from .patch import enable_jacobi_spec_decode, aggregate_trajectories  # noqa: F401


def install():
    K = int(os.environ.get("JACOBI_K", "16"))
    traj = os.environ.get("JACOBI_TRAJ_PATH") or None
    enable_jacobi_spec_decode(K=K, traj_path=traj)
    print(f"[jacobi_vllm_plugin] installed: K={K} traj={traj}", flush=True)
