"""Assert-style code reward for verl's experimental DAPO reward loop.

Per-sample entry point: `compute_score(data_source, solution_str,
ground_truth, extra_info, **kwargs)`. The model's Python code is extracted
from the first ```python ... ``` block (or the raw string), concatenated
with the test harness from `ground_truth`, and executed in a sandboxed
subprocess under a wall-clock + CPU limit. Reward is 1.0 if all tests pass,
0.0 otherwise.

The experimental `RewardLoopWorker` (verl/experimental/reward_loop/) calls
this function via `asyncio.run_in_executor`, so the framework already gives
us batch concurrency (~36 default threads). No need for a custom batch
reward manager.

Overlong-response penalty is left to the DAPO reward manager via
`+reward.reward_kwargs.overlong_buffer_cfg.*` — same as the math run.

WARNING — this executes untrusted model output locally. Acceptable for an
initial pilot inside an isolated training node; for production runs, swap
this for `reward_model.sandbox_fusion.url=...` against an actual sandbox
server. See `verl/utils/reward_score/sandbox_fusion/`.
"""

from __future__ import annotations

import json
import os
import re
import resource
import subprocess
import sys
import tempfile
import textwrap
from typing import Any

CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_python(solution_str: str) -> str:
    matches = CODE_FENCE_RE.findall(solution_str)
    if matches:
        # Prefer the last fenced block — models often re-state the final answer
        return matches[-1].strip()
    # Fallback: raw string. The exec sandbox will simply fail if it's not code.
    return solution_str.strip()


def _set_subprocess_limits(memory_mb: int | None = None, cpu_seconds: int = 12) -> None:
    """preexec_fn for subprocess — caps CPU. Memory limit is off by default
    because RLIMIT_AS conflicts with numpy/torch shared-lib mappings and
    causes spurious failures. Subprocess wall-clock timeout is the real
    safety net; RLIMIT_CPU catches multi-threaded busy loops."""
    if memory_mb is not None:
        try:
            resource.setrlimit(
                resource.RLIMIT_AS, (memory_mb * 1024 * 1024, memory_mb * 1024 * 1024)
            )
        except (ValueError, OSError):
            pass
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    except (ValueError, OSError):
        pass
    os.setsid()


def _run_one(args: tuple[str, str, int, int]) -> float:
    """Worker: assemble code+tests, run in subprocess, return 1.0 / 0.0."""
    solution_str, ground_truth_str, timeout_s, memory_mb = args
    try:
        gt = json.loads(ground_truth_str)
    except Exception:
        return 0.0
    test_code = gt.get("test_code", "")
    candidate = extract_python(solution_str)
    if not candidate or not test_code:
        return 0.0

    program = candidate + "\n\n" + test_code + "\n"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tf:
        tf.write(program)
        path = tf.name
    try:
        try:
            # NOTE: not using `-I` (isolated mode) because HumanEval+ test
            # harnesses commonly `import numpy`, which requires user site-packages.
            # cwd=sandbox: OpenCodeInstruct prompts often ask the model to write
            # CSV/JSON/SQLite files; without an isolated cwd those land in the
            # parent's cwd (the verl repo root) and pollute it.
            with tempfile.TemporaryDirectory(prefix="reward_sandbox_") as sandbox:
                result = subprocess.run(
                    [sys.executable, path],
                    capture_output=True,
                    timeout=timeout_s,
                    cwd=sandbox,
                    preexec_fn=lambda: _set_subprocess_limits(memory_mb=memory_mb, cpu_seconds=timeout_s + 2),
                )
            return 1.0 if result.returncode == 0 else 0.0
        except subprocess.TimeoutExpired:
            return 0.0
        except Exception:
            return 0.0
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    *,
    timeout_s: int = 10,
    memory_mb: int | None = None,
    **_: Any,
) -> dict[str, float]:
    """Per-sample entry point used by the experimental DAPO reward manager.

    Returns a dict with at least {"score": float, "acc": float}. The DAPO
    manager will apply its overlong-buffer penalty on top of `score`.
    """
    acc = _run_one((solution_str, ground_truth, timeout_s, memory_mb))
    return {"score": float(acc), "acc": float(acc)}


# Sanity check when run directly.
if __name__ == "__main__":
    gt = json.dumps(
        {
            "test_code": textwrap.dedent(
                """
                assert add(1, 2) == 3
                assert add(0, 0) == 0
                """
            ),
            "entry_point": "add",
            "language": "python",
        }
    )
    for sol in [
        "```python\ndef add(a, b):\n    return a + b\n```",
        "```python\ndef add(a, b):\n    return a - b\n```",
        "broken not python",
    ]:
        print(compute_score("opencodeinstruct", sol, gt, {}, timeout_s=5), "::", repr(sol[:40]))
