"""Unit tests for the v11.1 canvas changes (empirical construction +
CE-to-rollout canvas loss + canvas weight multiplier). CPU-only.

Also re-run scripts/test_canvas_pairs.py afterwards — the v11.0 golden
regression must keep passing (all v11.1 paths are env-gated).

Run:  python scripts/test_canvas_v11p1.py
"""
import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from consistency.pack import build_interleaved_batch  # noqa: E402
from consistency.loss import compute_consistency_loss  # noqa: E402

N = 8
VOCAB = 152064
V11P1_KEYS = (
    "CONSISTENCY_CANVAS_FRAC", "CONSISTENCY_CANVAS_LEVELS",
    "CONSISTENCY_CANVAS_PLAUSIBLE_FRAC", "CONSISTENCY_CANVAS_CONSTRUCTION",
    "CONSISTENCY_CANVAS_LEVELS_FRAC", "CONSISTENCY_CANVAS_P_NEAR",
    "CONSISTENCY_CANVAS_P_FAR", "CONSISTENCY_CANVAS_MAX_COMMIT",
    "CONSISTENCY_CANVAS_LOSS", "CONSISTENCY_CANVAS_WEIGHT_MULT",
    "CONSISTENCY_USE_DRAFT_MARKER", "CONSISTENCY_MARKER_TYPE",
)


def _reset_env():
    for k in V11P1_KEYS:
        os.environ.pop(k, None)
    os.environ["CONSISTENCY_VOCAB_SIZE"] = str(VOCAB)
    os.environ["CONSISTENCY_NOISE_SOURCE"] = "uniform"
    os.environ["CONSISTENCY_LOSS_TYPE"] = "kl"
    os.environ["CONSISTENCY_KL_DECAY"] = "1"


def _cascade(pool_base=70000):
    """Two trajectories whose drafts (= alt pools) live in a distinctive
    token range [pool_base, pool_base+16) — disjoint from clean (100..131)
    and statistically disjoint from uniform noise."""
    d0 = tuple(range(pool_base, pool_base + 8))
    d1 = tuple(range(pool_base + 8, pool_base + 16))
    return [[
        SimpleNamespace(start=0, end=8, draft=d0, prefix_match_len=3),
        SimpleNamespace(start=8, end=16, draft=d1, prefix_match_len=1),
    ]]


def test_empirical_construction():
    _reset_env()
    os.environ["CONSISTENCY_CANVAS_FRAC"] = "1.0"
    os.environ["CONSISTENCY_CANVAS_CONSTRUCTION"] = "empirical"
    os.environ["CONSISTENCY_CANVAS_LEVELS_FRAC"] = "0.0"
    os.environ["CONSISTENCY_CANVAS_MAX_COMMIT"] = "3"
    prompts = [torch.tensor([1, 2, 3])]
    responses = [torch.arange(100, 132)]
    pool_vals = set(range(70000, 70016))

    # P_NEAR=1, P_FAR=1 -> body all clean; tail (<=3) noise.
    os.environ["CONSISTENCY_CANVAS_P_NEAR"] = "1.0"
    os.environ["CONSISTENCY_CANVAS_P_FAR"] = "1.0"
    n_tail_seen = []
    for s in range(8):
        b = build_interleaved_batch(prompts, responses, block_size=N, pad_id=0,
                                    generator=torch.Generator().manual_seed(s),
                                    cascade_drafts=_cascade())
        for j in range(2):
            ks = int(b.noisy_starts[0, j].item())
            ls = int(b.clean_starts[0, j].item())
            tile = b.input_ids[0, ks:ks + N]
            clean = b.input_ids[0, ls:ls + N]
            diff = (tile != clean).nonzero().flatten().tolist()
            # all diffs must be a contiguous tail of length <= MAX_COMMIT
            assert len(diff) <= 3, f"tail too long: {diff}"
            if diff:
                assert diff == list(range(N - len(diff), N)), f"non-tail diff: {diff}"
            n_tail_seen.append(len(diff))
            assert int(b.onpolicy_prefix_lens[0, j].item()) == 0
    assert max(n_tail_seen) > 0, "no noise tail ever drawn"
    print(f"[empirical p=1]      OK  tails seen: {sorted(set(n_tail_seen))}")

    # P_NEAR=0, P_FAR=0 -> body all wrong: from pools (or noise fallback).
    os.environ["CONSISTENCY_CANVAS_P_NEAR"] = "0.0001"
    os.environ["CONSISTENCY_CANVAS_P_FAR"] = "0.0001"
    n_pool = n_clean = n_body = 0
    for s in range(8):
        b = build_interleaved_batch(prompts, responses, block_size=N, pad_id=0,
                                    generator=torch.Generator().manual_seed(100 + s),
                                    cascade_drafts=_cascade())
        for j in range(2):
            ks = int(b.noisy_starts[0, j].item())
            ls = int(b.clean_starts[0, j].item())
            tile = b.input_ids[0, ks:ks + N].tolist()
            clean = b.input_ids[0, ls:ls + N].tolist()
            for t, c in zip(tile[: N - 3], clean[: N - 3]):  # body only
                n_body += 1
                n_pool += int(t in pool_vals)
                n_clean += int(t == c)
    assert n_clean <= 1, f"p~0 but {n_clean}/{n_body} body tokens clean"
    assert n_pool / n_body > 0.9, f"only {n_pool}/{n_body} body tokens pool-sourced"
    print(f"[empirical p=0]      OK  pool-sourced {n_pool}/{n_body}, clean {n_clean}")

    # Gradient: P_NEAR=0.9, P_FAR=0.1 -> near positions cleaner than far.
    os.environ["CONSISTENCY_CANVAS_P_NEAR"] = "0.9"
    os.environ["CONSISTENCY_CANVAS_P_FAR"] = "0.1"
    os.environ["CONSISTENCY_CANVAS_MAX_COMMIT"] = "0"
    clean_at = [0] * N
    cnt = 0
    for s in range(200):
        b = build_interleaved_batch(prompts, responses, block_size=N, pad_id=0,
                                    generator=torch.Generator().manual_seed(1000 + s),
                                    cascade_drafts=_cascade())
        for j in range(2):
            ks = int(b.noisy_starts[0, j].item())
            ls = int(b.clean_starts[0, j].item())
            tile = b.input_ids[0, ks:ks + N]
            cleanb = b.input_ids[0, ls:ls + N]
            for p in range(N):
                clean_at[p] += int(tile[p] == cleanb[p])
        cnt += 1
    first = clean_at[0] / (2 * cnt)
    last = clean_at[N - 1] / (2 * cnt)
    assert first > 0.8 and last < 0.25, f"gradient broken: first={first:.2f} last={last:.2f}"
    print(f"[empirical decay]    OK  P(clean) {first:.2f} -> {last:.2f}")


def test_levels_frac_mix():
    _reset_env()
    os.environ["CONSISTENCY_CANVAS_FRAC"] = "1.0"
    os.environ["CONSISTENCY_CANVAS_CONSTRUCTION"] = "empirical"
    os.environ["CONSISTENCY_CANVAS_LEVELS_FRAC"] = "1.0"  # all pairs -> f-mixture
    os.environ["CONSISTENCY_CANVAS_P_NEAR"] = "1.0"
    os.environ["CONSISTENCY_CANVAS_P_FAR"] = "1.0"
    prompts = [torch.tensor([1, 2, 3])]
    responses = [torch.arange(100, 132)]
    levels_n = {1, 2, 4, 6, 8}
    b = build_interleaved_batch(prompts, responses, block_size=N, pad_id=0,
                                generator=torch.Generator().manual_seed(5),
                                cascade_drafts=_cascade())
    for j in range(2):
        ks = int(b.noisy_starts[0, j].item())
        ls = int(b.clean_starts[0, j].item())
        nd = int((b.input_ids[0, ks:ks + N] != b.input_ids[0, ls:ls + N]).sum().item())
        assert nd in levels_n, f"levels_frac=1 pair has non-level diff count {nd}"
    print("[levels_frac mix]    OK  f-mixture path reachable under empirical mode")


def _tiny_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(1234)
    cfg = Qwen2Config(vocab_size=1000, hidden_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2,
                      intermediate_size=128, max_position_embeddings=2048,
                      attn_implementation="sdpa")
    return Qwen2ForCausalLM(cfg).eval()


def _split_loss(model, mult, seed=11):
    compute_consistency_loss._diag_dumped = True
    os.environ["CONSISTENCY_CANVAS_WEIGHT_MULT"] = str(mult)
    prompts = [torch.tensor([1, 2, 3, 4, 5])]
    responses = [torch.arange(20, 84) % 1000]
    loss, _, metrics = compute_consistency_loss(
        model=model, prompt_ids=prompts, response_ids=responses,
        block_size=N, pad_id=0, max_pairs=None, T_soft=1.0, seed=seed,
        device="cpu", divergence="forward_kl", teacher_model=None,
    )
    return loss, metrics


def test_ce_split_and_mult():
    _reset_env()
    os.environ["CONSISTENCY_VOCAB_SIZE"] = "1000"
    os.environ["CONSISTENCY_CANVAS_FRAC"] = "0.5"
    os.environ["CONSISTENCY_CANVAS_LOSS"] = "ce"
    os.environ["CONSISTENCY_USE_DRAFT_MARKER"] = "1"
    os.environ["CONSISTENCY_MARKER_TYPE"] = "constant"
    model = _tiny_model()
    model.train()
    l1, m1 = _split_loss(model, 1.0)
    l10, m10 = _split_loss(model, 10.0)
    assert "cons_causal_term" in m1 and "cons_canvas_term" in m1, m1.keys()
    ca, cv = m1["cons_causal_term"], m1["cons_canvas_term"]
    assert cv > 0 and ca > 0
    assert abs(float(l1.item()) - (ca + cv)) < 1e-4, (l1.item(), ca, cv)
    assert abs(float(l10.item()) - (ca + 10.0 * cv)) < 1e-3, (l10.item(), ca, cv)
    l10.backward()
    g = sum(float(p.grad.norm()) for p in model.parameters() if p.grad is not None)
    assert g > 0 and all(torch.isfinite(p.grad).all() for p in model.parameters()
                         if p.grad is not None)
    print(f"[ce split + mult]    OK  causal={ca:.4f} canvas={cv:.4f} "
          f"loss(1x)={float(l1):.4f} loss(10x)={float(l10):.4f}")


def test_ce_mode_no_canvas_falls_back():
    """CANVAS_LOSS=ce with zero canvas pairs must produce the legacy KL value."""
    _reset_env()
    os.environ["CONSISTENCY_VOCAB_SIZE"] = "1000"
    model = _tiny_model()
    # Same CANVAS_FRAC both runs -> identical pack RNG stream; only the loss
    # routing env differs. With zero canvas pairs flipped, ce mode must fall
    # through to the bit-identical legacy KL path.
    os.environ["CONSISTENCY_CANVAS_FRAC"] = "0.0001"  # on, ~never flips
    base_loss, _ = _split_loss(model, 1.0)  # CANVAS_LOSS unset
    os.environ["CONSISTENCY_CANVAS_LOSS"] = "ce"
    ce_loss, m = _split_loss(model, 10.0)
    assert m.get("cons_canvas_pos_frac", 0.0) == 0.0
    assert "cons_canvas_term" not in m
    assert abs(float(base_loss.item()) - float(ce_loss.item())) < 1e-6, \
        (float(base_loss.item()), float(ce_loss.item()))
    print(f"[ce fallback]        OK  no-canvas microbatch == legacy KL ({float(ce_loss):.6f})")


if __name__ == "__main__":
    test_empirical_construction()
    test_levels_frac_mix()
    test_ce_split_and_mult()
    test_ce_mode_no_canvas_falls_back()
    print("\nALL V11.1 TESTS PASSED")
