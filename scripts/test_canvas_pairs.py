"""Unit tests for the v11 canvas-pair machinery (pack/attention/loss).

CPU-only (tiny random Qwen2 for the forward tests) — safe on the login node,
style of `scripts/test_mask_tail_loss.py`.

Covers:
  1. HARD REGRESSION: with canvas mode off, batches/masks/loss values are
     bit-identical to the pre-v11 golden snapshot (scripts/.golden_v9_cons.pt,
     produced by scripts/_make_v9_golden.py on the pre-v11 working tree).
  2. canvas_pairs=all-False mask == canvas_pairs=None mask (both mask paths).
  3. Canvas tile construction: renoise counts match the level set; renoised
     positions are far-weighted; pad tail handled.
  4. On-policy: canvas pairs REPLACE the cascade draft and zero the pml;
     causal pairs keep draft + pml exactly.
  5. Per-pair attention: canvas tiles fully bidir within themselves, causal
     tiles strictly causal, no cross-tile leakage; uniform + variable paths.
  6. End-to-end tiny-model forward: mixed batch trains (finite loss + grads),
     canvas/causal metric split present.
  7. Marker restricted to canvas positions: with a canvas_mask present but
     all-False, marker on == marker off exactly.

Run:  python scripts/test_canvas_pairs.py
"""
import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from consistency.pack import build_interleaved_batch  # noqa: E402
from consistency.attention import build_sdpa_attention_mask  # noqa: E402
from consistency.loss import compute_consistency_loss  # noqa: E402

VOCAB = 1000
N = 8
GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".golden_v9_cons.pt")

CANVAS_KEYS = (
    "CONSISTENCY_CANVAS_FRAC", "CONSISTENCY_CANVAS_LEVELS",
    "CONSISTENCY_CANVAS_PLAUSIBLE_FRAC", "CONSISTENCY_USE_DRAFT_MARKER",
    "CONSISTENCY_MARKER_TYPE",
)


def _set_base_env():
    os.environ["CONSISTENCY_VOCAB_SIZE"] = str(VOCAB)
    os.environ["CONSISTENCY_NOISE_SOURCE"] = "uniform"
    os.environ["CONSISTENCY_LOSS_TYPE"] = "kl"
    os.environ["CONSISTENCY_KL_DECAY"] = "1"
    os.environ["CONSISTENCY_DFLASH_GAMMA"] = "12.0"
    for k in CANVAS_KEYS:
        os.environ.pop(k, None)


def tiny_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(1234)
    cfg = Qwen2Config(
        vocab_size=VOCAB, hidden_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, intermediate_size=128,
        max_position_embeddings=2048, attn_implementation="sdpa",
    )
    return Qwen2ForCausalLM(cfg).eval()


def _cascade():
    d0 = tuple(int(x) for x in (torch.arange(7, 15)).tolist())
    d1 = tuple(int(x) for x in (torch.arange(200, 208) % VOCAB).tolist())
    return [[
        SimpleNamespace(start=0, end=8, draft=d0, prefix_match_len=3),
        SimpleNamespace(start=12, end=20, draft=d1, prefix_match_len=1),  # gap -> bridge
    ]]


def _loss(model, cascade=False, marker=False, seed=7):
    compute_consistency_loss._diag_dumped = True
    if marker:
        os.environ["CONSISTENCY_USE_DRAFT_MARKER"] = "1"
        os.environ["CONSISTENCY_MARKER_TYPE"] = "constant"
    else:
        os.environ.pop("CONSISTENCY_USE_DRAFT_MARKER", None)
        os.environ.pop("CONSISTENCY_MARKER_TYPE", None)
    prompts = [torch.tensor([1, 2, 3, 4, 5])]
    responses = [torch.arange(20, 44) % VOCAB]
    loss, anchor, metrics = compute_consistency_loss(
        model=model, prompt_ids=prompts, response_ids=responses,
        block_size=N, pad_id=0, max_pairs=None, T_soft=1.0, seed=seed,
        device="cpu", divergence="forward_kl", teacher_model=None,
        cascade_drafts=_cascade() if cascade else None,
    )
    return loss, metrics


# ----------------------------------------------------------------------------
def test_golden_regression():
    """Canvas mode off → bit-identical to the pre-v11 snapshot."""
    _set_base_env()
    g = torch.load(GOLDEN, weights_only=False)

    gen = torch.Generator().manual_seed(42)
    prompts = [torch.tensor([1, 2, 3, 4, 5]), torch.tensor([10, 11, 12])]
    responses = [torch.arange(20, 44) % VOCAB, torch.arange(50, 62) % VOCAB]
    b_off = build_interleaved_batch(prompts, responses, block_size=N, pad_id=0,
                                    max_pairs=None, generator=gen)
    assert b_off.canvas_pairs is None and b_off.canvas_mask is None
    assert torch.equal(b_off.input_ids, g["off_input_ids"]), "off-policy input_ids drifted"
    assert torch.equal(b_off.position_ids, g["off_position_ids"])
    assert torch.equal(b_off.pad_mask, g["off_pad_mask"])
    assert torch.equal(b_off.noisy_mask, g["off_noisy_mask"])
    m_off = build_sdpa_attention_mask(
        b_off.prompt_lens, b_off.num_pairs, b_off.block_lens, b_off.pad_mask,
        device="cpu", dtype=torch.float32, causal_region_size=0,
    )
    assert torch.equal(m_off, g["off_sdpa_mask"]), "off-policy sdpa mask drifted"

    gen = torch.Generator().manual_seed(43)
    prompts = [torch.tensor([1, 2, 3])]
    responses = [torch.arange(100, 132) % VOCAB]
    b_on = build_interleaved_batch(prompts, responses, block_size=N, pad_id=0,
                                   max_pairs=None, generator=gen,
                                   cascade_drafts=_cascade())
    assert torch.equal(b_on.input_ids, g["on_input_ids"]), "on-policy input_ids drifted"
    assert torch.equal(b_on.position_ids, g["on_position_ids"])
    assert torch.equal(b_on.pad_mask, g["on_pad_mask"])
    assert torch.equal(b_on.noisy_mask, g["on_noisy_mask"])
    assert torch.equal(b_on.onpolicy_prefix_lens, g["on_prefix_lens"])
    m_on = build_sdpa_attention_mask(
        b_on.prompt_lens, b_on.num_pairs, b_on.block_lens, b_on.pad_mask,
        device="cpu", dtype=torch.float32, causal_region_size=0,
        role_per_pos=b_on.role_per_pos, pair_idx_per_pos=b_on.pair_idx_per_pos,
        noisy_starts=b_on.noisy_starts,
    )
    assert torch.equal(m_on, g["on_sdpa_mask"]), "on-policy sdpa mask drifted"

    model = tiny_model()
    l_off, _ = _loss(model, cascade=False, marker=False)
    l_on, met_on = _loss(model, cascade=True, marker=False)
    l_mk, _ = _loss(model, cascade=True, marker=True)
    for got, want, nm in (
        (float(l_off.item()), g["loss_offpolicy"], "loss_offpolicy"),
        (float(l_on.item()), g["loss_onpolicy"], "loss_onpolicy"),
        (float(l_mk.item()), g["loss_onpolicy_marker"], "loss_onpolicy_marker"),
    ):
        assert abs(got - want) < 1e-6, f"{nm}: {got} != golden {want}"
    assert abs(met_on.get("cons_argmax_correct", -1.0) - g["argmax_correct_onpolicy"]) < 1e-6
    print(f"[golden regression]   OK  (off={float(l_off):.6f} on={float(l_on):.6f} mk={float(l_mk):.6f})")


def test_allfalse_canvas_equals_none():
    _set_base_env()
    gen = torch.Generator().manual_seed(42)
    prompts = [torch.tensor([1, 2, 3, 4, 5])]
    responses = [torch.arange(20, 44) % VOCAB]
    b = build_interleaved_batch(prompts, responses, block_size=N, pad_id=0, generator=gen)
    cp_false = torch.zeros((1, int(b.num_pairs.max().item())), dtype=torch.bool)
    args = dict(device="cpu", dtype=torch.float32, causal_region_size=0)
    m_none = build_sdpa_attention_mask(b.prompt_lens, b.num_pairs, b.block_lens, b.pad_mask, **args)
    m_false = build_sdpa_attention_mask(b.prompt_lens, b.num_pairs, b.block_lens, b.pad_mask,
                                        canvas_pairs=cp_false, **args)
    assert torch.equal(m_none, m_false), "all-False canvas_pairs changed the uniform mask"

    b_on = build_interleaved_batch([torch.tensor([1, 2, 3])], [torch.arange(100, 132) % VOCAB],
                                   block_size=N, pad_id=0,
                                   generator=torch.Generator().manual_seed(43),
                                   cascade_drafts=_cascade())
    kw = dict(role_per_pos=b_on.role_per_pos, pair_idx_per_pos=b_on.pair_idx_per_pos,
              noisy_starts=b_on.noisy_starts)
    m_none = build_sdpa_attention_mask(b_on.prompt_lens, b_on.num_pairs, b_on.block_lens,
                                       b_on.pad_mask, **args, **kw)
    m_false = build_sdpa_attention_mask(b_on.prompt_lens, b_on.num_pairs, b_on.block_lens,
                                        b_on.pad_mask, canvas_pairs=cp_false[:, :2], **args, **kw)
    assert torch.equal(m_none, m_false), "all-False canvas_pairs changed the variable mask"
    print("[allFalse == None]    OK")


def test_canvas_construction_offpolicy():
    _set_base_env()
    os.environ["CONSISTENCY_VOCAB_SIZE"] = "152064"  # big vocab: collision prob ~0
    os.environ["CONSISTENCY_CANVAS_FRAC"] = "1.0"
    gen = torch.Generator().manual_seed(0)
    prompts = [torch.tensor([1, 2, 3])]
    responses = [torch.arange(5000, 5000 + 8 * N)]  # 8 full blocks
    b = build_interleaved_batch(prompts, responses, block_size=N, pad_id=0, generator=gen)
    assert b.canvas_pairs is not None and bool(b.canvas_pairs.all())
    assert torch.equal(b.canvas_mask, b.noisy_mask), "frac=1: canvas_mask must cover all noisy"
    levels_n = {int(round(f * N)) for f in (1.0, 0.75, 0.5, 0.25, 0.125)}
    Pb = 3
    seen_levels = set()
    for j in range(8):
        ks = Pb + 2 * j * N
        ls = ks + N
        tile = b.input_ids[0, ks:ks + N]
        clean = b.input_ids[0, ls:ls + N]
        n_diff = int((tile != clean).sum().item())
        assert n_diff in levels_n, f"pair {j}: renoise count {n_diff} not a level multiple"
        seen_levels.add(n_diff)
    assert len(seen_levels) >= 2, f"only one level drawn across 8 pairs: {seen_levels}"
    print(f"[canvas construction] OK  levels seen (of {sorted(levels_n)}): {sorted(seen_levels)}")


def test_far_weighting():
    _set_base_env()
    os.environ["CONSISTENCY_VOCAB_SIZE"] = "152064"
    os.environ["CONSISTENCY_CANVAS_FRAC"] = "1.0"
    os.environ["CONSISTENCY_CANVAS_LEVELS"] = "0.5"
    gen = torch.Generator().manual_seed(123)
    T = 64
    prompts = [torch.tensor([1])]
    responses = [torch.arange(3000, 3000 + T * N)]
    b = build_interleaved_batch(prompts, responses, block_size=N, pad_id=0, generator=gen)
    Pb = 1
    renoised_idx = []
    for j in range(T):
        ks = Pb + 2 * j * N
        ls = ks + N
        diff = (b.input_ids[0, ks:ks + N] != b.input_ids[0, ls:ls + N]).nonzero().flatten()
        renoised_idx.extend(diff.tolist())
    mean_r = sum(renoised_idx) / len(renoised_idx)
    # w_j ∝ 0.5+j/N → E[renoised idx] ≈ 3.96 for N=8 (without-replacement, half
    # drawn) vs uniform 3.5. Loose statistical band.
    assert mean_r > 3.6, f"renoised positions not far-weighted: mean idx {mean_r:.2f}"
    print(f"[far weighting]       OK  mean renoised idx {mean_r:.2f} > uniform 3.5")


def test_onpolicy_canvas_vs_causal():
    _set_base_env()
    os.environ["CONSISTENCY_VOCAB_SIZE"] = "152064"
    # Find a seed giving one canvas + one causal pair at frac=0.5.
    os.environ["CONSISTENCY_CANVAS_FRAC"] = "0.5"
    prompts = [torch.tensor([1, 2, 3])]
    responses = [torch.arange(100, 132)]
    found = None
    for s in range(64):
        b = build_interleaved_batch(prompts, responses, block_size=N, pad_id=0,
                                    generator=torch.Generator().manual_seed(s),
                                    cascade_drafts=_cascade())
        flags = b.canvas_pairs[0, :2].tolist()
        if flags == [True, False] or flags == [False, True]:
            found = (s, b, flags)
            break
    assert found is not None, "no mixed-mode seed in 64 tries (coin broken?)"
    s, b, flags = found
    cas = _cascade()[0]
    for j, is_cv in enumerate(flags):
        ks = int(b.noisy_starts[0, j].item())
        tile = b.input_ids[0, ks:ks + N]
        draft = torch.tensor(list(cas[j].draft), dtype=torch.long)
        pml = int(b.onpolicy_prefix_lens[0, j].item())
        if is_cv:
            assert not torch.equal(tile, draft), f"canvas pair {j} still equals draft"
            assert pml == 0, f"canvas pair {j} pml={pml}, want 0"
            assert bool(b.canvas_mask[0, ks:ks + N].all())
        else:
            assert torch.equal(tile, draft), f"causal pair {j} draft was modified"
            assert pml == cas[j].prefix_match_len, f"causal pair {j} pml changed"
            assert not bool(b.canvas_mask[0, ks:ks + N].any())
    print(f"[on-policy mix]       OK  seed={s} flags={flags}")


def _allowed(mask, q, k):
    return bool(mask[0, 0, q, k].item() >= -1e30)


def test_mask_per_pair_bidir():
    _set_base_env()
    # Handcrafted: B=1, P=2, T=2, N=4, uniform layout. Pair 0 canvas, pair 1 causal.
    P = torch.tensor([2]); T = torch.tensor([2]); Nn = torch.tensor([4])
    L = 2 + 2 * 2 * 4
    pad = torch.zeros((1, 128), dtype=torch.bool); pad[0, :L] = True
    cp = torch.tensor([[True, False]])
    m = build_sdpa_attention_mask(P, T, Nn, pad, device="cpu", dtype=torch.float32,
                                  causal_region_size=0, canvas_pairs=cp)
    n0, c0, n1, c1 = 2, 6, 10, 14  # noisy0, clean0, noisy1, clean1 starts
    assert _allowed(m, n0, n0 + 3), "canvas tile q->future k blocked"
    assert _allowed(m, n0 + 1, n0 + 2), "canvas tile interior bidir blocked"
    assert not _allowed(m, n1, n1 + 1), "causal tile leaks future"
    assert _allowed(m, n1 + 1, n1), "causal tile past blocked?!"
    assert not _allowed(m, n0, c0), "noisy attends own clean (must not)"
    assert not _allowed(m, n0 + 3, n1), "cross-tile noisy leak"
    assert _allowed(m, n1, c0 + 3), "pair1 noisy must see pair0 clean"
    assert not _allowed(m, c0 + 1, c0 + 2), "clean block went bidir"
    assert _allowed(m, c0 + 2, c0 + 1), "clean causal broken"
    # Variable path (on-policy bridge) with the same flags.
    _set_base_env()
    os.environ["CONSISTENCY_VOCAB_SIZE"] = "152064"
    os.environ["CONSISTENCY_CANVAS_FRAC"] = "0.5"
    prompts = [torch.tensor([1, 2, 3])]
    responses = [torch.arange(100, 132)]
    b = None
    for s in range(64):
        bb = build_interleaved_batch(prompts, responses, block_size=N, pad_id=0,
                                     generator=torch.Generator().manual_seed(s),
                                     cascade_drafts=_cascade())
        if bb.canvas_pairs[0, :2].tolist() == [True, False]:
            b = bb
            break
    assert b is not None
    mv = build_sdpa_attention_mask(
        b.prompt_lens, b.num_pairs, b.block_lens, b.pad_mask, device="cpu",
        dtype=torch.float32, causal_region_size=0, role_per_pos=b.role_per_pos,
        pair_idx_per_pos=b.pair_idx_per_pos, noisy_starts=b.noisy_starts,
        canvas_pairs=b.canvas_pairs,
    )
    k0 = int(b.noisy_starts[0, 0].item()); k1 = int(b.noisy_starts[0, 1].item())
    assert _allowed(mv, k0, k0 + N - 1), "variable: canvas tile not bidir"
    assert not _allowed(mv, k1, k1 + 1), "variable: causal tile leaks future"
    assert not _allowed(mv, k0 + N - 1, k1), "variable: cross-pair noisy leak"
    print("[mask per-pair]       OK  (uniform + variable paths)")


def test_end_to_end_mixed():
    _set_base_env()
    os.environ["CONSISTENCY_CANVAS_FRAC"] = "0.5"
    os.environ["CONSISTENCY_USE_DRAFT_MARKER"] = "1"
    os.environ["CONSISTENCY_MARKER_TYPE"] = "constant"
    compute_consistency_loss._diag_dumped = True
    model = tiny_model()
    model.train()
    prompts = [torch.tensor([1, 2, 3, 4, 5])]
    responses = [torch.arange(20, 84) % VOCAB]  # 8 pairs -> mixed at frac=0.5
    loss, anchor, metrics = compute_consistency_loss(
        model=model, prompt_ids=prompts, response_ids=responses,
        block_size=N, pad_id=0, max_pairs=None, T_soft=1.0, seed=11,
        device="cpu", divergence="forward_kl", teacher_model=None,
    )
    assert torch.isfinite(loss) and float(loss.item()) > 0
    assert 0.0 < metrics["cons_canvas_pos_frac"] < 1.0, metrics["cons_canvas_pos_frac"]
    assert "cons_argmax_correct_canvas" in metrics and "cons_argmax_correct_causal" in metrics
    loss.backward()
    gnorm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
            gnorm += float(p.grad.norm().item() ** 2)
    assert gnorm > 0
    print(f"[end-to-end mixed]    OK  loss={float(loss):.4f} "
          f"canvas_frac={metrics['cons_canvas_pos_frac']:.2f} grad_norm={gnorm ** 0.5:.4f}")


def test_marker_only_on_canvas():
    """canvas_mask present but all-False → marker on == marker off exactly."""
    _set_base_env()
    os.environ["CONSISTENCY_CANVAS_FRAC"] = "0.0001"  # active machinery, ~never flips
    compute_consistency_loss._diag_dumped = True
    model = tiny_model()
    prompts = [torch.tensor([1, 2, 3, 4, 5])]
    responses = [torch.arange(20, 44) % VOCAB]

    def run(marker):
        if marker:
            os.environ["CONSISTENCY_USE_DRAFT_MARKER"] = "1"
            os.environ["CONSISTENCY_MARKER_TYPE"] = "constant"
        else:
            os.environ.pop("CONSISTENCY_USE_DRAFT_MARKER", None)
        loss, _, m = compute_consistency_loss(
            model=model, prompt_ids=prompts, response_ids=responses,
            block_size=N, pad_id=0, max_pairs=None, T_soft=1.0, seed=5,
            device="cpu", divergence="forward_kl", teacher_model=None,
        )
        return float(loss.item()), m

    l_off, m_off = run(False)
    l_on, m_on = run(True)
    assert m_on.get("cons_canvas_pos_frac", 0.0) == 0.0, "a pair flipped canvas; pick another seed"
    assert abs(l_on - l_off) < 1e-7, f"marker leaked outside canvas positions: {l_on} vs {l_off}"
    print(f"[marker scoping]      OK  loss identical with all-False canvas_mask ({l_on:.6f})")


if __name__ == "__main__":
    test_golden_regression()
    test_allfalse_canvas_equals_none()
    test_canvas_construction_offpolicy()
    test_far_weighting()
    test_onpolicy_canvas_vs_causal()
    test_mask_per_pair_bidir()
    test_end_to_end_mixed()
    test_marker_only_on_canvas()
    print("\nALL CANVAS-PAIR TESTS PASSED")
