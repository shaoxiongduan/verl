"""Capture golden v9 cons-loss behavior BEFORE the v11 canvas changes.

Run once with the pre-v11 working tree; the output .pt is then asserted
bit-identical by scripts/test_canvas_pairs.py (regression: with canvas mode
off, causal pairs behave exactly as v9). CPU-only, tiny random Qwen2.

Run:  python scripts/_make_v9_golden.py
Out:  scripts/.golden_v9_cons.pt
"""
import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

VOCAB = 1000
N = 8
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".golden_v9_cons.pt")

# Fixed env for the golden scenario (v9 recipe shape: kl + decay, onpolicy).
os.environ["CONSISTENCY_VOCAB_SIZE"] = str(VOCAB)
os.environ["CONSISTENCY_NOISE_SOURCE"] = "uniform"
os.environ["CONSISTENCY_LOSS_TYPE"] = "kl"
os.environ["CONSISTENCY_KL_DECAY"] = "1"
os.environ["CONSISTENCY_DFLASH_GAMMA"] = "12.0"
os.environ.pop("CONSISTENCY_CANVAS_FRAC", None)
os.environ.pop("CONSISTENCY_USE_DRAFT_MARKER", None)

from consistency.pack import build_interleaved_batch  # noqa: E402
from consistency.attention import build_sdpa_attention_mask  # noqa: E402
from consistency.loss import compute_consistency_loss  # noqa: E402


def tiny_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(1234)
    cfg = Qwen2Config(
        vocab_size=VOCAB, hidden_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, intermediate_size=128,
        max_position_embeddings=2048, attn_implementation="sdpa",
    )
    m = Qwen2ForCausalLM(cfg).eval()
    return m


def offpolicy_case():
    g = torch.Generator().manual_seed(42)
    prompts = [torch.tensor([1, 2, 3, 4, 5]), torch.tensor([10, 11, 12])]
    responses = [torch.arange(20, 44) % VOCAB, torch.arange(50, 62) % VOCAB]
    batch = build_interleaved_batch(prompts, responses, block_size=N, pad_id=0,
                                    max_pairs=None, generator=g)
    mask = build_sdpa_attention_mask(
        batch.prompt_lens, batch.num_pairs, batch.block_lens, batch.pad_mask,
        device="cpu", dtype=torch.float32, causal_region_size=0,
    )
    return batch, mask


def onpolicy_case():
    g = torch.Generator().manual_seed(43)
    prompts = [torch.tensor([1, 2, 3])]
    responses = [torch.arange(100, 132) % VOCAB]
    d0 = tuple(int(x) for x in (torch.arange(7, 15)).tolist())
    d1 = tuple(int(x) for x in (torch.arange(200, 208) % VOCAB).tolist())
    cascade = [[
        SimpleNamespace(start=0, end=8, draft=d0, prefix_match_len=3),
        SimpleNamespace(start=12, end=20, draft=d1, prefix_match_len=1),  # gap of 4 -> bridge
    ]]
    batch = build_interleaved_batch(prompts, responses, block_size=N, pad_id=0,
                                    max_pairs=None, generator=g, cascade_drafts=cascade)
    mask = build_sdpa_attention_mask(
        batch.prompt_lens, batch.num_pairs, batch.block_lens, batch.pad_mask,
        device="cpu", dtype=torch.float32, causal_region_size=0,
        role_per_pos=batch.role_per_pos, pair_idx_per_pos=batch.pair_idx_per_pos,
        noisy_starts=batch.noisy_starts,
    )
    return batch, mask


def loss_value(model, cascade=False, marker=False):
    # Skip the one-shot diag block — it calls torch.cuda.synchronize().
    compute_consistency_loss._diag_dumped = True
    if marker:
        os.environ["CONSISTENCY_USE_DRAFT_MARKER"] = "1"
        os.environ["CONSISTENCY_MARKER_TYPE"] = "constant"
    else:
        os.environ.pop("CONSISTENCY_USE_DRAFT_MARKER", None)
        os.environ.pop("CONSISTENCY_MARKER_TYPE", None)
    prompts = [torch.tensor([1, 2, 3, 4, 5])]
    responses = [torch.arange(20, 44) % VOCAB]
    cd = None
    if cascade:
        d0 = tuple(int(x) for x in (torch.arange(7, 15)).tolist())
        d1 = tuple(int(x) for x in (torch.arange(200, 208) % VOCAB).tolist())
        cd = [[
            SimpleNamespace(start=0, end=8, draft=d0, prefix_match_len=3),
            SimpleNamespace(start=12, end=20, draft=d1, prefix_match_len=1),
        ]]
    loss, anchor, metrics = compute_consistency_loss(
        model=model, prompt_ids=prompts, response_ids=responses,
        block_size=N, pad_id=0, max_pairs=None, T_soft=1.0, seed=7,
        device="cpu", divergence="forward_kl", teacher_model=None,
        cascade_drafts=cd,
    )
    return float(loss.item()), metrics


def main():
    model = tiny_model()
    b_off, m_off = offpolicy_case()
    b_on, m_on = onpolicy_case()
    l_off, met_off = loss_value(model, cascade=False, marker=False)
    l_on, met_on = loss_value(model, cascade=True, marker=False)
    l_marker, met_marker = loss_value(model, cascade=True, marker=True)
    golden = {
        "off_input_ids": b_off.input_ids, "off_position_ids": b_off.position_ids,
        "off_pad_mask": b_off.pad_mask, "off_noisy_mask": b_off.noisy_mask,
        "off_sdpa_mask": m_off,
        "on_input_ids": b_on.input_ids, "on_position_ids": b_on.position_ids,
        "on_pad_mask": b_on.pad_mask, "on_noisy_mask": b_on.noisy_mask,
        "on_prefix_lens": b_on.onpolicy_prefix_lens, "on_sdpa_mask": m_on,
        "loss_offpolicy": l_off, "loss_onpolicy": l_on, "loss_onpolicy_marker": l_marker,
        "argmax_correct_onpolicy": met_on.get("cons_argmax_correct", -1.0),
    }
    torch.save(golden, OUT)
    print(f"golden saved -> {OUT}")
    print(f"  loss_offpolicy={l_off:.6f} loss_onpolicy={l_on:.6f} loss_onpolicy_marker={l_marker:.6f}")
    print(f"  argmax_correct_onpolicy={golden['argmax_correct_onpolicy']:.4f}")


if __name__ == "__main__":
    main()
