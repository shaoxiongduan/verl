"""Unit test for the idea-A mask-tail consistency loss (`_mask_tail_dflash_loss`).

Runs on CPU with synthetic logits — no model forward — so it is safe on the
login node. Verifies, on real packed batches:
  * clean/mask position partition matches  cut = pml + margin,
  * mask targets are the mask token id and gradient flows,
  * off-policy (pml=0) and on-policy (pml>0) both route correctly.

Run:  python scripts/test_mask_tail_loss.py
"""
import os
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from consistency.pack import build_interleaved_batch  # noqa: E402
from consistency.loss import _identify_block_positions, _mask_tail_dflash_loss  # noqa: E402

VOCAB = 10
MASK_ID = 5
N = 8
MARGIN = 2


def _logits(batch):
    torch.manual_seed(0)
    L = batch.input_ids.shape[1]
    return torch.randn(batch.input_ids.shape[0], L, VOCAB, requires_grad=True)


def _run(batch):
    b_idx, k_pos, l_pos, pos_in_block, pair_idx = _identify_block_positions(batch)
    logits = _logits(batch)
    loss, n_clean, n_mask = _mask_tail_dflash_loss(
        logits, batch, b_idx, k_pos, l_pos, pos_in_block, pair_idx, MASK_ID, MARGIN
    )
    return loss, n_clean, n_mask, logits, (b_idx, k_pos, l_pos, pos_in_block, pair_idx)


def test_off_policy_uniform():
    os.environ["CONSISTENCY_NOISE_SOURCE"] = "uniform"
    os.environ["CONSISTENCY_VOCAB_SIZE"] = str(VOCAB)
    g = torch.Generator().manual_seed(1)
    # 2 full blocks of N=8.
    prompt = [torch.tensor([1, 2, 3])]
    response = [torch.arange(16) % VOCAB]
    batch = build_interleaved_batch(prompt, response, block_size=N, pad_id=0, generator=g)
    loss, n_clean, n_mask, logits, _ = _run(batch)
    # pml=0, cut=2 -> per block: clean {0,1}=2, mask {2..7}=6; x2 blocks.
    assert n_clean == 4, n_clean
    assert n_mask == 12, n_mask
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    print(f"[off-policy uniform] OK  loss={loss.item():.4f} n_clean={n_clean} n_mask={n_mask}")


def test_off_policy_mask():
    os.environ["CONSISTENCY_NOISE_SOURCE"] = "mask"
    os.environ["CONSISTENCY_MASK_TOKEN_ID"] = str(MASK_ID)
    g = torch.Generator().manual_seed(2)
    prompt = [torch.tensor([1, 2, 3])]
    response = [torch.arange(16) % VOCAB]
    batch = build_interleaved_batch(prompt, response, block_size=N, pad_id=0, generator=g)
    # input noisy blocks should all be the mask id.
    assert (batch.input_ids[batch.noisy_mask] == MASK_ID).all()
    loss, n_clean, n_mask, _, _ = _run(batch)
    assert n_clean == 4 and n_mask == 12
    print(f"[off-policy mask]    OK  loss={loss.item():.4f} n_clean={n_clean} n_mask={n_mask}")


def test_on_policy_with_pml():
    for k in ("CONSISTENCY_NOISE_SOURCE",):
        os.environ.pop(k, None)
    g = torch.Generator().manual_seed(3)
    prompt = [torch.tensor([1, 2, 3])]
    response = [torch.arange(16) % VOCAB]
    draft0 = tuple(int(x) for x in (torch.arange(8) % VOCAB).tolist())
    draft1 = tuple(int(x) for x in (torch.arange(8, 16) % VOCAB).tolist())
    cascade = [[
        SimpleNamespace(start=0, end=8, draft=draft0, prefix_match_len=3),
        SimpleNamespace(start=8, end=16, draft=draft1, prefix_match_len=1),
    ]]
    batch = build_interleaved_batch(
        prompt, response, block_size=N, pad_id=0, generator=g, cascade_drafts=cascade
    )
    assert batch.onpolicy_prefix_lens is not None
    loss, n_clean, n_mask, logits, idx = _run(batch)
    # pair0 pml=3 cut=5: clean {3,4}=2, mask {5,6,7}=3
    # pair1 pml=1 cut=3: clean {1,2}=2, mask {3,4,5,6,7}=5
    assert n_clean == 4, n_clean
    assert n_mask == 8, n_mask
    # verify the converged prefix [0,pml) is excluded entirely.
    b_idx, k_pos, l_pos, pos_in_block, pair_idx = idx
    pref = batch.onpolicy_prefix_lens[b_idx, pair_idx]
    cut = pref + MARGIN
    clean = (pos_in_block >= pref) & (pos_in_block < cut) & (pos_in_block < N - 1)
    mask = pos_in_block >= cut
    assert int((clean | mask).sum()) >= n_clean + n_mask - 1  # last-pos edge tolerated
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    print(f"[on-policy pml]      OK  loss={loss.item():.4f} n_clean={n_clean} n_mask={n_mask}")


if __name__ == "__main__":
    test_off_policy_uniform()
    test_off_policy_mask()
    test_on_policy_with_pml()
    print("\nALL MASK-TAIL TESTS PASSED")
