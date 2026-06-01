"""Build the Decode-Learning interleaved `[prompt, k_0, last_0, ...]` batch
from verl's standard (prompt, response) rollouts.

  k_j   (noisy block):  initialized from existing sequence tokens (matches
                         JF's `random.choice(generated_ids)` init)
  last_j (clean block): copy of the j-th block of the rollout response

Both k_j and last_j get the SAME position_ids — they represent two views
of the same Jacobi step (essential for RoPE consistency at the model
level).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class InterleavedBatch:
    input_ids: torch.Tensor          # (B, Lmax) int64
    position_ids: torch.Tensor       # (B, Lmax) int64 — shared positions per pair/triple
    prompt_lens: torch.Tensor        # (B,) int64
    num_pairs: torch.Tensor          # (B,) int64 — T per sample
    block_lens: torch.Tensor         # (B,) int64 — N per sample (currently uniform)
    seq_lens: torch.Tensor           # (B,) int64
    pad_mask: torch.Tensor           # (B, Lmax) bool — True where valid
    noisy_mask: torch.Tensor         # (B, Lmax) bool — True at ANY noisy position
    # Multi-noise tiling (2-block layout only): K independent noisy tiles per
    # clean block, all sharing the same RoPE positions as the clean block.
    # K=1 (default) is the historical [noisy | clean] layout. K>1 enables
    # diffusion-style Monte-Carlo averaging over noise samples.
    num_noisy_tiles: int = 1
    # triple_mode only:
    triple_mode: bool = False        # True if batch was built with the 3-block layout
    marked_mask: torch.Tensor | None = None    # (B, Lmax) bool — True at marker-side noisy positions only
    unmarked_mask: torch.Tensor | None = None  # (B, Lmax) bool — True at no-marker-side noisy positions only

    def to(self, device):
        return InterleavedBatch(
            input_ids=self.input_ids.to(device),
            position_ids=self.position_ids.to(device),
            prompt_lens=self.prompt_lens.to(device),
            num_pairs=self.num_pairs.to(device),
            block_lens=self.block_lens.to(device),
            seq_lens=self.seq_lens.to(device),
            pad_mask=self.pad_mask.to(device),
            noisy_mask=self.noisy_mask.to(device),
            num_noisy_tiles=self.num_noisy_tiles,
            triple_mode=self.triple_mode,
            marked_mask=self.marked_mask.to(device) if self.marked_mask is not None else None,
            unmarked_mask=self.unmarked_mask.to(device) if self.unmarked_mask is not None else None,
        )


def build_interleaved_batch(
    prompt_ids: list[torch.Tensor],
    response_ids: list[torch.Tensor],
    block_size: int,
    pad_id: int,
    max_pairs: int | None = None,
    generator: torch.Generator | None = None,
    triple_mode: bool = False,
    num_noisy_tiles: int = 1,
) -> InterleavedBatch:
    """Build the interleaved batch from a list of (prompt, response) tensors.

    For each sample with response of length R:
      T = ceil(R / block_size) clean blocks, each of length N=block_size
        (last block right-padded with pad_id to N if R % N != 0)
      For each clean block last_j (positions [P + (2j+1)N, P + (2j+2)N) in the
      packed sequence), construct a paired noisy block k_j (positions
      [P + 2jN, P + (2j+1)N)) by sampling `block_size` tokens uniformly
      with replacement from (prompt + response). This matches the JF
      inference prefill init (random.choice from generated_ids).

    The total packed length per sample is P + 2*T*N. Across the batch,
    we pad to the max packed length with pad_id.
    """
    assert len(prompt_ids) == len(response_ids)
    B = len(prompt_ids)
    N = int(block_size)
    K = int(num_noisy_tiles)
    assert K >= 1, f"num_noisy_tiles must be >= 1, got {K}"
    if triple_mode and K != 1:
        raise ValueError(
            f"triple_mode is incompatible with num_noisy_tiles > 1 (got K={K}); "
            f"multi-noise tiling is only implemented for the 2-block layout."
        )

    # Per-sample packed lengths.
    #   2-block (K=1):       [prompt | noisy(N) | clean(N) | noisy(N) | clean(N) | ... ]
    #   2-block multi-noise: [prompt | noisy1(N) ... noisyK(N) | clean(N) | noisy1(N) ... | ...]
    #   3-block:             [prompt | marked(N) | unmarked(N) | clean(N) | ... ]
    blocks_per_group = 3 if triple_mode else (K + 1)
    P = torch.tensor([int(p.numel()) for p in prompt_ids], dtype=torch.long)
    Rn = torch.tensor([int(r.numel()) for r in response_ids], dtype=torch.long)
    T_full = ((Rn + N - 1) // N).long()
    if max_pairs is not None:
        T_full = torch.minimum(T_full, torch.tensor(int(max_pairs), dtype=torch.long))
    seq_lens = P + blocks_per_group * T_full * N
    Lmax_raw = int(seq_lens.max().item())
    # flex_attention backward Triton kernel is unreliable when Lmax isn't
    # a multiple of its internal block tile (typically 128). Pad up.
    BLOCK_TILE = 128
    Lmax = ((Lmax_raw + BLOCK_TILE - 1) // BLOCK_TILE) * BLOCK_TILE
    Lmax = max(Lmax, BLOCK_TILE)

    input_ids = torch.full((B, Lmax), int(pad_id), dtype=torch.long)
    position_ids = torch.zeros((B, Lmax), dtype=torch.long)
    pad_mask = torch.zeros((B, Lmax), dtype=torch.bool)
    noisy_mask = torch.zeros((B, Lmax), dtype=torch.bool)
    marked_mask = torch.zeros((B, Lmax), dtype=torch.bool) if triple_mode else None
    unmarked_mask = torch.zeros((B, Lmax), dtype=torch.bool) if triple_mode else None

    import os as _os
    # Default is "uniform" — "sample" mode (draw from prompt+response) leaks the
    # training-prompt distribution into the denoiser. Confirmed failure mode in
    # FINDINGS.md (HE+ pass@8 collapsed 0.762 → 0.601 by step 120). Keep "sample"
    # available behind the env var for ablation but never as the silent default.
    _noise_source = _os.environ.get("CONSISTENCY_NOISE_SOURCE", "uniform").lower()
    _vocab_size = int(_os.environ.get("CONSISTENCY_VOCAB_SIZE", "152064"))
    # dFlash-style mask-token init. When CONSISTENCY_NOISE_SOURCE=mask, every
    # noisy-block position gets the SAME single token id — a consistent "predict
    # here" signal the model can specialize on (cf. dFlash arXiv:2602.06036).
    # Default mask id = Qwen2.5 pad_id (151643), which is reserved and never
    # appears in clean responses.
    _mask_token_id = int(_os.environ.get("CONSISTENCY_MASK_TOKEN_ID", "151643"))

    # Multi-noise tiling is incompatible with mask noise (all K tiles would be
    # identical — defeats the Monte-Carlo averaging purpose). Reject early.
    if K > 1 and _noise_source == "mask":
        raise ValueError(
            f"CONSISTENCY_NUM_NOISY_TILES={K} requires a stochastic noise source; "
            f"got CONSISTENCY_NOISE_SOURCE='mask' which is deterministic."
        )

    for b in range(B):
        Pb = int(P[b].item())
        Tb = int(T_full[b].item())
        # prompt section
        input_ids[b, :Pb] = prompt_ids[b]
        position_ids[b, :Pb] = torch.arange(Pb, dtype=torch.long)
        pad_mask[b, :Pb] = True

        pool = torch.cat([prompt_ids[b].long(), response_ids[b].long()])

        def _draw_noisy(g):
            if _noise_source == "uniform":
                return torch.randint(low=0, high=_vocab_size, size=(N,),
                                      generator=g, dtype=torch.long) if g is not None \
                    else torch.randint(low=0, high=_vocab_size, size=(N,), dtype=torch.long)
            elif _noise_source == "mask":
                # dFlash-style: every position gets the SAME mask token id.
                return torch.full((N,), _mask_token_id, dtype=torch.long)
            else:
                idx = torch.randint(low=0, high=int(pool.numel()), size=(N,), generator=g) if g is not None \
                    else torch.randint(low=0, high=int(pool.numel()), size=(N,))
                return pool[idx]

        if not triple_mode:
            # 2-block layout (K=1, historical):
            #   [prompt | noisy | clean | noisy | clean | ...]
            # Multi-noise variant (K>=1):
            #   [prompt | noisy_1 | noisy_2 | ... | noisy_K | clean | noisy_1 | ... ]
            # All K+1 tiles in each group share the same RoPE positions as
            # the clean block they're paired with. Each noisy tile is an
            # INDEPENDENT noise draw (separate _draw_noisy call). Attention
            # masking (in attention.py) prevents sibling noisy tiles from
            # attending to each other.
            for j in range(Tb):
                group_start = Pb + (K + 1) * j * N
                ls = group_start + K * N  # clean tile starts after all K noisy tiles
                r_start = j * N
                r_end = min((j + 1) * N, int(Rn[b].item()))
                valid_n = r_end - r_start
                shared_pos = torch.arange(Pb + j * N, Pb + (j + 1) * N, dtype=torch.long)
                if valid_n > 0:
                    input_ids[b, ls : ls + valid_n] = response_ids[b][r_start:r_end]
                    pad_mask[b, ls : ls + valid_n] = True
                position_ids[b, ls : ls + N] = shared_pos
                for k_tile in range(K):
                    ks = group_start + k_tile * N
                    input_ids[b, ks : ks + N] = _draw_noisy(generator)
                    pad_mask[b, ks : ks + N] = True
                    noisy_mask[b, ks : ks + N] = True
                    position_ids[b, ks : ks + N] = shared_pos
        else:
            # 3-block layout: [prompt | marked | unmarked | clean | marked | unmarked | clean | ...]
            # Each triple j has 3 sub-blocks at:
            #   marked  : [Pb + 3j*N      , Pb + (3j+1)*N)
            #   unmarked: [Pb + (3j+1)*N  , Pb + (3j+2)*N)
            #   clean   : [Pb + (3j+2)*N  , Pb + (3j+3)*N)
            for j in range(Tb):
                ms = Pb + 3 * j * N
                us = Pb + (3 * j + 1) * N
                ls = Pb + (3 * j + 2) * N
                r_start = j * N
                r_end = min((j + 1) * N, int(Rn[b].item()))
                valid_n = r_end - r_start
                if valid_n > 0:
                    input_ids[b, ls : ls + valid_n] = response_ids[b][r_start:r_end]
                    pad_mask[b, ls : ls + valid_n] = True
                # Same noisy tokens for marked and unmarked copies — only the
                # input-side marker embedding differs between the two slots.
                noisy_toks = _draw_noisy(generator)
                input_ids[b, ms : ms + N] = noisy_toks
                input_ids[b, us : us + N] = noisy_toks
                pad_mask[b, ms : ms + N] = True
                pad_mask[b, us : us + N] = True
                noisy_mask[b, ms : ms + N] = True
                noisy_mask[b, us : us + N] = True
                marked_mask[b, ms : ms + N] = True
                unmarked_mask[b, us : us + N] = True
                shared_pos = torch.arange(Pb + j * N, Pb + (j + 1) * N, dtype=torch.long)
                position_ids[b, ms : ms + N] = shared_pos
                position_ids[b, us : us + N] = shared_pos
                position_ids[b, ls : ls + N] = shared_pos

    return InterleavedBatch(
        input_ids=input_ids,
        position_ids=position_ids,
        prompt_lens=P,
        num_pairs=T_full,
        block_lens=torch.full((B,), N, dtype=torch.long),
        seq_lens=seq_lens,
        pad_mask=pad_mask,
        noisy_mask=noisy_mask,
        num_noisy_tiles=K,
        triple_mode=triple_mode,
        marked_mask=marked_mask,
        unmarked_mask=unmarked_mask,
    )


if __name__ == "__main__":
    # quick self-test
    g = torch.Generator().manual_seed(42)
    prompt_ids = [torch.tensor([1, 2, 3, 4, 5]), torch.tensor([10, 11, 12])]
    response_ids = [torch.tensor([20, 21, 22, 23, 24, 25, 26, 27]),
                    torch.tensor([30, 31, 32, 33])]
    b = build_interleaved_batch(prompt_ids, response_ids, block_size=4, pad_id=0, generator=g)
    print("seq_lens:", b.seq_lens.tolist())
    print("num_pairs:", b.num_pairs.tolist())
    print("input_ids[0]:", b.input_ids[0].tolist())
    print("position_ids[0]:", b.position_ids[0].tolist())
    print("pad_mask[0]:", b.pad_mask[0].tolist())
