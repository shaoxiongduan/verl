"""Per-iter best-of-K lookahead Jacobi simulation.

For each Jacobi iteration, we:
  1. Greedy verify the current draft against the "clean reference" (a greedy AR
     completion captured upfront). Determine n_acc.
  2. For the K - n_acc - 1 noisy tail positions, sample K_alt alternative
     draft tails at temperature T from the model conditioned on
     [prompt | clean_prefix_committed | bonus].
  3. For each alternative tail, run ONE EXTRA forward pass (the lookahead)
     and count how many of its K positions would be accepted at the NEXT iter
     (i.e. match the clean reference at the lookahead positions).
  4. Pick the alternative with the highest next-iter n_acc'.
  5. Commit per that alternative's verification at the current iter (we still
     only commit n_acc + 1 tokens — what changes is the draft we feed the NEXT
     iter, hence next-iter acceptance).

Compares two TPF numbers:
  - vanilla (no resampling): single greedy draft at each iter
  - lookahead-best-of-K: K alternatives, pick by next-iter n_acc

If lookahead substantially beats vanilla, there exist "good" confusing-token
choices the model COULD generate via sampling, which validates the idea of
training the model to produce them (per-iter TPF reward).

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 scripts/_sim_jacobi_lookahead.py \\
        --model PATH --prompts_jsonl ... --K_alt 8 --out_jsonl ...
"""
from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompts_jsonl", required=True)
    p.add_argument("--out_jsonl", required=True)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--K_alt", type=int, default=8)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--max_new", type=int, default=256)
    p.add_argument("--max_iters", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def greedy_ref(model, prompt_ids: list[int], max_new: int, eos_id: int) -> list[int]:
    """Generate a clean AR reference completion via vanilla greedy."""
    device = model.device
    out_ids: list[int] = []
    cur = list(prompt_ids)
    for _ in range(max_new):
        inp = torch.tensor([cur], dtype=torch.long, device=device)
        log = model(input_ids=inp).logits[0, -1]
        tok = int(log.argmax().item())
        out_ids.append(tok)
        cur.append(tok)
        if tok == eos_id:
            break
    return out_ids


@torch.no_grad()
def forward_K(model, committed: list[int], draft: list[int]) -> list[int]:
    """Return target_argmax[j] for j=0..K-1 (each predicts token at L+j)."""
    device = model.device
    inp = torch.tensor([committed + draft], dtype=torch.long, device=device)
    K = len(draft)
    L = len(committed)
    logits = model(input_ids=inp).logits[0, L - 1 : L - 1 + K, :]
    return logits.argmax(dim=-1).cpu().tolist()


def count_match(draft: list[int], ref: list[int]) -> int:
    """First position where draft[j] != ref[j]."""
    n = 0
    for d, r in zip(draft, ref):
        if d == r:
            n += 1
        else:
            break
    return n


@torch.no_grad()
def sample_alternatives_parallel(logits_K: torch.Tensor, K_alt: int, temp: float,
                                  gen: torch.Generator) -> list[list[int]]:
    """Generate K_alt alternative drafts by sampling independently at each of
    K positions from the SAME pre-computed forward-pass logits — i.e. each
    alt is a noisy parallel sample under the SAME (causal-on-current-draft)
    context that produced target_argmax. No extra forward passes; reuses the
    logits computed for verification.

    Returns K_alt token lists of length K (one alternative draft each).
    """
    # logits_K: (K, vocab). Apply temperature, sample K_alt times per position.
    if temp > 0:
        probs = torch.softmax(logits_K.float() / temp, dim=-1)  # (K, V)
    else:
        # temp=0 → degenerate; just return argmax K_alt times.
        am = logits_K.argmax(dim=-1).cpu().tolist()
        return [list(am) for _ in range(K_alt)]
    # Multinomial sample K_alt independent rows for each position.
    # torch.multinomial requires 2D input; sample K_alt times for each of K rows.
    # Shape (K, K_alt) of sampled token ids.
    samples = torch.multinomial(probs, num_samples=K_alt, replacement=True, generator=gen)
    samples = samples.cpu().tolist()  # (K, K_alt) → list of lists
    # Transpose: K_alt alternatives, each of length K.
    alts: list[list[int]] = [[samples[j][k] for j in range(len(samples))] for k in range(K_alt)]
    return alts


@torch.no_grad()
def simulate(model, tok, prompt_ids: list[int], args, gen: torch.Generator) -> dict:
    device = model.device
    K = args.K
    eos_id = tok.eos_token_id

    # Reference clean completion (greedy AR from this model).
    ref = greedy_ref(model, prompt_ids, args.max_new, eos_id)

    committed_vanilla = list(prompt_ids)
    committed_lookahead = list(prompt_ids)
    prompt_len = len(prompt_ids)

    # Initial drafts: prompt_sample
    rng_cpu = torch.Generator(device='cpu').manual_seed(args.seed)
    init_draft = [int(prompt_ids[torch.randint(0, len(prompt_ids), (1,), generator=rng_cpu).item()])
                  for _ in range(K)]
    draft_v = list(init_draft)
    draft_l = list(init_draft)

    total_v = 0; iters_v = 0
    total_l = 0; iters_l = 0
    n_acc_v_history: list[int] = []
    n_acc_l_history: list[int] = []

    # Vanilla loop
    while total_v < args.max_new and iters_v < args.max_iters:
        target = forward_K(model, committed_vanilla, draft_v)
        n_acc = count_match(draft_v, target)
        n_acc_v_history.append(n_acc)
        for j in range(n_acc):
            committed_vanilla.append(int(draft_v[j]))
        if n_acc < K:
            committed_vanilla.append(int(target[n_acc]))
            total_v += n_acc + 1
        else:
            total_v += K
        iters_v += 1
        if committed_vanilla[-1] == eos_id:
            break
        shifted = list(target[n_acc + 1:])
        new_draft = list(shifted)
        while len(new_draft) < K:
            new_draft.append(int(committed_vanilla[torch.randint(0, len(committed_vanilla),
                                                                  (1,), generator=rng_cpu).item()]))
        draft_v = new_draft[:K]

    # Lookahead loop — at each iter, sample K_alt alternative PARALLEL drafts
    # from the SAME forward pass (reuse logits) and pick best by ref-window match.
    while total_l < args.max_new and iters_l < args.max_iters:
        # Forward once, keep full logits_K so we can reuse them for sampling.
        L_now = len(committed_lookahead)
        inp = torch.tensor([committed_lookahead + draft_l], dtype=torch.long, device=device)
        logits_K = model(input_ids=inp).logits[0, L_now - 1 : L_now - 1 + K, :]
        target = logits_K.argmax(dim=-1).cpu().tolist()

        n_acc = count_match(draft_l, target)
        n_acc_l_history.append(n_acc)
        for j in range(n_acc):
            committed_lookahead.append(int(draft_l[j]))
        if n_acc < K:
            committed_lookahead.append(int(target[n_acc]))
            total_l += n_acc + 1
        else:
            total_l += K
        iters_l += 1
        if committed_lookahead[-1] == eos_id:
            break

        # Build ref window at the new committed length.
        ref_idx_start = len(committed_lookahead) - prompt_len
        if ref_idx_start >= len(ref):
            break
        ref_window = ref[ref_idx_start : ref_idx_start + K]
        if len(ref_window) < K:
            ref_window = ref_window + [-1] * (K - len(ref_window))

        # Vanilla shifted candidate (deterministic argmax tail + prompt-sample
        # refill). This is what plain Jacobi would use.
        shifted = list(target[n_acc + 1:])
        vanilla_next_draft = list(shifted)
        while len(vanilla_next_draft) < K:
            vanilla_next_draft.append(int(committed_lookahead[torch.randint(
                0, len(committed_lookahead), (1,), generator=rng_cpu).item()]))
        vanilla_next_draft = vanilla_next_draft[:K]

        # Parallel multinomial sampling from the SAME logits_K we just computed.
        alts = sample_alternatives_parallel(logits_K, args.K_alt, args.temp, gen)
        candidates: list[list[int]] = [vanilla_next_draft]
        for alt in alts:
            alt_tail = list(alt[n_acc + 1:])
            while len(alt_tail) < K:
                alt_tail.append(int(committed_lookahead[torch.randint(
                    0, len(committed_lookahead), (1,), generator=rng_cpu).item()]))
            candidates.append(alt_tail[:K])

        # FIX: instead of scoring by count_match(cand, ref_window) — which uses
        # the alt's OLD (pre-commit) context — score each candidate by its actual
        # next-iter acceptance under the NEW context [committed_new | cand]. We
        # do this with ONE batched forward across all K_alt+1 candidates
        # (they all share the committed prefix).
        L_new = len(committed_lookahead)
        # Build batch: each row = committed_lookahead + cand_i
        batch_ids = torch.tensor(
            [committed_lookahead + c for c in candidates],
            dtype=torch.long, device=device,
        )  # (K_alt+1, L_new + K)
        batch_logits = model(input_ids=batch_ids).logits  # (B, L_new+K, V)
        # For each candidate i, extract logits at positions L_new-1 .. L_new-1+K-1
        # (these predict the K candidate positions).
        cand_logits_K = batch_logits[:, L_new - 1 : L_new - 1 + K, :]  # (B, K, V)
        cand_argmax = cand_logits_K.argmax(dim=-1).cpu().tolist()  # (B, K)

        # n_acc_next[i] = count_match(cand_i, cand_argmax_i): the actual prefix
        # the model would accept if cand_i were used as next iter's draft.
        best_idx = 0
        best_n_acc_next = 0
        for i, cand in enumerate(candidates):
            n_acc_next_i = count_match(cand, cand_argmax[i])
            if n_acc_next_i > best_n_acc_next:
                best_n_acc_next = n_acc_next_i
                best_idx = i
        draft_l = list(candidates[best_idx][:K])

    tpf_v = total_v / iters_v if iters_v else 0.0
    tpf_l = total_l / iters_l if iters_l else 0.0
    return {
        "tpf_vanilla": tpf_v,
        "tpf_lookahead": tpf_l,
        "n_tok_vanilla": total_v,
        "n_tok_lookahead": total_l,
        "iters_vanilla": iters_v,
        "iters_lookahead": iters_l,
        "mean_n_acc_v": sum(n_acc_v_history)/len(n_acc_v_history) if n_acc_v_history else 0,
        "mean_n_acc_l": sum(n_acc_l_history)/len(n_acc_l_history) if n_acc_l_history else 0,
        "n_acc_v_history": n_acc_v_history,
        "n_acc_l_history": n_acc_l_history,
        "completion_v_preview": tok.decode(committed_vanilla[prompt_len:][:200], skip_special_tokens=False),
        "completion_l_preview": tok.decode(committed_lookahead[prompt_len:][:200], skip_special_tokens=False),
        "ref_preview": tok.decode(ref[:200], skip_special_tokens=False),
        "ref_len": len(ref),
    }


def main() -> None:
    args = parse_args()
    print(f"[sim-lookahead] loading {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()

    prompts = [json.loads(l) for l in open(args.prompts_jsonl)]
    print(f"[sim-lookahead] {len(prompts)} prompts, K_alt={args.K_alt} temp={args.temp}", flush=True)

    gen = torch.Generator(device="cuda:0").manual_seed(args.seed)
    out_fp = open(args.out_jsonl, "w")
    rows = []
    for i, pdata in enumerate(prompts):
        chat = [{"role": "user", "content": pdata["input"]}]
        prompt_text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        prompt_ids = tok(prompt_text, return_tensors="pt").input_ids[0].tolist()
        r = simulate(model, tok, prompt_ids, args, gen)
        r["batch_idx"] = i
        rows.append(r)
        out_fp.write(json.dumps(r) + "\n")
        out_fp.flush()
        print(f"[sim-lookahead] [{i+1}/{len(prompts)}] "
              f"vanilla TPF={r['tpf_vanilla']:.3f} (n_acc={r['mean_n_acc_v']:.2f})  "
              f"lookahead TPF={r['tpf_lookahead']:.3f} (n_acc={r['mean_n_acc_l']:.2f})  "
              f"Δ={r['tpf_lookahead']-r['tpf_vanilla']:+.3f}", flush=True)

    n = len(rows)
    mean_v = sum(r['tpf_vanilla'] for r in rows)/n
    mean_l = sum(r['tpf_lookahead'] for r in rows)/n
    print(f"\n[sim-lookahead] MEAN vanilla TPF={mean_v:.3f}  lookahead TPF={mean_l:.3f}  Δ={mean_l-mean_v:+.3f}", flush=True)
    out_fp.close()


if __name__ == "__main__":
    main()
