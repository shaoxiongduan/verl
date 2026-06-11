"""Assembly-line decode sim: causal AR commit zone + bidir diffusion canvas,
one forward per step via a custom 4D attention mask.

Window of W tokens after the committed prefix:
  positions [0, W_ar)   : AR zone — strictly causal attention; its logits are
                          exact AR logits → prefix-accept + bonus commits the
                          verifier-greedy text (quality exact by construction).
                          Commits are capped at W_ar (+1 bonus inside zone).
  positions [W_ar, W)   : canvas — attends causally to everything before it
                          and BIDIRECTIONALLY within the canvas region
                          (--canvas_attn causal disables, as control).
Per forward: AR zone verifies its input (Jacobi prefix match), canvas updates
(argmax of its logits; --canvas_update keepnoise re-noises high-entropy
positions). Window slides by the commit length; canvas tokens graduate into
the AR zone; fresh random tokens enter at the far end.
"""
from __future__ import annotations
import argparse, json, os, random, sys
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from consistency.loss import _constant_marker  # same vector as training-time marker

STOP_IDS = {151645, 151643}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompts_jsonl", required=True)
    p.add_argument("--out_jsonl", required=True)
    p.add_argument("--W", type=int, default=32)
    p.add_argument("--W_ar", type=int, default=8)
    p.add_argument("--canvas_attn", choices=["bidir", "causal"], default="bidir")
    p.add_argument("--canvas_update", choices=["argmax", "keepnoise"], default="argmax")
    p.add_argument("--keep_tau", type=float, default=2.0)
    # v11: additive mode marker on canvas-position input embeddings. "constant"
    # = the training-time CONSISTENCY_MARKER_TYPE=constant vector (sinusoidal
    # row 0, imported from consistency.loss for bit-equality with training).
    p.add_argument("--marker", choices=["none", "constant"], default="none")
    p.add_argument("--marker_scale", type=float, default=1.0)
    # Canvas candidate sampling: argmax (v1) or Gumbel logits/T + g (the AR
    # zone always uses argmax — its verify/commit must stay exact-greedy).
    p.add_argument("--candidates", choices=["argmax", "gumbel"], default="argmax")
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--max_new", type=int, default=512)
    p.add_argument("--max_fwd", type=int, default=512)
    # Per-forward trace dump: window state BEFORE update, candidates, keep
    # mask, acceptance — plus the final committed tokens per prompt. Feeds
    # _analyze_assembly_trace.py (decode-time noise-profile vs training).
    p.add_argument("--trace_jsonl", default=None)
    p.add_argument("--n_prompts", type=int, default=16)
    p.add_argument("--vocab_size", type=int, default=152064)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_mask(L0, W, W_ar, bidir, device, dtype):
    L = L0 + W
    neg = torch.finfo(dtype).min
    m = torch.full((L, L), neg, dtype=dtype, device=device)
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))
    m.masked_fill_(causal, 0.0)
    if bidir:
        c0 = L0 + W_ar
        m[c0:L, c0:L] = 0.0   # canvas attends bidirectionally within itself
    return m.unsqueeze(0).unsqueeze(0)  # (1,1,L,L)


@torch.no_grad()
def decode(model, prompt_ids, args, rng, marker_vec=None, trace=None, prompt_idx=0):
    W, W_ar = args.W, args.W_ar
    committed = list(prompt_ids)
    window = [rng.randrange(args.vocab_size) for _ in range(W)]
    total = n_fwd = 0
    emb_layer = model.get_input_embeddings() if marker_vec is not None else None
    while total < args.max_new and n_fwd < args.max_fwd:
        L0 = len(committed)
        inp = torch.tensor([committed + window], dtype=torch.long, device=model.device)
        mask = build_mask(L0, W, W_ar, args.canvas_attn == "bidir",
                          model.device, model.dtype)
        if marker_vec is not None:
            # Add the mode marker to canvas-position embeddings only (the AR
            # zone and committed prefix stay unmarked = AR mode).
            embeds = emb_layer(inp)
            embeds[0, L0 + W_ar:, :] += args.marker_scale * marker_vec
            logits = model(inputs_embeds=embeds, attention_mask=mask).logits[0, L0 - 1: L0 - 1 + W, :]
        else:
            logits = model(input_ids=inp, attention_mask=mask).logits[0, L0 - 1: L0 - 1 + W, :]
        cur = logits.argmax(dim=-1).cpu().tolist()
        if args.candidates == "gumbel":
            # Gumbel candidates for the CANVAS only; AR-zone verification at
            # positions [0, W_ar) must remain exact-greedy argmax.
            lg = logits[W_ar:].float() / max(args.temp, 1e-6)
            gn = -torch.log(-torch.log(torch.rand_like(lg).clamp_min(1e-20)).clamp_min(1e-20))
            cur[W_ar:] = (lg + gn).argmax(dim=-1).cpu().tolist()
        n_fwd += 1
        # AR-zone verification (positions 0..W_ar-1 only; logits there are exact AR)
        n_acc = 0
        for j in range(W_ar):
            if cur[j] == window[j]:
                n_acc += 1
            else:
                break
        toks = window[:n_acc] + ([cur[n_acc]] if n_acc < W_ar else [])
        hit = False
        for si, t in enumerate(toks):
            if t in STOP_IDS:
                toks = toks[: si + 1]; hit = True; break
        committed += toks
        total += len(toks)
        if hit or total >= args.max_new:
            break
        # zone updates: AR zone Jacobi (causal argmax), canvas per policy
        upd = list(cur)
        keep_mask = None
        if args.canvas_update == "keepnoise":
            lp = F.log_softmax(logits[W_ar:].float(), dim=-1)
            ent = (-(lp.exp() * lp).sum(-1)).cpu().tolist()
            keep_mask = [ent[j] <= args.keep_tau for j in range(W - W_ar)]
            for j in range(W - W_ar):
                if not keep_mask[j]:
                    upd[W_ar + j] = rng.randrange(args.vocab_size)
        if trace is not None:
            trace.write(json.dumps({
                "p": prompt_idx, "fwd": n_fwd, "L0": L0, "win": window,
                "cand": cur, "keep": keep_mask, "n_acc": n_acc,
                "n_commit": len(toks),
            }) + "\n")
        n = len(toks)
        window = upd[n:] + [rng.randrange(args.vocab_size) for _ in range(n)]
    if trace is not None:
        trace.write(json.dumps({
            "p": prompt_idx, "final": committed, "prompt_len": len(prompt_ids),
        }) + "\n")
    return total, n_fwd


@torch.no_grad()
def main():
    args = parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="sdpa").eval()
    prompts = [json.loads(l) for l in open(args.prompts_jsonl)][: args.n_prompts]
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)  # gumbel noise reproducibility
    marker_vec = None
    if args.marker == "constant":
        H = model.config.hidden_size
        marker_vec = _constant_marker(H, model.device, model.dtype)
        print(f"[asm] marker=constant scale={args.marker_scale} "
              f"norm={float(marker_vec.float().norm().item()):.3f}", flush=True)
    T = F_ = 0
    trace = open(args.trace_jsonl, "w") if args.trace_jsonl else None
    with open(args.out_jsonl, "w") as f:
        for i, p in enumerate(prompts):
            chat = [{"role": "user", "content": p["input"]}]
            text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            pid = tok(text, return_tensors="pt").input_ids[0].tolist()
            tot, nf = decode(model, pid, args, rng, marker_vec=marker_vec,
                             trace=trace, prompt_idx=i)
            T += tot; F_ += nf
            f.write(json.dumps({"batch_idx": i, "n_tokens": tot, "n_forwards": nf,
                                "tpf": tot / max(1, nf)}) + "\n")
            print(f"[asm] [{i+1}/{len(prompts)}] {args.canvas_attn}/{args.canvas_update} "
                  f"marker={args.marker} tok={tot} fwd={nf} TPF={tot/max(1,nf):.2f}", flush=True)
    if trace is not None:
        trace.close()
    print(f"[asm] canvas={args.canvas_attn} update={args.canvas_update} W_ar={args.W_ar} "
          f"marker={args.marker} cand={args.candidates}: "
          f"CORPUS TPF = {T}/{F_} = {T/max(1,F_):.4f}")


if __name__ == "__main__":
    main()
