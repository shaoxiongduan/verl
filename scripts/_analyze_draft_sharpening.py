"""Per-position draft-distribution analysis across mask-tail checkpoints.

Measures how the drafter's K-window distribution evolves over training
("sharpening"): for a FIXED committed prefix (generated once by the base JF
model so all checkpoints see identical contexts), feed
``[prefix | tail]`` for three tail types and record per-position stats of
the logits over the K window:

  - top1_prob, entropy
  - p_mask    = probability of the mask token (151643)
  - argmax_is_mask
  - agree_self = argmax matches the checkpoint's OWN greedy continuation

Tail types (matching the three training input schemes):
  - uniform : K uniform-random vocab tokens          (A3 input dist)
  - mask    : K mask tokens                           (A2 input dist)
  - cascade : run N_ITERS Jacobi iters from uniform   (A1 / rollout dist),
              stats recorded at every iter

Usage:
  # once: generate fixed prefixes with the base model
  python scripts/_analyze_draft_sharpening.py --model <BASE> \
      --prompts_jsonl eval_passk/deepscaler_tpf_prompts_16.jsonl \
      --gen_prefix_jsonl /path/prefixes.jsonl --prefix_len 96

  # per checkpoint:
  python scripts/_analyze_draft_sharpening.py --model <CKPT> \
      --prefix_jsonl /path/prefixes.jsonl --out_jsonl <OUT>
"""
from __future__ import annotations
import argparse, json, random
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompts_jsonl", default=None)
    p.add_argument("--gen_prefix_jsonl", default=None,
                   help="If set: greedy-generate prefixes with --model and exit.")
    p.add_argument("--prefix_jsonl", default=None)
    p.add_argument("--out_jsonl", default=None)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--prefix_len", type=int, default=96)
    p.add_argument("--n_iters", type=int, default=4, help="cascade iterations")
    p.add_argument("--mask_id", type=int, default=151643)
    p.add_argument("--vocab_size", type=int, default=152064)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def gen_prefixes(args, tok, model):
    prompts = [json.loads(l) for l in open(args.prompts_jsonl)]
    with open(args.gen_prefix_jsonl, "w") as f:
        for i, p in enumerate(prompts):
            chat = [{"role": "user", "content": p["input"]}]
            text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            ids = tok(text, return_tensors="pt").input_ids.to(model.device)
            out = model.generate(ids, max_new_tokens=args.prefix_len, do_sample=False,
                                 pad_token_id=151643)
            f.write(json.dumps({"batch_idx": i, "ids": out[0].tolist()}) + "\n")
            print(f"[prefix] {i+1}/{len(prompts)} len={out.shape[1]}", flush=True)


@torch.no_grad()
def window_stats(model, ctx_ids: list[int], tail: list[int], mask_id: int):
    """One forward over [ctx | tail]; returns per-position stats over the K window."""
    L = len(ctx_ids)
    K = len(tail)
    inp = torch.tensor([ctx_ids + tail], dtype=torch.long, device=model.device)
    logits = model(input_ids=inp).logits[0, L - 1 : L - 1 + K, :].float()
    probs = F.softmax(logits, dim=-1)
    top1 = probs.max(dim=-1).values
    ent = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
    p_mask = probs[:, mask_id]
    am = logits.argmax(dim=-1)
    return {
        "argmax": am.cpu().tolist(),
        "top1_prob": top1.cpu().tolist(),
        "entropy": ent.cpu().tolist(),
        "p_mask": p_mask.cpu().tolist(),
    }


@torch.no_grad()
def main():
    args = parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 device_map="cuda:0")
    model.eval()

    if args.gen_prefix_jsonl:
        assert args.prompts_jsonl
        gen_prefixes(args, tok, model)
        return

    assert args.prefix_jsonl and args.out_jsonl
    rng = random.Random(args.seed)
    K = args.K
    rows = []
    with open(args.out_jsonl, "w") as f:
        for line in open(args.prefix_jsonl):
            r = json.loads(line)
            ctx = r["ids"]
            # checkpoint's OWN greedy continuation of the fixed prefix (reference)
            inp = torch.tensor([ctx], dtype=torch.long, device=model.device)
            own = model.generate(inp, max_new_tokens=K, do_sample=False,
                                 pad_token_id=151643)[0, len(ctx):].tolist()

            rec = {"batch_idx": r["batch_idx"], "own_greedy": own, "tails": {}}

            uni = [rng.randrange(args.vocab_size) for _ in range(K)]
            rec["tails"]["uniform"] = window_stats(model, ctx, uni, args.mask_id)
            msk = [args.mask_id] * K
            rec["tails"]["mask"] = window_stats(model, ctx, msk, args.mask_id)

            # cascade: iterate Jacobi from uniform init, record every iter
            draft = list(uni)
            iters = []
            for _ in range(args.n_iters):
                s = window_stats(model, ctx, draft, args.mask_id)
                iters.append(s)
                draft = s["argmax"]
            rec["tails"]["cascade_iters"] = iters

            for name in ("uniform", "mask"):
                s = rec["tails"][name]
                s["agree_self"] = [int(a == g) for a, g in zip(s["argmax"], own)]
            for s in rec["tails"]["cascade_iters"]:
                s["agree_self"] = [int(a == g) for a, g in zip(s["argmax"], own)]

            rows.append(rec)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"[sharp] prompt {r['batch_idx']} done", flush=True)

    # quick aggregate to stdout
    def agg(get):
        import statistics
        K_ = K
        out = []
        for j in range(K_):
            out.append(sum(get(rec)[j] for rec in rows) / len(rows))
        return out

    for name in ("uniform", "mask"):
        pm = agg(lambda rec: rec["tails"][name]["p_mask"])
        t1 = agg(lambda rec: rec["tails"][name]["top1_prob"])
        print(f"[agg] {name}: mean p_mask by pos  = " + " ".join(f"{x:.2f}" for x in pm))
        print(f"[agg] {name}: mean top1_p by pos  = " + " ".join(f"{x:.2f}" for x in t1))
    last = len(rows[0]["tails"]["cascade_iters"]) - 1
    pm = agg(lambda rec: rec["tails"]["cascade_iters"][last]["p_mask"])
    t1 = agg(lambda rec: rec["tails"]["cascade_iters"][last]["top1_prob"])
    print(f"[agg] cascade_iter{last}: mean p_mask by pos = " + " ".join(f"{x:.2f}" for x in pm))
    print(f"[agg] cascade_iter{last}: mean top1_p by pos = " + " ".join(f"{x:.2f}" for x in t1))


if __name__ == "__main__":
    main()
