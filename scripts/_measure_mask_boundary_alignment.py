"""Measure alignment between the drafter's self-declared mask boundary and the
Jacobi acceptance boundary, under VANILLA decode (random reinit, the production
setting for mask-tail models).

Per iteration, with draft = this iter's input window:
  draft_first_mask = first position j with draft[j] == mask_id  (K if none)
  n_acc            = standard Jacobi prefix-acceptance length
Classification (only iters where draft_first_mask < K, i.e. the model put a
mask boundary inside the window):
  exact   : n_acc == draft_first_mask    -> every non-mask token accepted,
                                            rejection happens exactly at the mask
  leak    : n_acc <  draft_first_mask    -> a REAL token was rejected before the
                                            mask boundary (boundary over-claims)
  through : n_acc >  draft_first_mask    -> committed THROUGH a mask==mask match
                                            (vanilla mode allows it; counts + stops)
Also tracks stop reason: eos / pad(mask) commit / max_new / max_iters.
"""
from __future__ import annotations
import argparse, json, random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompts_jsonl", required=True)
    p.add_argument("--out_jsonl", required=True)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--max_new", type=int, default=512)
    p.add_argument("--max_iters", type=int, default=512)
    p.add_argument("--mask_id", type=int, default=151643)
    p.add_argument("--vocab_size", type=int, default=152064)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def run(model, prompt_ids, args, rng):
    K = args.K
    eos_id, mask_id = 151645, args.mask_id
    committed = list(prompt_ids)
    draft = [rng.randrange(args.vocab_size) for _ in range(K)]
    total, n_fwd = 0, 0
    iters = []
    stop = "max"
    while total < args.max_new and n_fwd < args.max_iters:
        L = len(committed)
        inp = torch.tensor([committed + draft], dtype=torch.long, device=model.device)
        cur = inp.new_zeros(0)
        cur = model(input_ids=inp).logits[0, L - 1 : L - 1 + K, :].argmax(dim=-1).cpu().tolist()
        n_fwd += 1

        dfm = draft.index(mask_id) if mask_id in draft else K
        n_acc = 0
        for j in range(K):
            if cur[j] == draft[j]:
                n_acc += 1
            else:
                break
        rej_is_mask = n_acc < K and draft[n_acc] == mask_id

        iters.append({"draft_first_mask": dfm, "n_acc": n_acc,
                      "rej_at_mask": bool(rej_is_mask),
                      "out_first_mask": cur.index(mask_id) if mask_id in cur else K})

        if n_acc > 0:
            committed += [int(t) for t in cur[:n_acc]]
            total += n_acc
            if committed[-1] == eos_id:
                stop = "eos"; break
            if committed[-1] == mask_id:
                stop = "mask_commit"; break
            if total >= args.max_new:
                stop = "max_new"; break
        shifted = cur[n_acc:K]
        draft = shifted + [rng.randrange(args.vocab_size) for _ in range(n_acc)]
    return {"n_tokens": total, "n_forwards": n_fwd, "tpf": total / max(1, n_fwd),
            "stop": stop, "iters": iters}


def main():
    args = parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    prompts = [json.loads(l) for l in open(args.prompts_jsonl)]
    rng = random.Random(args.seed)
    rows = []
    with open(args.out_jsonl, "w") as f:
        for i, p in enumerate(prompts):
            chat = [{"role": "user", "content": p["input"]}]
            text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            pid = tok(text, return_tensors="pt").input_ids[0].tolist()
            r = run(model, pid, args, rng)
            r["batch_idx"] = i
            rows.append(r)
            f.write(json.dumps(r) + "\n"); f.flush()
            print(f"[align] [{i+1}/{len(prompts)}] tok={r['n_tokens']} fwd={r['n_forwards']} "
                  f"TPF={r['tpf']:.3f} stop={r['stop']}", flush=True)

    # aggregate
    all_iters = [it for r in rows for it in r["iters"]]
    with_b = [it for it in all_iters if it["draft_first_mask"] < args.K]
    exact = sum(1 for it in with_b if it["n_acc"] == it["draft_first_mask"])
    leak = [it["draft_first_mask"] - it["n_acc"] for it in with_b if it["n_acc"] < it["draft_first_mask"]]
    through = sum(1 for it in with_b if it["n_acc"] > it["draft_first_mask"])
    rejm = sum(1 for it in all_iters if it["rej_at_mask"])
    print(f"\n[align] iters total={len(all_iters)}  with_mask_boundary={len(with_b)} "
          f"({100*len(with_b)/max(1,len(all_iters)):.0f}%)")
    print(f"[align] exact (all non-mask accepted, rejected AT mask): {exact} "
          f"({100*exact/max(1,len(with_b)):.1f}%)")
    print(f"[align] leak  (real token rejected before mask): {len(leak)} "
          f"({100*len(leak)/max(1,len(with_b)):.1f}%)  mean_leak={sum(leak)/max(1,len(leak)):.2f}  "
          f"leak_hist={ {d: leak.count(d) for d in sorted(set(leak))[:8]} }")
    print(f"[align] through-mask commits: {through}")
    print(f"[align] rejection lands on a mask token: {rejm}/{len(all_iters)} "
          f"({100*rejm/max(1,len(all_iters)):.1f}%)")
    print(f"[align] stops: { {s: sum(1 for r in rows if r['stop']==s) for s in set(r['stop'] for r in rows)} }")
    print(f"[align] corpus TPF = {sum(r['n_tokens'] for r in rows)/max(1,sum(r['n_forwards'] for r in rows)):.4f}")


if __name__ == "__main__":
    main()
