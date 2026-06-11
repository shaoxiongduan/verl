"""Within-completion TPF split: repeating vs non-repeating segments.

Standard vanilla Jacobi block decode (1 forward/iter, random reinit, greedy
argmax — same protocol as _sim_jacobi_predictor_refresh refresh=none), but
records every forward's n_acc and the committed token ids.

Post-hoc, committed tokens are labeled REPEATING if the REP_N-token n-gram
ending at that token occurred earlier in the response. Forwards are grouped
into commit cycles (consecutive zero-commit forwards + the committing
forward); a cycle is REPEATING if >=half of its committed tokens are flagged.

  TPF_rep   = sum(tokens committed by rep cycles)   / sum(forwards in rep cycles)
  TPF_clean = same over clean cycles

This answers: is the model's high TPF earned on repetitive text?
"""
from __future__ import annotations
import argparse, json, random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REP_N = 8


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompts_jsonl", required=True)
    p.add_argument("--out_jsonl", required=True)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--max_new", type=int, default=2048)
    p.add_argument("--max_iters", type=int, default=1024)
    p.add_argument("--vocab_size", type=int, default=152064)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def decode(model, prompt_ids, args, rng):
    K = args.K
    stop_ids = {151645, 151643}
    committed = list(prompt_ids)
    n_prompt = len(committed)
    draft = [rng.randrange(args.vocab_size) for _ in range(K)]
    cycles = []  # list of (n_forwards_in_cycle, [token ids committed])
    fwd_in_cycle = 0
    total, n_fwd = 0, 0
    while total < args.max_new and n_fwd < args.max_iters:
        L = len(committed)
        inp = torch.tensor([committed + draft], dtype=torch.long, device=model.device)
        cur = model(input_ids=inp).logits[0, L - 1 : L - 1 + K, :].argmax(dim=-1).cpu().tolist()
        n_fwd += 1
        fwd_in_cycle += 1
        n_acc = 0
        for j in range(K):
            if cur[j] == draft[j]:
                n_acc += 1
            else:
                break
        if n_acc > 0:
            toks = [int(t) for t in cur[:n_acc]]
            # Truncate the commit at the FIRST stop token (mid-block stops count;
            # checking only the last token lets generation run past <|im_end|>).
            hit_stop = False
            for si, t in enumerate(toks):
                if t in stop_ids:
                    toks = toks[: si + 1]
                    hit_stop = True
                    break
            committed += toks
            total += len(toks)
            cycles.append((fwd_in_cycle, toks))
            fwd_in_cycle = 0
            if hit_stop or total >= args.max_new:
                break
        shifted = cur[n_acc:K]
        draft = shifted + [rng.randrange(args.vocab_size) for _ in range(n_acc)]
    if fwd_in_cycle:
        cycles.append((fwd_in_cycle, []))  # trailing no-commit forwards
    return committed[n_prompt:], cycles, n_fwd, total


def label_repeating(resp_ids):
    """LOOSE criterion: flag[i]=True if the REP_N-gram ending at i appeared
    earlier ANYWHERE in the response. Catches structural reuse (formula
    patterns, restated steps) as well as loops."""
    flag = [False] * len(resp_ids)
    seen = set()
    for i in range(len(resp_ids) - REP_N + 1):
        g = tuple(resp_ids[i:i + REP_N])
        if g in seen:
            for j in range(i, i + REP_N):
                flag[j] = True
        seen.add(g)
    return flag


def label_loops(resp_ids, max_p=256, min_run=32):
    """STRICT degenerate-loop criterion: flag positions inside a sustained
    periodic self-repeat — ids[i] == ids[i-p] continuously for at least
    max(2*p, min_run) tokens for some period p<=max_p. This is the
    "model is stuck cycling the same content" signature; one-off distant
    reuse of an 8-gram does NOT trigger it."""
    n = len(resp_ids)
    flag = [False] * n
    for p in range(1, min(max_p, n // 2) + 1):
        need = max(2 * p, min_run)
        run = 0
        for i in range(p, n):
            if resp_ids[i] == resp_ids[i - p]:
                run += 1
                if run == need:
                    for j in range(i - run + 1, i + 1):
                        flag[j] = True
                elif run > need:
                    flag[i] = True
            else:
                run = 0
    return flag


def main():
    args = parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 device_map="cuda:0")
    model.eval()
    prompts = [json.loads(l) for l in open(args.prompts_jsonl)]
    rng = random.Random(args.seed)
    crits = {"rep8": label_repeating, "loop": label_loops}
    agg = {c: {"rep_tok": 0, "rep_fwd": 0, "cl_tok": 0, "cl_fwd": 0} for c in crits}
    with open(args.out_jsonl, "w") as f:
        for i, p in enumerate(prompts):
            chat = [{"role": "user", "content": p["input"]}]
            text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            pid = tok(text, return_tensors="pt").input_ids[0].tolist()
            resp, cycles, n_fwd, total = decode(model, pid, args, rng)
            row = {"batch_idx": i, "n_tokens": total, "n_forwards": n_fwd,
                   "resp_ids": resp,
                   "cycles": [[nf, len(toks)] for nf, toks in cycles]}
            for cname, labeler in crits.items():
                flag = labeler(resp)
                pos = 0
                r = {"rep_tok": 0, "rep_fwd": 0, "cl_tok": 0, "cl_fwd": 0}
                for nf, toks in cycles:
                    if not toks:    # trailing no-commit forwards: clean bucket
                        r["cl_fwd"] += nf
                        continue
                    fl = flag[pos:pos + len(toks)]
                    pos += len(toks)
                    if sum(fl) * 2 >= len(fl):
                        r["rep_tok"] += len(toks); r["rep_fwd"] += nf
                    else:
                        r["cl_tok"] += len(toks); r["cl_fwd"] += nf
                for k in agg[cname]: agg[cname][k] += r[k]
                row[cname] = {**r, "frac": sum(flag) / max(1, len(flag))}
            f.write(json.dumps(row) + "\n"); f.flush()
            print(f"[rsplit] [{i+1}/{len(prompts)}] tok={total} fwd={n_fwd} "
                  f"rep8_frac={row['rep8']['frac']:.2f} loop_frac={row['loop']['frac']:.2f}",
                  flush=True)
    for cname, a in agg.items():
        print(f"\n[rsplit] CORPUS [{cname}]: rep TPF = {a['rep_tok']}/{a['rep_fwd']} = "
              f"{a['rep_tok']/max(1,a['rep_fwd']):.3f}   clean TPF = {a['cl_tok']}/{a['cl_fwd']} = "
              f"{a['cl_tok']/max(1,a['cl_fwd']):.3f}   rep token share = "
              f"{a['rep_tok']/max(1,a['rep_tok']+a['cl_tok']):.3f}")


if __name__ == "__main__":
    main()
