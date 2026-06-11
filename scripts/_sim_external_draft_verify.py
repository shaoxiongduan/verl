"""Feasibility probe for the free-running-drafter + AR-verifier architecture.

A DRAFTER model produces a fluent greedy trace D (stand-in for a diffusion
drafter that has converged its canvas — fluency upper bound, alignment
unknown). A VERIFIER consumes D in K-token windows with spec-decode
semantics + bonus correction:

  per verifier forward over [committed | D-window]:
    n_acc = prefix where verifier argmax == draft
    commit draft[:n_acc] + the verifier's correction token (bonus)
    then RESYNC the draft stream (two models of drafter behavior):
      resync=skip   : drafter perfectly resyncs — drop its token at the
                      mismatch position, continue its trace (optimistic)
      resync=none   : drafter trace is immutable — divergence accumulates
                      (pessimistic)

Reports tokens-per-VERIFIER-forward (drafter assumed cheap), accepted-run
distribution, and positional agreement p(draft == verifier greedy).
Verifier-side commits use verifier argmax (greedy law preserved).
"""
from __future__ import annotations
import argparse, json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

STOP_IDS = {151645, 151643}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--drafter", required=True)
    p.add_argument("--verifier", required=True)
    p.add_argument("--prompts_jsonl", required=True)
    p.add_argument("--out_jsonl", required=True)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--max_new", type=int, default=512)
    p.add_argument("--resync", choices=["static", "live"], default="static")
    p.add_argument("--n_prompts", type=int, default=16)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    tok = AutoTokenizer.from_pretrained(args.verifier)
    drafter = AutoModelForCausalLM.from_pretrained(args.drafter, dtype=torch.bfloat16,
                                                   device_map="cuda:0").eval()
    verifier = AutoModelForCausalLM.from_pretrained(args.verifier, dtype=torch.bfloat16,
                                                    device_map="cuda:0").eval()
    prompts = [json.loads(l) for l in open(args.prompts_jsonl)][: args.n_prompts]
    K = args.K
    tot_t = tot_f = 0
    runs = []
    with open(args.out_jsonl, "w") as f:
        for i, p in enumerate(prompts):
            chat = [{"role": "user", "content": p["input"]}]
            text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            pid = tok(text, return_tensors="pt").input_ids.to(drafter.device)
            # drafter free-runs: fluent greedy trace (KV-cached)
            D = drafter.generate(pid, max_new_tokens=args.max_new + K, do_sample=False,
                                 pad_token_id=151643)[0, pid.shape[1]:].tolist()
            committed = pid[0].tolist()
            n0 = len(committed)
            di = 0          # drafter-trace cursor
            total = n_fwd = 0
            prun = []
            while total < args.max_new and n_fwd < args.max_new and di < len(D):
                win = D[di: di + K]
                if len(win) < K:
                    win = win + [151643] * (K - len(win))
                L = len(committed)
                inp = torch.tensor([committed + win], dtype=torch.long, device=verifier.device)
                logits = verifier(input_ids=inp).logits[0, L - 1: L - 1 + K, :]
                cur = logits.argmax(dim=-1).cpu().tolist()
                n_fwd += 1
                n_acc = 0
                for a, b in zip(cur, win):
                    if a != b:
                        break
                    n_acc += 1
                toks = win[:n_acc] + ([cur[n_acc]] if n_acc < K else [])  # accepted + bonus
                hit = False
                for si, t in enumerate(toks):
                    if t in STOP_IDS:
                        toks = toks[: si + 1]; hit = True; break
                committed += toks
                total += len(toks)
                prun.append(n_acc)
                if hit:
                    break
                di += n_acc + (1 if n_acc < K else 0)   # consumed + superseded-by-correction
                if args.resync == "live" and n_acc < K:
                    # drafter re-anchors: regenerate its trace from the corrected prefix
                    ids = torch.tensor([committed], device=drafter.device)
                    D = drafter.generate(ids, max_new_tokens=args.max_new - total + K,
                                         do_sample=False, pad_token_id=151643
                                         )[0, len(committed):].tolist()
                    di = 0
            runs.extend(prun)
            tot_t += total; tot_f += n_fwd
            f.write(json.dumps({"batch_idx": i, "n_tokens": total, "n_fwd": n_fwd,
                                "tpf_verifier": total / max(1, n_fwd),
                                "runs": prun}) + "\n")
            print(f"[xdv] [{i+1}/{len(prompts)}] tok={total} vfwd={n_fwd} "
                  f"TPFv={total/max(1,n_fwd):.2f} mean_acc={sum(prun)/max(1,len(prun)):.2f}",
                  flush=True)
    import statistics as st
    print(f"[xdv] drafter={args.drafter.split('/')[-1]} verifier={args.verifier.split('/')[-1]} "
          f"resync={args.resync}: TPF_verifier = {tot_t}/{tot_f} = {tot_t/max(1,tot_f):.3f}  "
          f"mean accepted run = {st.mean(runs):.2f}  "
          f"runs>=8: {100*sum(1 for r in runs if r>=8)/len(runs):.0f}%  "
          f"runs=0: {100*sum(1 for r in runs if r==0)/len(runs):.0f}%")


if __name__ == "__main__":
    main()
