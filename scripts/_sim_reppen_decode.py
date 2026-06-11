"""Full-decode eval of the repetition-penalized draft policy (reppen).

Identical to stop-fixed vanilla jsim (streaming Jacobi, 1 fwd/iter, random
reinit), except the shifted portion of the next draft is built from the
forward's TOP-2 tokens with a window-repetition penalty: take argmax unless it
equals one of the 2 previously chosen window tokens, in which case take the
runner-up. Acceptance still requires exact match vs the model's argmax, so a
penalized token only commits if the next forward's argmax agrees with it.
"""
from __future__ import annotations
import argparse, json, random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

STOP_IDS = {151645, 151643}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompts_jsonl", required=True)
    p.add_argument("--out_jsonl", required=True)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--max_new", type=int, default=512)
    p.add_argument("--max_iters", type=int, default=512)
    p.add_argument("--policy", choices=["vanilla", "reppen", "gumbel", "confgumbel", "gumbelfrozen", "gumbelfar"],
                   default="reppen")
    p.add_argument("--lookback", type=int, default=2,
                   help="penalize argmax that equals any of the previous N chosen window tokens")
    p.add_argument("--temp", type=float, default=0.7,
                   help="gumbel/confgumbel: draft sampling temperature")
    p.add_argument("--tau", type=float, default=1.0,
                   help="confgumbel: entropy threshold — argmax below, gumbel-sample above")
    p.add_argument("--prompt_field", default="input")
    p.add_argument("--n_prompts", type=int, default=0, help="0 = all")
    p.add_argument("--vocab_size", type=int, default=152064)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def decode(model, prompt_ids, args, rng):
    K = args.K
    LB = args.lookback
    committed = list(prompt_ids)
    n_prompt = len(committed)
    draft = [rng.randrange(args.vocab_size) for _ in range(K)]
    total, n_fwd = 0, 0
    # instrumentation: each penalty fire = (abs_pos, runnerup_tok, argmax_tok)
    fires = []
    n_window_pos = 0
    while total < args.max_new and n_fwd < args.max_iters:
        L = len(committed)
        inp = torch.tensor([committed + draft], dtype=torch.long, device=model.device)
        logits = model(input_ids=inp).logits[0, L - 1 : L - 1 + K, :]
        top2 = logits.topk(2, dim=-1).indices.cpu().tolist()
        cur = [t[0] for t in top2]
        n_fwd += 1
        n_acc = 0
        for a, b in zip(cur, draft):
            if a != b:
                break
            n_acc += 1
        if n_acc > 0:
            toks = cur[:n_acc]
            hit = False
            for si, t in enumerate(toks):
                if t in STOP_IDS:
                    toks = toks[: si + 1]; hit = True; break
            committed += toks
            total += len(toks)
            if hit or total >= args.max_new:
                break
        # next draft: shifted region from cur (argmax), reppen(top2), or
        # gumbel-sampled candidates (acceptance above stays ARGMAX-based, so
        # committed text remains exactly the greedy AR continuation).
        if args.policy == "reppen":
            shifted = []
            for j in range(n_acc, K):
                t = top2[j][0]
                n_window_pos += 1
                if LB > 0 and shifted[-LB:] and t in shifted[-LB:]:
                    t = top2[j][1]
                    # abs response position this window slot would commit at
                    fires.append((total + (j - n_acc), t, top2[j][0]))
                shifted.append(t)
        elif args.policy == "gumbelfar":
            # far-only frozen gumbel: argmax within `lookahead` of the commit
            # frontier (self-releasing, no deadlock), frozen-noise sampling
            # beyond (iteration-stable diversification of the scaffolding).
            FAR = 8
            sl = logits[n_acc:K].float()
            shifted = []
            for j in range(K - n_acc):
                if j < FAR:
                    shifted.append(cur[n_acc + j])
                else:
                    gen = torch.Generator(device=sl.device)
                    gen.manual_seed(args.seed * 1000003 + (total + j))
                    u = torch.rand(sl.shape[-1], generator=gen, device=sl.device)
                    gg = -torch.log(-torch.log(u))
                    shifted.append(int((sl[j] / args.temp + gg).argmax().item()))
        elif args.policy == "gumbelfrozen":
            # noise-reuse: Gumbel noise is a deterministic function of the
            # ABSOLUTE position, so the draft policy is iteration-stable and
            # the Jacobi fixed-point engine still works.
            sl = logits[n_acc:K].float()
            rows = []
            for j in range(K - n_acc):
                gen = torch.Generator(device=sl.device)
                gen.manual_seed(args.seed * 1000003 + (total + j))
                u = torch.rand(sl.shape[-1], generator=gen, device=sl.device)
                rows.append(-torch.log(-torch.log(u)))
            g = torch.stack(rows)
            shifted = (sl / args.temp + g).argmax(dim=-1).cpu().tolist()
        elif args.policy in ("gumbel", "confgumbel"):
            sl = logits[n_acc:K].float()
            g = -torch.log(-torch.log(torch.rand_like(sl)))
            samp = (sl / args.temp + g).argmax(dim=-1).cpu().tolist()
            if args.policy == "gumbel":
                shifted = samp
            else:
                lp = torch.log_softmax(sl, dim=-1)
                ent = (-(lp.exp() * lp).sum(dim=-1)).cpu().tolist()
                shifted = [cur[n_acc + j] if ent[j] <= args.tau else samp[j]
                           for j in range(K - n_acc)]
                n_window_pos += K - n_acc
                fires.extend((total + j, shifted[j], cur[n_acc + j])
                             for j in range(K - n_acc) if ent[j] > args.tau)
        else:
            shifted = cur[n_acc:K]
        draft = shifted + [rng.randrange(args.vocab_size) for _ in range(n_acc)]
    # post-hoc accounting: at fired positions, what token was finally committed?
    resp = committed[n_prompt:]
    acct = {"runnerup_right": 0, "argmax_right": 0, "neither": 0, "unreached": 0}
    for pos, ru, am in fires:
        if pos >= len(resp):
            acct["unreached"] += 1
        elif resp[pos] == ru:
            acct["runnerup_right"] += 1
        elif resp[pos] == am:
            acct["argmax_right"] += 1
        else:
            acct["neither"] += 1
    return total, n_fwd, len(fires), n_window_pos, acct


def main():
    args = parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 device_map="cuda:0")
    model.eval()
    prompts = [json.loads(l) for l in open(args.prompts_jsonl)]
    if args.n_prompts:
        prompts = prompts[: args.n_prompts]
    rng = random.Random(args.seed)
    rows = []
    agg = {"runnerup_right": 0, "argmax_right": 0, "neither": 0, "unreached": 0}
    tot_fires = tot_wpos = 0
    with open(args.out_jsonl, "w") as f:
        for i, p in enumerate(prompts):
            chat = [{"role": "user", "content": p[args.prompt_field]}]
            text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            pid = tok(text, return_tensors="pt").input_ids[0].tolist()
            tot, fwd_n, nf, nw, acct = decode(model, pid, args, rng)
            rows.append((tot, fwd_n))
            tot_fires += nf; tot_wpos += nw
            for k in agg: agg[k] += acct[k]
            f.write(json.dumps({"batch_idx": i, "n_tokens": tot, "n_forwards": fwd_n,
                                "tpf": tot / max(1, fwd_n), "fires": nf, "acct": acct}) + "\n")
            print(f"[reppen] [{i+1}/{len(prompts)}] {args.policy} lb={args.lookback} "
                  f"tok={tot} fwd={fwd_n} TPF={tot/max(1,fwd_n):.3f} fires={nf}", flush=True)
    T = sum(t for t, _ in rows); F_ = sum(f_ for _, f_ in rows)
    print(f"[reppen] {args.policy} lb={args.lookback} CORPUS TPF = {T}/{F_} = {T/F_:.4f}  "
          f"per-prompt mean = {sum(t/max(1,f_) for t,f_ in rows)/len(rows):.4f}")
    if tot_fires:
        print(f"[reppen] fire_rate = {tot_fires}/{tot_wpos} = {tot_fires/max(1,tot_wpos):.3f}  "
              f"accounting: {agg}  "
              f"(runnerup_right = penalty placed the CORRECT token; "
              f"argmax_right = penalty displaced the correct token)")


if __name__ == "__main__":
    main()
