"""Parse verl stdout logs into per-step metric tables for the mask-tail runs."""
import re, sys, json

KEYS = {
    "tpf": "actor/rollout/jacobi_tpf_mean",
    "tpf_correct": "actor/rollout_correct/jacobi_tpf_mean",
    "tok_per_req": "actor/rollout/jacobi_tok_per_req",
    "iters_per_req": "actor/rollout/jacobi_iters_per_req",
    "entropy": "actor/entropy",
    "score": "critic/score/mean",
    "resp_len": "response_length/mean",
    "cons_loss": "actor/cons_loss",
    "cons_correct_frac": "actor/cons_correct_fraction",
    "pg_loss": "actor/pg_loss",
    "kl": "actor/ppo_kl",
    "grad_norm": "actor/grad_norm",
}

def parse(path):
    rows = {}
    pat_step = re.compile(r"step:(\d+) - ")
    for line in open(path, errors="replace"):
        m = pat_step.search(line)
        if not m or "global_seqlen" not in line:
            continue
        step = int(m.group(1))
        row = rows.setdefault(step, {})
        for short, key in KEYS.items():
            km = re.search(re.escape(key) + r":(?:np\.float64\()?(-?[0-9.eE+-]+)\)?(?:\s|$|-)", line)
            if km:
                try:
                    row[short] = float(km.group(1).rstrip(".-"))
                except ValueError:
                    pass
    return rows

if __name__ == "__main__":
    for path in sys.argv[1:]:
        rows = parse(path)
        out = path.replace(".log", "_metrics.jsonl")
        with open(out, "w") as f:
            for step in sorted(rows):
                f.write(json.dumps({"step": step, **rows[step]}) + "\n")
        print(f"{path}: {len(rows)} steps -> {out}")
