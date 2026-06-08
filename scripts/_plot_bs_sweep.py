"""Plot DFlash vs AR TPS scaling vs BS."""
import argparse, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--title", default="vLLM TPS scaling — Qwen3-8B on H200")
    args = p.parse_args()

    data = {"dflash": [], "ar": []}
    with open(args.csv) as f:
        for r in csv.DictReader(f):
            data[r["mode"]].append((int(r["bs"]), float(r["tps"])))
    for k in data:
        data[k].sort()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for mode, marker, color in [("dflash", "o", "tab:blue"),
                                ("ar", "s", "tab:orange")]:
        if not data[mode]:
            continue
        xs = [b for b, _ in data[mode]]
        ys = [t for _, t in data[mode]]
        ax1.plot(xs, ys, marker=marker, color=color,
                 label=f"vLLM + {'DFlash' if mode == 'dflash' else 'AR'}")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("max_num_seqs (batch size cap)")
    ax1.set_ylabel("TPS (tok/s, aggregate)")
    ax1.set_title("Throughput vs BS")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    if data["dflash"] and data["ar"]:
        d_map = dict(data["dflash"]); a_map = dict(data["ar"])
        bs_common = sorted(set(d_map) & set(a_map))
        speedups = [d_map[b] / a_map[b] for b in bs_common]
        ax2.plot(bs_common, speedups, marker="^", color="tab:green",
                 label="DFlash / AR")
        ax2.set_xscale("log", base=2)
        ax2.set_xlabel("max_num_seqs (batch size cap)")
        ax2.set_ylabel("Speedup ×")
        ax2.set_title("DFlash speedup vs BS")
        ax2.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

    fig.suptitle(args.title)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
