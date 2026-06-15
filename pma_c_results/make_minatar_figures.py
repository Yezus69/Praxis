"""Merge continual-MinAtar results.json files (across seed splits) -> markdown table + figure.

Usage: python make_minatar_figures.py <json1,json2,...> <out.png>
Presents BOTH per-game retention (final/peak) and mean final return (captures plasticity).
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, GRAY, GREEN = "#1f77b4", "#ff7f0e", "#7f7f7f", "#2ca02c"


def load_all(paths):
    """Return {mode: {"matrices":[...], "finals":[...], "learned":[...], "metrics":[...]}} merged over seeds."""
    games = None
    agg = {}
    for p in paths:
        d = json.load(open(p, encoding="utf-8"))
        games = d.get("games", games)
        for seed, run in d["runs"].items():
            for mode, res in run["results"].items():
                a = agg.setdefault(mode, {"matrices": [], "finals": [], "learned": [], "metrics": []})
                a["matrices"].append(np.array(res["return_matrix"], dtype=float))
                a["finals"].append(np.array(res["final_returns"], dtype=float))
                a["learned"].append(np.array(res["learned_returns"], dtype=float))
                a["metrics"].append(res["metrics"])
    return games, agg


def stat(vals):
    v = np.asarray(vals, dtype=float)
    return float(v.mean()), float(v.std())


def main():
    paths = sys.argv[1].split(",")
    out = sys.argv[2]
    games, agg = load_all(paths)
    gnames = [g.replace("-MinAtar", "") for g in (games or [])]
    modes = [m for m in ["baseline", "pmac", "pmac_no_conservation", "pmac_no_replay"] if m in agg]
    T = len(gnames)
    n_seeds = len(agg["baseline"]["finals"]) if "baseline" in agg else 0

    # Aggregate
    print(f"### Continual MinAtar — {T} games, {n_seeds} seeds\n")
    print("| mode | mean final return | mean retention | worst retention | forgetting |")
    print("|" + "---|" * 5)
    summ = {}
    for m in modes:
        mf = stat([x["mean_final_return"] for x in agg[m]["metrics"]])
        mr = stat([x["mean_retention"] for x in agg[m]["metrics"]])
        wr = stat([x["worst_retention"] for x in agg[m]["metrics"]])
        fg = stat([x["forgetting"] for x in agg[m]["metrics"]])
        summ[m] = dict(mf=mf, mr=mr, wr=wr, fg=fg)
        print(f"| {m} | {mf[0]:.1f} ± {mf[1]:.1f} | {mr[0]:.3f} ± {mr[1]:.3f} | "
              f"{wr[0]:.3f} ± {wr[1]:.3f} | {fg[0]:.1f} ± {fg[1]:.1f} |")

    # Per-game final + retention (baseline vs pmac)
    print("\n### Per-game final return (mean over seeds): learned -> final\n")
    print("| game | baseline learned→final (retention) | PMA-C learned→final (retention) |")
    print("|" + "---|" * 3)
    for j, g in enumerate(gnames):
        bl_l = np.mean([x[j] for x in agg["baseline"]["learned"]])
        bl_f = np.mean([x[j] for x in agg["baseline"]["finals"]])
        pm_l = np.mean([x[j] for x in agg["pmac"]["learned"]])
        pm_f = np.mean([x[j] for x in agg["pmac"]["finals"]])
        print(f"| {g} | {bl_l:.1f}→{bl_f:.1f} ({bl_f/max(bl_l,1e-9):.2f}) | "
              f"{pm_l:.1f}→{pm_f:.1f} ({pm_f/max(pm_l,1e-9):.2f}) |")

    # Figure
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    # Panel 1: per-game RETENTION bars (scale-invariant), baseline vs pmac, error bars over seeds
    x = np.arange(T)
    def per_game_ret(mode):
        # retention per game per seed = final/peak
        rets = []
        for k in range(len(agg[mode]["finals"])):
            M = agg[mode]["matrices"][k]
            final = M[-1]; peak = M.max(axis=0)
            rets.append(final / np.maximum(peak, 1e-9))
        rets = np.array(rets)
        return rets.mean(0), rets.std(0)
    bm, bs = per_game_ret("baseline"); pm, ps = per_game_ret("pmac")
    ax[0].bar(x - 0.2, bm, 0.4, yerr=bs, label="baseline", color=BLUE, capsize=3)
    ax[0].bar(x + 0.2, pm, 0.4, yerr=ps, label="PMA-C", color=ORANGE, capsize=3)
    ax[0].set_title("Per-game retention after all games (final/peak)")
    ax[0].set_xticks(x); ax[0].set_xticklabels(gnames, rotation=20, ha="right")
    ax[0].set_ylabel("retention"); ax[0].set_ylim(0, 1.1); ax[0].legend(); ax[0].grid(axis="y", alpha=0.3)

    # Panel 2: game-0 return across training (the forgetting curve), normalized to its peak
    def game0_curve(mode):
        curves = []
        for M in agg[mode]["matrices"]:
            c = M[:, 0]
            curves.append(c / max(c.max(), 1e-9))
        c = np.array(curves)
        return c.mean(0), c.std(0)
    b0m, b0s = game0_curve("baseline"); p0m, p0s = game0_curve("pmac")
    xs = np.arange(T)
    ax[1].plot(xs, b0m, "-o", color=BLUE, label="baseline"); ax[1].fill_between(xs, b0m-b0s, b0m+b0s, color=BLUE, alpha=0.15)
    ax[1].plot(xs, p0m, "-o", color=ORANGE, label="PMA-C"); ax[1].fill_between(xs, p0m-p0s, p0m+p0s, color=ORANGE, alpha=0.15)
    ax[1].set_title(f"{gnames[0]} (game 0) score across the training sequence")
    ax[1].set_xlabel("after training game #"); ax[1].set_ylabel(f"{gnames[0]} score (norm. to peak)")
    ax[1].set_ylim(0, 1.1); ax[1].legend(); ax[1].grid(alpha=0.3)

    mfb, mfp = summ["baseline"]["mf"][0], summ["pmac"]["mf"][0]
    fig.suptitle(f"Continual MinAtar (hard RL) — PMA-C vs Baseline   |   "
                 f"mean final return {mfb:.1f}→{mfp:.1f}   mean retention "
                 f"{summ['baseline']['mr'][0]:.2f}→{summ['pmac']['mr'][0]:.2f}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=120); print("\nwrote", out)


if __name__ == "__main__":
    main()
