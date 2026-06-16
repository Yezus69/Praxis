"""Merge continual-Atari results.json files (across seeds) -> markdown table + figure.

Usage: python make_atari_figures.py <json1,json2,...> <out.png>
Metric = random-NORMALIZED retention (handles signed Atari returns):
  norm_ret[j] = (final[j]-random[j]) / (learned[j]-random[j]); also reports mean final return.
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, GRAY = "#1f77b4", "#ff7f0e", "#7f7f7f"


def load_all(paths):
    games = None
    agg = {}
    for p in paths:
        d = json.load(open(p, encoding="utf-8"))
        games = d.get("games", games)
        for seed, run in d["runs"].items():
            for mode, res in run["results"].items():
                a = agg.setdefault(mode, {"R": [], "learned": [], "final": [], "random": [], "m": []})
                a["R"].append(np.array(res["return_matrix"], dtype=float))
                a["learned"].append(np.array(res["learned"], dtype=float))
                a["final"].append(np.array(res["final"], dtype=float))
                a["random"].append(np.array(res["random_scores"], dtype=float))
                a["m"].append(res["metrics"])
    return games, agg


def norm_ret(final, learned, random):
    """Random-normalized retention, robust to not-learned games.

    A game only counts for retention if it was meaningfully learned (learned clearly
    above its random baseline); otherwise the ratio is degenerate, so we return NaN
    and exclude it from the mean. Clip to [0, 1.2].
    """
    final = np.asarray(final, float); learned = np.asarray(learned, float); random = np.asarray(random, float)
    denom = learned - random
    thr = np.maximum(1.0, 0.15 * np.abs(random))  # learned must beat random by a clear margin
    nr = np.where(denom > thr, (final - random) / np.where(denom > thr, denom, 1.0), np.nan)
    return np.clip(nr, 0.0, 1.2)


def stat(v):
    v = np.asarray(v, float)
    return float(np.nanmean(v)), float(np.nanstd(v))


def main():
    paths = sys.argv[1].split(",")
    out = sys.argv[2]
    games, agg = load_all(paths)
    gnames = [g.replace("-v5", "") for g in (games or [])]
    T = len(gnames)
    modes = [m for m in ["baseline", "pmac", "pmac_no_conservation", "pmac_no_replay"] if m in agg]
    nseeds = len(agg["baseline"]["R"]) if "baseline" in agg else 0

    print(f"### Continual full-Atari — {T} games, {nseeds} seeds\n")
    print("| mode | mean norm. retention | worst norm. retention | mean final return |")
    print("|" + "---|" * 4)
    summ = {}
    for m in modes:
        # recompute per-seed normalized retention from matrices (robust to signed returns)
        mr_seed, wr_seed = [], []
        for k in range(len(agg[m]["R"])):
            R = agg[m]["R"][k]
            learned = np.diag(R); final = R[-1]; rand = agg[m]["random"][k]
            nr = norm_ret(final, learned, rand)
            mr_seed.append(np.nanmean(nr)); wr_seed.append(np.nanmin(nr))
        mf = stat([x["mean_final_return"] for x in agg[m]["m"]])
        summ[m] = dict(mr=stat(mr_seed), wr=stat(wr_seed), mf=mf)
        print(f"| {m} | {summ[m]['mr'][0]:.3f} ± {summ[m]['mr'][1]:.3f} | "
              f"{summ[m]['wr'][0]:.3f} ± {summ[m]['wr'][1]:.3f} | {mf[0]:.1f} ± {mf[1]:.1f} |")

    print("\n### Per-game (mean over seeds): random / learned / final  (norm. retention)\n")
    print("| game | baseline learned→final (norm.ret) | PMA-C learned→final (norm.ret) |")
    print("|" + "---|" * 3)
    for j, g in enumerate(gnames):
        def cell(mode):
            # average per-seed retention (robust) rather than retention-of-the-average
            R = np.mean(agg[mode]["R"], axis=0); rnd = np.mean(agg[mode]["random"], axis=0)
            learned = np.diag(R)[j]; final = R[-1][j]
            nrs = []
            for k in range(len(agg[mode]["R"])):
                Rk = agg[mode]["R"][k]; rk = agg[mode]["random"][k]
                nrs.append(norm_ret(Rk[-1], np.diag(Rk), rk)[j])
            nr = np.nanmean(nrs)
            nrtxt = "—" if np.isnan(nr) else f"{nr:.2f}"
            return f"{learned:.0f}→{final:.0f} ({nrtxt})"
        print(f"| {g} | {cell('baseline')} | {cell('pmac')} |")

    # Figure: per-game normalized retention bars + game-0 normalized score across training
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    x = np.arange(T)
    def per_game_nr(mode):
        nrs = []
        for k in range(len(agg[mode]["R"])):
            R = agg[mode]["R"][k]; learned = np.diag(R); final = R[-1]; rnd = agg[mode]["random"][k]
            nrs.append(norm_ret(final, learned, rnd))
        nrs = np.array(nrs)
        return np.nanmean(nrs, 0), np.nanstd(nrs, 0)
    bm, bs = per_game_nr("baseline"); pm, ps = per_game_nr("pmac")
    ax[0].bar(x - 0.2, bm, 0.4, yerr=bs, label="baseline", color=BLUE, capsize=3)
    ax[0].bar(x + 0.2, pm, 0.4, yerr=ps, label="PMA-C", color=ORANGE, capsize=3)
    ax[0].set_title("Per-game normalized retention after all games")
    ax[0].set_xticks(x); ax[0].set_xticklabels(gnames, rotation=20, ha="right")
    ax[0].set_ylabel("(final-random)/(learned-random)"); ax[0].set_ylim(0, 1.2)
    ax[0].legend(); ax[0].grid(axis="y", alpha=0.3)

    def g0_curve(mode):
        cs = []
        for k in range(len(agg[mode]["R"])):
            R = agg[mode]["R"][k]; rnd = agg[mode]["random"][k][0]; learned = np.diag(R)[0]
            denom = learned - rnd
            cs.append((R[:, 0] - rnd) / (denom if abs(denom) > 1e-6 else 1.0))
        cs = np.array(cs); return np.nanmean(cs, 0), np.nanstd(cs, 0)
    b0, b0s = g0_curve("baseline"); p0, p0s = g0_curve("pmac")
    ax[1].plot(x, b0, "-o", color=BLUE, label="baseline"); ax[1].fill_between(x, b0-b0s, b0+b0s, color=BLUE, alpha=0.15)
    ax[1].plot(x, p0, "-o", color=ORANGE, label="PMA-C"); ax[1].fill_between(x, p0-p0s, p0+p0s, color=ORANGE, alpha=0.15)
    ax[1].set_title(f"{gnames[0]} (game 0) normalized score across training")
    ax[1].set_xlabel("after training game #"); ax[1].set_ylabel("normalized score"); ax[1].set_ylim(-0.1, 1.2)
    ax[1].legend(); ax[1].grid(alpha=0.3)

    fig.suptitle(f"Continual full-ALE-Atari — PMA-C vs Baseline   |   "
                 f"mean norm. retention {summ['baseline']['mr'][0]:.2f}→{summ['pmac']['mr'][0]:.2f}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=120); print("\nwrote", out)


if __name__ == "__main__":
    main()
