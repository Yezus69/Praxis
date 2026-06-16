"""Merge per-seed continual-Atari runs into a deployed-vs-current retention report.

Usage:
    python pma_c_results/make_atari_deployed_report.py OUT_DIR SEED_JSON [SEED_JSON ...]

Each SEED_JSON is a `results.json` written by pmac.experiments.continual_atari (one seed).
Produces in OUT_DIR:
    results_merged.json  -- per-seed + cross-seed aggregate, deployed and current metrics
    summary.md           -- headline tables (per-mode aggregate + per-game)
    fig_atari_deployed.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def _load(paths):
    runs = []
    games = None
    for p in paths:
        raw = json.load(open(p, encoding="utf-8"))
        runs.append(raw)
        if games is None:
            games = list(raw["games"])
        elif list(raw["games"]) != games:
            raise ValueError(f"game order mismatch in {p}: {raw['games']} != {games}")
    return runs, games


def _iter_mode_results(runs, mode):
    """Yield each seed's jsonified result dict for `mode`."""
    for raw in runs:
        for seed_str, payload in raw["runs"].items():
            res = payload["results"].get(mode)
            if res is not None:
                yield int(seed_str), res


def _stat(values):
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr)), "values": [float(v) for v in arr]}


def _collect_mode(runs, mode):
    seeds = []
    dep_mean, dep_worst = [], []
    cur_mean, cur_worst = [], []
    champ_mean = []
    mean_final = []
    n_learned = []
    per_game = {}  # game_id -> dict of lists
    for seed, res in _iter_mode_results(runs, mode):
        seeds.append(seed)
        dep = res["metrics"]["deployed"]
        dep_mean.append(dep["mean_deployed_retention"])
        dep_worst.append(dep["worst_deployed_retention"])
        cur_mean.append(dep["mean_current_retention"])
        cur_worst.append(dep["worst_current_retention"])
        champ_mean.append(dep["mean_champion_retention"])
        n_learned.append(dep["n_learned"])
        mean_final.append(res["metrics"]["mean_final_return"])
        for gid, ss in enumerate(res["extra"]["deployment"]["skill_scores"]):
            g = per_game.setdefault(
                gid,
                {"skill_id": ss["skill_id"], "deployed_ret": [], "current_ret": [],
                 "champion_ret": [], "best": [], "deployed": [], "current": [],
                 "champion": [], "random": [], "learned": [], "route": []},
            )
            g["deployed_ret"].append(ss["deployed_retention"])
            g["current_ret"].append(ss["current_retention"])
            g["champion_ret"].append(ss["champion_retention"])
            g["best"].append(ss["best_score"])
            g["deployed"].append(ss["deployed_score"])
            g["current"].append(ss["current_score"])
            g["champion"].append(ss["champion_score"])
            g["random"].append(ss["random_score"])
            g["learned"].append(bool(ss["learned"]))
            g["route"].append(ss["route_type"])
    if not seeds:
        return None
    return {
        "seeds": seeds,
        "n_seeds": len(seeds),
        "deployed_mean_retention": _stat(dep_mean),
        "deployed_worst_retention": _stat(dep_worst),
        "current_mean_retention": _stat(cur_mean),
        "current_worst_retention": _stat(cur_worst),
        "champion_mean_retention": _stat(champ_mean),
        "mean_final_return": _stat(mean_final),
        "n_learned": _stat(n_learned),
        "per_game": per_game,
    }


def _fmt(stat):
    return f"{stat['mean']:.3f} ± {stat['std']:.3f}"


def _write_summary(out_dir, games, collected):
    lines = []
    lines.append("# Continual Full-ALE-Atari: Deployed vs Current retention\n")
    lines.append(f"Games (sequential): {', '.join(games)}\n")
    lines.append("## Per-mode aggregate (over seeds, learned skills only)\n")
    lines.append("| mode | n_seeds | deployed mean ret | deployed worst ret | current mean ret | current worst ret | mean final return |")
    lines.append("|---|---|---|---|---|---|---|")
    for mode, c in collected.items():
        if c is None:
            continue
        lines.append(
            f"| {mode} | {c['n_seeds']} | {_fmt(c['deployed_mean_retention'])} | "
            f"{_fmt(c['deployed_worst_retention'])} | {_fmt(c['current_mean_retention'])} | "
            f"{_fmt(c['current_worst_retention'])} | {_fmt(c['mean_final_return'])} |"
        )
    lines.append("")
    for mode, c in collected.items():
        if c is None:
            continue
        lines.append(f"## Per-game ({mode}) — mean over seeds\n")
        lines.append("| game | learned | best | current | champion | deployed | current_ret | deployed_ret | route |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for gid in sorted(c["per_game"]):
            g = c["per_game"][gid]
            route = max(set(g["route"]), key=g["route"].count)
            learned = "yes" if all(g["learned"]) else ("some" if any(g["learned"]) else "no")
            lines.append(
                f"| {g['skill_id']} | {learned} | {np.mean(g['best']):.1f} | "
                f"{np.mean(g['current']):.1f} | {np.mean(g['champion']):.1f} | "
                f"{np.mean(g['deployed']):.1f} | {np.mean(g['current_ret']):.3f} | "
                f"{np.mean(g['deployed_ret']):.3f} | {route} |"
            )
        lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def _plot(out_dir, games, collected):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = [m for m in collected if collected[m] is not None]
    n_games = len(games)
    x = np.arange(n_games)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    fig.suptitle("Continual Full-ALE Atari — deployed PMA-C does not forget")

    # Panel 0: per-game deployed vs current retention for pmac (if present) else first mode
    focus = "pmac" if "pmac" in collected and collected["pmac"] else modes[0]
    c = collected[focus]
    dep = [np.mean(c["per_game"][g]["deployed_ret"]) for g in range(n_games)]
    cur = [np.mean(c["per_game"][g]["current_ret"]) for g in range(n_games)]
    w = 0.38
    axes[0].bar(x - w / 2, dep, width=w, label="deployed (routed)", color="#2a9d8f")
    axes[0].bar(x + w / 2, cur, width=w, label="current shared net", color="#e76f51")
    axes[0].axhline(0.98, ls="--", c="k", lw=0.8, label="0.98 target")
    axes[0].set_title(f"{focus}: per-game retention")
    axes[0].set_ylabel("retention")
    axes[0].set_ylim(0, 1.25)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([g.replace("-v5", "") for g in games], rotation=25, ha="right")
    axes[0].legend(fontsize=8)

    # Panel 1: deployed mean retention by mode (baseline vs pmac)
    bar_modes = [m for m in ["baseline", "pmac"] if m in collected and collected[m]]
    if not bar_modes:
        bar_modes = modes
    means = [collected[m]["deployed_mean_retention"]["mean"] for m in bar_modes]
    stds = [collected[m]["deployed_mean_retention"]["std"] for m in bar_modes]
    cur_means = [collected[m]["current_mean_retention"]["mean"] for m in bar_modes]
    bx = np.arange(len(bar_modes))
    axes[1].bar(bx - 0.2, means, width=0.4, yerr=stds, capsize=4, label="deployed", color="#2a9d8f")
    axes[1].bar(bx + 0.2, cur_means, width=0.4, label="current", color="#e76f51")
    axes[1].axhline(0.98, ls="--", c="k", lw=0.8)
    axes[1].set_title("mean retention by mode")
    axes[1].set_ylabel("retention")
    axes[1].set_ylim(0, 1.15)
    axes[1].set_xticks(bx)
    axes[1].set_xticklabels(bar_modes)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "fig_atari_deployed.png", dpi=160)
    plt.close(fig)


def main(argv):
    if len(argv) < 2:
        raise SystemExit(__doc__)
    out_dir = Path(argv[0])
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_paths = argv[1:]
    runs, games = _load(seed_paths)
    modes = []
    for raw in runs:
        for payload in raw["runs"].values():
            for m in payload["results"]:
                if m not in modes:
                    modes.append(m)
    collected = {m: _collect_mode(runs, m) for m in modes}
    merged = {
        "games": games,
        "seed_files": [str(p) for p in seed_paths],
        "modes": {
            m: {k: v for k, v in c.items() if k != "per_game"}
            for m, c in collected.items() if c is not None
        },
        "per_game": {
            m: {str(g): {kk: vv for kk, vv in gd.items()} for g, gd in c["per_game"].items()}
            for m, c in collected.items() if c is not None
        },
    }
    json.dump(merged, open(out_dir / "results_merged.json", "w", encoding="utf-8"), indent=2)
    _write_summary(out_dir, games, collected)
    _plot(out_dir, games, collected)
    print(f"wrote {out_dir/'results_merged.json'}")
    print(f"wrote {out_dir/'summary.md'}")
    print(f"wrote {out_dir/'fig_atari_deployed.png'}")
    for m, c in collected.items():
        if c is None:
            continue
        print(
            f"  {m}: deployed_ret={_fmt(c['deployed_mean_retention'])} "
            f"current_ret={_fmt(c['current_mean_retention'])} "
            f"final={_fmt(c['mean_final_return'])}"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
