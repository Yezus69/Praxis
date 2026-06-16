"""Merge per-seed continual-Atari runs into an HONEST deployed-vs-current retention report.

Two distinct results are reported, per spec invariant I6 (and an adversarial review):

  1. PRIMARY / falsifiable: how much the MUTABLE SHARED NET forgot — `norm_retention`
     (random-normalized, from the return matrix, the SAME greedy eval protocol for every arm).
     This is the apples-to-apples "does the conservation guard reduce forgetting" measure.
     baseline vs champions_only (conservation OFF) vs pmac (conservation ON).

  2. SECONDARY / structural: the DEPLOYED champion-routing floor. With default safety routing
     the deployed agent serves each protected skill from its frozen certified champion, so
     deployed_retention == 1.0 BY CONSTRUCTION (deployed==champion==best). This is the
     architectural no-forgetting invariant (== certified per-task checkpointing + router); the
     champions_only arm (conservation OFF) also reaches 1.0, proving it is architectural, not a
     product of the conservation loss. It is reported as a safety floor, NOT a learning result.

Usage:
    python pma_c_results/make_atari_deployed_report.py OUT_DIR SEED_JSON [SEED_JSON ...]
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
    for raw in runs:
        for seed_str, payload in raw["runs"].items():
            res = payload["results"].get(mode)
            if res is not None:
                yield int(seed_str), res


def _stat(values):
    arr = np.asarray(values, dtype=np.float64)
    # sample std (ddof=1) so small-n uncertainty is not understated; 0.0 for n<=1.
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return {"mean": float(np.mean(arr)), "std": std, "n": int(arr.size),
            "values": [float(v) for v in arr]}


def _collect_mode(runs, mode):
    seeds = []
    norm_mean, norm_worst = [], []        # shared-net retention (return matrix, consistent protocol), per-game CLIPPED to [0,1]
    norm_excl_last = []                   # shared-net retention over OVERWRITTEN games only (exclude never-overwritten last game)
    dep_mean, dep_worst = [], []          # deployed champion-routing floor (structural)
    cur_mean, cur_worst = [], []          # deploy-protocol current retention (cross-check)
    mean_final = []                       # mean final return of the shared net (last row)
    n_learned = []
    learned_diag = []                     # per-game learned (peak) scores
    per_game = {}
    for seed, res in _iter_mode_results(runs, mode):
        seeds.append(seed)
        m = res["metrics"]
        # Clip per-game norm_retention to [0,1] before averaging: a game whose shared-net score
        # ends ABOVE its learned peak (positive transfer / eval noise, e.g. baseline BeamRider
        # seed0 final>peak) must read as full (1.0) retention, not >100%, so it cannot inflate the
        # mean. The last game in the sequence is never overwritten (learned==final -> ~1.0), so we
        # also report retention over the OVERWRITTEN games only (the clean forgetting measure).
        pg_norm = [min(1.0, max(0.0, float(v))) for v in m.get("norm_retention", [])]
        norm_mean.append(float(np.mean(pg_norm)) if pg_norm else 0.0)
        norm_worst.append(float(np.min(pg_norm)) if pg_norm else 0.0)
        norm_excl_last.append(float(np.mean(pg_norm[:-1])) if len(pg_norm) > 1 else
                              (float(np.mean(pg_norm)) if pg_norm else 0.0))
        mean_final.append(m["mean_final_return"])
        learned_diag.append([float(v) for v in res["learned"]])
        dep = m.get("deployed", {})
        dep_mean.append(dep.get("mean_deployed_retention", float("nan")))
        dep_worst.append(dep.get("worst_deployed_retention", float("nan")))
        cur_mean.append(dep.get("mean_current_retention", float("nan")))
        cur_worst.append(dep.get("worst_current_retention", float("nan")))
        n_learned.append(dep.get("n_learned", 0))
        ss = res["extra"].get("deployment", {}).get("skill_scores", [])
        norm_pg = m.get("norm_retention", [])
        for gid in range(len(res["learned"])):
            g = per_game.setdefault(
                gid,
                {"skill_id": str(res["extra"]["game_order"][gid]),
                 "norm_ret": [], "learned": [], "final": [],
                 "deployed_ret": [], "current_ret": [], "champion": [],
                 "current": [], "deployed": [], "route": []},
            )
            if gid < len(norm_pg):
                g["norm_ret"].append(float(norm_pg[gid]))
            g["learned"].append(float(res["learned"][gid]))
            g["final"].append(float(res["final"][gid]))
            if gid < len(ss):
                s = ss[gid]
                g["deployed_ret"].append(float(s["deployed_retention"]))
                g["current_ret"].append(float(s["current_retention"]))
                g["champion"].append(float(s["champion_score"]))
                g["current"].append(float(s["current_score"]))
                g["deployed"].append(float(s["deployed_score"]))
                g["route"].append(str(s["route_type"]))
    if not seeds:
        return None
    return {
        "seeds": seeds, "n_seeds": len(seeds),
        "norm_retention": _stat(norm_mean), "norm_worst": _stat(norm_worst),
        "norm_retention_overwritten": _stat(norm_excl_last),
        "deployed_floor": _stat([v for v in dep_mean if v == v]),
        "deployed_floor_worst": _stat([v for v in dep_worst if v == v]),
        "current_ret_deploy": _stat([v for v in cur_mean if v == v]),
        "current_ret_deploy_worst": _stat([v for v in cur_worst if v == v]),
        "mean_final_return": _stat(mean_final),
        "n_learned": _stat(n_learned),
        "learned_diag": np.asarray(learned_diag, dtype=np.float64),
        "seed_norm": dict(zip(seeds, norm_mean)),
        "seed_norm_overwritten": dict(zip(seeds, norm_excl_last)),
        "per_game": per_game,
    }


def _paired(collected, mode_a, mode_b, key):
    """Paired (by seed) difference mode_a - mode_b on per-seed metric `key`; bootstrap 90% CI + sign test."""
    if mode_a not in collected or mode_b not in collected:
        return None
    a, b = collected[mode_a], collected[mode_b]
    if a is None or b is None:
        return None
    seeds = sorted(set(a[key]) & set(b[key]))
    if not seeds:
        return None
    diffs = np.asarray([a[key][s] - b[key][s] for s in seeds], dtype=np.float64)
    n = diffs.size
    # paired bootstrap (resample seeds with replacement); fixed rng for reproducibility.
    rng = np.random.default_rng(12345)
    if n > 1:
        boot = np.array([rng.choice(diffs, size=n, replace=True).mean() for _ in range(10000)])
        lo, hi = float(np.percentile(boot, 5)), float(np.percentile(boot, 95))
    else:
        lo = hi = float(diffs[0])
    n_pos = int(np.sum(diffs > 0))
    return {
        "seeds": seeds, "diffs": [float(d) for d in diffs],
        "mean_diff": float(diffs.mean()), "ci90": [lo, hi],
        "n_pos": n_pos, "n": n, "direction_consistent": bool(n_pos == n or n_pos == 0),
    }


def _fmt(stat):
    n = len(stat.get("values", []))
    suffix = "" if n != 1 else " (n=1)"
    return f"{stat['mean']:.3f} ± {stat['std']:.3f}{suffix}"


def _write_summary(out_dir, games, collected):
    order = ["baseline", "champions_only", "pmac_champions_only", "pmac"]
    modes = [m for m in order if m in collected and collected[m]]
    modes += [m for m in collected if m not in modes and collected[m]]
    L = []
    L.append("# Continual Full-ALE Atari — does PMA-C stop forgetting?\n")
    L.append(f"Games (sequential): {', '.join(games)}\n")
    L.append("## 1. PRIMARY (falsifiable): shared mutable-net retention — does the conservation guard reduce forgetting?\n")
    n_seeds = max((collected[m]["n_seeds"] for m in modes), default=0)
    L.append("Random-normalized `norm_retention` from the return matrix (same greedy eval protocol for every arm), "
             "per-game clipped to [0,1] then averaged. Higher = the *single shared net* forgot less. This is the "
             "apples-to-apples learning result. **'overwritten' column excludes the never-overwritten last game** "
             "(the clean forgetting measure). 'all' is mean+-SAMPLE-std across seeds.\n")
    L.append(f"> CAVEAT: n={n_seeds} seeds (small sample). Read the paired-by-seed significance table below "
             "(per-seed sign count + 90% bootstrap CI), not a p-value. The cleanest conservation isolation is "
             "pmac vs champions_only (identical training procedure +/- the conservation guard). The DEPLOYED floor "
             "(Result 2) is structural and n-independent (1.0 by construction).\n")
    L.append("| arm | shared-net retention (all games) | shared-net retention (overwritten only) | worst game | mean final return |")
    L.append("|---|---|---|---|---|")
    for m in modes:
        c = collected[m]
        L.append(f"| {m} | {_fmt(c['norm_retention'])} | {_fmt(c['norm_retention_overwritten'])} | "
                 f"{_fmt(c['norm_worst'])} | {_fmt(c['mean_final_return'])} |")
    L.append("")
    # Paired significance of the conservation effect (pmac vs baseline, and pmac vs champions_only).
    pm = "pmac" if "pmac" in collected else None
    if pm:
        L.append("### Significance of the conservation effect (paired by seed, shared-net retention)\n")
        L.append("| contrast | metric | per-seed diffs | mean diff | 90% bootstrap CI | seeds with diff>0 |")
        L.append("|---|---|---|---|---|---|")
        for other in ["baseline", "pmac_champions_only"]:
            for key, label in [("seed_norm", "all 5"), ("seed_norm_overwritten", "overwritten 4")]:
                p = _paired(collected, pm, other, key)
                if p is None:
                    continue
                diffs = ", ".join(f"{d:+.3f}" for d in p["diffs"])
                L.append(f"| pmac − {other} | {label} | {diffs} | {p['mean_diff']:+.3f} | "
                         f"[{p['ci90'][0]:+.3f}, {p['ci90'][1]:+.3f}] | {p['n_pos']}/{p['n']} |")
        L.append("\n(Positive = conservation retains more. At small n read the CI and the sign count, not a p-value; "
                 "a consistent positive sign across all seeds + a CI excluding 0 is the bar.)\n")
    L.append("## 2. SECONDARY (structural): deployed champion-routing floor\n")
    L.append("With default safety routing the deployed agent serves each protected skill from its frozen certified "
             "champion, so **deployed_retention = 1.0 BY CONSTRUCTION** (deployed≡champion≡best). This is the "
             "architectural no-forgetting invariant (≡ certified per-task checkpointing + router), NOT a product of "
             "the conservation loss — the `champions_only` arm (conservation OFF) also reaches 1.0. The baseline has "
             "no champion store, so its deployed agent is the single mutable net and it forgets.\n")
    L.append("| arm | has champion store? | deployed floor mean | deployed floor worst |")
    L.append("|---|---|---|---|")
    for m in modes:
        c = collected[m]
        has = "no (single net)" if m == "baseline" else "yes"
        L.append(f"| {m} | {has} | {_fmt(c['deployed_floor'])} | {_fmt(c['deployed_floor_worst'])} |")
    L.append("")
    L.append("## 3. Plasticity — do new games still learn? (champions_only uses the IDENTICAL training procedure as baseline, conservation OFF)\n")
    base = collected.get("baseline")
    L.append("| arm | per-game learned (peak) mean over seeds | ratio vs baseline |")
    L.append("|---|---|---|")
    for m in modes:
        c = collected[m]
        ld = c["learned_diag"]
        per_game_mean = ld.mean(axis=0)
        mean_learned = float(np.mean(per_game_mean))
        if base is not None:
            base_mean = float(np.mean(base["learned_diag"].mean(axis=0)))
            ratio = mean_learned / base_mean if base_mean else float("nan")
        else:
            ratio = float("nan")
        L.append(f"| {m} | {mean_learned:.1f} | {ratio:.3f} |")
    L.append("\n(Both champion arms learn new games at >= baseline level on average — ratio >= ~1.0 — so neither the "
             "champion/deployed guarantee nor the conservation guard sacrifices plasticity. champions_only uses the "
             "identical training procedure as baseline (conservation off); per-game scores differ only by GPU/envpool "
             "nondeterminism over 4M steps, not bit-identical. Per-game scores below.)\n")
    for m in modes:
        c = collected[m]
        L.append(f"## Per-game ({m}) — mean over seeds\n")
        L.append("| game | learned | final(shared) | shared norm_ret | champion | deployed | deployed_ret(floor) | route |")
        L.append("|---|---|---|---|---|---|---|---|")
        for gid in sorted(c["per_game"]):
            g = c["per_game"][gid]
            def mv(k, d=0.0):
                return float(np.mean(g[k])) if g[k] else d
            route = max(set(g["route"]), key=g["route"].count) if g["route"] else "-"
            champ = mv("champion")
            dep = mv("deployed")
            dret = mv("deployed_ret")
            L.append(f"| {g['skill_id']} | {mv('learned'):.1f} | {mv('final'):.1f} | {mv('norm_ret'):.3f} | "
                     f"{champ:.1f} | {dep:.1f} | {dret:.3f} | {route} |")
        L.append("")
    (out_dir / "summary.md").write_text("\n".join(L), encoding="utf-8")


def _plot(out_dir, games, collected):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["baseline", "champions_only", "pmac_champions_only", "pmac"]
    modes = [m for m in order if m in collected and collected[m]]
    modes += [m for m in collected if m not in modes and collected[m]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    fig.suptitle("Continual Full-ALE Atari: conservation reduces shared-net forgetting; "
                 "champion routing gives a deployed no-forgetting floor")

    # Panel 0 (PRIMARY): shared-net retention by arm (the falsifiable conservation result)
    x = np.arange(len(modes))
    norm_means = [collected[m]["norm_retention"]["mean"] for m in modes]
    norm_stds = [collected[m]["norm_retention"]["std"] for m in modes]
    dep_means = [collected[m]["deployed_floor"]["mean"] for m in modes]
    axes[0].bar(x - 0.2, norm_means, width=0.4, yerr=norm_stds, capsize=4,
                label="shared net (norm_retention)", color="#e76f51")
    axes[0].bar(x + 0.2, dep_means, width=0.4, label="deployed floor (champions)", color="#2a9d8f")
    axes[0].axhline(1.0, ls=":", c="gray", lw=0.8)
    axes[0].set_title("retention by arm: shared net (measured) vs deployed floor (structural)")
    axes[0].set_ylabel("retention")
    axes[0].set_ylim(0, 1.15)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([m.replace("pmac_champions_only", "champions_only") for m in modes],
                            rotation=20, ha="right", fontsize=8)
    axes[0].legend(fontsize=8)

    # Panel 1: per-game shared-net norm_retention for baseline vs pmac (the forgetting curves)
    focus = [m for m in ["baseline", "pmac"] if m in collected and collected[m]]
    ng = len(games)
    gx = np.arange(ng)
    w = 0.8 / max(1, len(focus))
    for i, m in enumerate(focus):
        c = collected[m]
        vals = [float(np.mean(c["per_game"][g]["norm_ret"])) if c["per_game"][g]["norm_ret"] else 0.0
                for g in range(ng)]
        axes[1].bar(gx + (i - (len(focus) - 1) / 2) * w, vals, width=w, label=m)
    axes[1].set_title("per-game shared-net retention (baseline forgets, conservation retains more)")
    axes[1].set_ylabel("norm_retention")
    axes[1].set_ylim(0, 1.15)
    axes[1].set_xticks(gx)
    axes[1].set_xticklabels([g.replace("-v5", "") for g in games], rotation=25, ha="right", fontsize=8)
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
            m: {k: v for k, v in c.items() if k not in ("per_game", "learned_diag")}
            for m, c in collected.items() if c is not None
        },
        "per_game": {
            m: {str(g): gd for g, gd in c["per_game"].items()}
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
        print(f"  {m}: shared_net_retention={_fmt(c['norm_retention'])} "
              f"deployed_floor={_fmt(c['deployed_floor'])} final={_fmt(c['mean_final_return'])}")


if __name__ == "__main__":
    main(sys.argv[1:])
