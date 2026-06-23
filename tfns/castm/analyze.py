"""Analyze CASTM Atari ladder results and emit the metrics report (spec 20, 21).

Reads oracle/5-game ``results.json`` files and the matched single-task reference
``final.json`` files, computes normalized progress / retention / forgetting,
checks the acceptance gates, and writes a markdown + JSON report.

Run:
    python -m tfns.castm.analyze --runs castm_runs/oracle/seed1 castm_runs/oracle/five_seed1 \
        --refs castm_runs/refs --out castm_runs/REPORT.md
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load(path):
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def load_reference_scores(refs_dir) -> dict[str, dict]:
    """Return {game: {single, final, random?}} from baseline_ppo reference runs."""

    out = {}
    refs = Path(refs_dir)
    if not refs.exists():
        return out
    for game_dir in sorted(refs.iterdir()):
        final = _load(game_dir / "final.json")
        if final is None:
            continue
        best = final.get("best_eval") or {}
        last = final.get("final_eval") or {}
        out[game_dir.name] = {
            "single": float(best.get("mean_return", float("nan"))),
            "final": float(last.get("mean_return", float("nan"))),
        }
    return out


def progress(s, s_rand, s_single, eps=1e-8):
    return (s - s_rand) / (s_single - s_rand + eps)


def retention(s, s_rand, s_best, eps=1e-8):
    return (s - s_rand) / (s_best - s_rand + eps)


def analyze_run(results: dict, refs: dict[str, dict]) -> dict:
    games = results["config"]["games"]
    random_scores = results["random_scores"]
    best_after = results["best_after_learn"]
    matrix = results["retention_matrix"]
    n = len(games)

    # Score of game i after the final game (last row of the retention matrix).
    final_row = matrix[-1]["scores"] if matrix else {}
    # Per-game score history across rows (for forgetting).
    history = {g: [] for g in games}
    for row in matrix:
        for g, ev in row["scores"].items():
            history[g].append(float(ev["mean"]))

    per_game = {}
    for gi, g in enumerate(games):
        s_rand = float(random_scores.get(g, 0.0))
        s_single = float(refs.get(g, {}).get("single", best_after.get(g, float("nan"))))
        # Retention baseline (spec 20): best score observed AFTER learning game g,
        # i.e., the max over the retention-matrix rows from game g onward (the
        # certified-policy score), NOT the transient training-curve peak.
        post = [float(matrix[r]["scores"][g]["mean"]) for r in range(gi, len(matrix))
                if g in matrix[r]["scores"]]
        s_best = max(post) if post else float(best_after.get(g, float("nan")))
        s_final = float(final_row.get(g, {}).get("mean", float("nan")))
        P_final = progress(s_final, s_rand, s_single)
        R_final = retention(s_final, s_rand, s_best)
        # forgetting on normalized progress
        P_hist = [progress(s, s_rand, s_single) for s in history[g]]
        F = (max(P_hist) - P_hist[-1]) if P_hist else float("nan")
        per_game[g] = {
            "S_random": s_rand, "S_single": s_single, "S_best_after_learn": s_best,
            "S_final": s_final, "progress_final": P_final, "retention_final": R_final,
            "forgetting": F, "score_history": history[g],
        }

    current = games[-1]
    min_P = min(per_game[g]["progress_final"] for g in games)
    min_R = min(per_game[g]["retention_final"] for g in games)
    P_current = per_game[current]["progress_final"]

    out = {
        "games": list(games),
        "per_game": per_game,
        "min_progress": min_P,
        "min_retention": min_R,
        "current_progress": P_current,
    }

    # Inferred (Stage D) comparison if present.
    inferred = results.get("inferred")
    if inferred:
        out["routing_accuracy"] = inferred["routing_accuracy"]
        inf = {}
        for g in games:
            ev = inferred["scores"].get(g, {})
            s_rand = float(random_scores.get(g, 0.0))
            s_single = float(refs.get(g, {}).get("single", best_after.get(g, float("nan"))))
            inf[g] = {
                "inferred_mean": float(ev.get("mean", float("nan"))),
                "route_acc": float(ev.get("route_acc", float("nan"))),
                "inferred_progress": progress(float(ev.get("mean", float("nan"))), s_rand, s_single),
                "oracle_mean": per_game[g]["S_final"],
            }
        out["inferred"] = inf

    # Gates.
    if n == 2:
        out["gate_21_2_oracle"] = {
            "P2>=0.90": per_game[games[1]]["progress_final"] >= 0.90,
            "R1>=0.90": per_game[games[0]]["retention_final"] >= 0.90,
            "P2": per_game[games[1]]["progress_final"],
            "R1": per_game[games[0]]["retention_final"],
        }
        if inferred:
            ra = inferred["routing_accuracy"]["overall"]
            out["gate_21_3_inferred"] = {
                "P2>=0.90": out["inferred"][games[1]]["inferred_progress"] >= 0.90,
                "R1(routing)>=0.99": ra >= 0.99,
                "routing_overall": ra,
            }
    out["gate_21_4_five"] = {
        "min_P>=0.90": min_P >= 0.90,
        "min_R>=0.90 & P_cur>=0.90": (min_R >= 0.90 and P_current >= 0.90),
        "min_P": min_P, "min_R": min_R, "P_current": P_current,
    }
    return out


def _fmt(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def render_markdown(analyses: dict[str, dict], refs: dict[str, dict]) -> str:
    lines = ["# CASTM Atari Ladder Results", ""]
    lines.append("## Matched single-task references (stochastic eval)")
    lines.append("")
    lines.append("| Game | reference best | reference final |")
    lines.append("|---|---|---|")
    for g, r in refs.items():
        lines.append(f"| {g} | {_fmt(r['single'])} | {_fmt(r['final'])} |")
    lines.append("")
    for name, a in analyses.items():
        lines.append(f"## Run: `{name}`")
        lines.append("")
        lines.append("| Game | S_rand | S_single | S_best | S_final | Progress | Retention | Forgetting |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for g in a["games"]:
            pg = a["per_game"][g]
            lines.append(
                f"| {g} | {_fmt(pg['S_random'])} | {_fmt(pg['S_single'])} | {_fmt(pg['S_best_after_learn'])} "
                f"| {_fmt(pg['S_final'])} | {_fmt(pg['progress_final'])} | {_fmt(pg['retention_final'])} "
                f"| {_fmt(pg['forgetting'])} |"
            )
        lines.append("")
        lines.append(f"- min progress = {_fmt(a['min_progress'])}, min retention = {_fmt(a['min_retention'])}, "
                     f"current progress = {_fmt(a['current_progress'])}")
        if "gate_21_2_oracle" in a:
            g2 = a["gate_21_2_oracle"]
            lines.append(f"- **Gate 21.2 (oracle 2-game):** P2={_fmt(g2['P2'])} (>=0.90: {g2['P2>=0.90']}), "
                         f"R1={_fmt(g2['R1'])} (>=0.90: {g2['R1>=0.90']})")
        if "gate_21_3_inferred" in a:
            g3 = a["gate_21_3_inferred"]
            lines.append(f"- **Gate 21.3 (inferred 2-game):** routing_acc={_fmt(g3['routing_overall'])} "
                         f"(>=0.99: {g3['R1(routing)>=0.99']}), P2>=0.90: {g3['P2>=0.90']}")
        g4 = a["gate_21_4_five"]
        lines.append(f"- **Gate 21.4 (five-game):** min_P={_fmt(g4['min_P'])} (>=0.90: {g4['min_P>=0.90']}); "
                     f"or min_R={_fmt(g4['min_R'])} & P_cur={_fmt(g4['P_current'])} "
                     f"({g4['min_R>=0.90 & P_cur>=0.90']})")
        if "inferred" in a:
            lines.append("")
            lines.append("### Oracle vs inferred address (spec 24)")
            lines.append("| Game | oracle | inferred | route acc |")
            lines.append("|---|---|---|---|")
            for g in a["games"]:
                inf = a["inferred"][g]
                lines.append(f"| {g} | {_fmt(inf['oracle_mean'])} | {_fmt(inf['inferred_mean'])} "
                             f"| {_fmt(inf['route_acc'])} |")
        lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--refs", type=str, default="castm_runs/refs")
    p.add_argument("--out", type=str, default="castm_runs/REPORT.md")
    args = p.parse_args()

    refs = load_reference_scores(args.refs)
    analyses = {}
    for run_dir in args.runs:
        res = _load(Path(run_dir) / "results.json")
        if res is None:
            print(f"skip {run_dir} (no results.json)")
            continue
        analyses[run_dir] = analyze_run(res, refs)

    md = render_markdown(analyses, refs)
    Path(args.out).write_text(md, encoding="utf-8")
    Path(args.out).with_suffix(".json").write_text(
        json.dumps({"refs": refs, "analyses": analyses}, indent=2), encoding="utf-8"
    )
    print(md)
    print(f"\nreport -> {args.out}")


if __name__ == "__main__":
    main()
