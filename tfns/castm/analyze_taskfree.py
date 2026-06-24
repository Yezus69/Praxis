"""Analyse a task-free CASTM run: normalized progress, retention, gate status.

Reads a run's ``results.json`` plus matched single-task references and prints raw +
normalized scores, the retention matrix, routing metrics, and the strict-gate
verdict (R_old>=0.90, P_new>=0.90, A_router>=0.99). Oracle (diagnostic) and
inferred (primary) results are kept clearly separate.

Usage:
    python -m tfns.castm.analyze_taskfree castm_runs/taskfree/stage1/results.json \
        --refs-dir castm_runs/newset/refs [--ref-json extra_refs.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_refs(refs_dir: str | None, ref_json: str | None) -> dict[str, float]:
    refs: dict[str, float] = {}
    if refs_dir:
        for p in Path(refs_dir).glob("*/progress.jsonl"):
            game = p.parent.name
            try:
                last = [json.loads(l) for l in p.read_text().splitlines() if l.strip()][-1]
                refs[game] = float(last.get("mean_return", last.get("mean", float("nan"))))
            except (IndexError, ValueError):
                continue
    if ref_json:
        refs.update({k: float(v) for k, v in json.loads(Path(ref_json).read_text()).items()})
    return refs


def _norm(score, random, ref):
    denom = ref - random
    if abs(denom) < 1e-6:
        return float("nan"), True  # degenerate reference (ref ~ random)
    return (score - random) / denom, False


def analyze(results_path: str, refs_dir: str | None, ref_json: str | None) -> dict:
    data = json.loads(Path(results_path).read_text())
    games = list(data["config"]["games"])
    random_scores = data.get("random_scores", {})
    best = data.get("best_after_learn", {})
    refs = load_refs(refs_dir, ref_json)
    g2c = data.get("game_to_ctx_LABEL_ONLY", {})
    rmat = data.get("retention_matrix", [])
    final_row = rmat[-1] if rmat else {"oracle": {}, "inferred": {}}
    oracle_final = final_row.get("oracle", {})
    inferred = data.get("inferred") or {}
    inferred_scores = inferred.get("scores", {})
    routing = inferred.get("routing_accuracy", {})

    rows = []
    for gi, g in enumerate(games):
        rnd = random_scores.get(g, float("nan"))
        ref = refs.get(g, float("nan"))
        or_score = (oracle_final.get(g) or {}).get("mean", float("nan"))
        inf_score = (inferred_scores.get(g) or {}).get("mean", float("nan"))
        learned = best.get(g, float("nan"))
        or_prog, degen = _norm(or_score, rnd, ref)
        inf_prog, _ = _norm(inf_score, rnd, ref)
        # retention = current / learned, normalized over random
        ret_denom = learned - rnd
        oracle_ret = ((or_score - rnd) / ret_denom) if abs(ret_denom) > 1e-6 else float("nan")
        inf_ret = ((inf_score - rnd) / ret_denom) if abs(ret_denom) > 1e-6 else float("nan")
        rows.append({
            "game": g, "is_last": gi == len(games) - 1, "random": rnd, "ref": ref,
            "learned_peak": learned, "oracle_final": or_score, "inferred_final": inf_score,
            "oracle_progress": or_prog, "inferred_progress": inf_prog,
            "oracle_retention": oracle_ret, "inferred_retention": inf_ret,
            "degenerate_ref": degen, "ctx": g2c.get(g, -1),
            "route_acc": routing.get("per_game", {}).get(g, float("nan")),
        })

    # Gates: old games = all but the last; new game = the last learned.
    olds = [r for r in rows if not r["is_last"] and not r["degenerate_ref"]]
    last = rows[-1] if rows else None
    r_old_oracle = min([r["oracle_retention"] for r in olds], default=float("nan"))
    r_old_inf = min([r["inferred_retention"] for r in olds], default=float("nan"))
    p_new = last["inferred_progress"] if last else float("nan")
    a_router = routing.get("overall", float("nan"))
    n_ctx = data.get("num_contexts", -1)
    gates = {
        "R_old_inferred>=0.90": (r_old_inf >= 0.90),
        "P_new_inferred>=0.90": (p_new >= 0.90),
        "A_router>=0.99": (a_router >= 0.99),
        "no_proliferation": (n_ctx == len(games)),
    }
    return {"rows": rows, "gates": gates, "num_contexts": n_ctx, "n_games": len(games),
            "R_old_oracle": r_old_oracle, "R_old_inferred": r_old_inf,
            "P_new_inferred": p_new, "A_router": a_router, "refs_used": refs}


def _fmt(x, w=8, p=2):
    try:
        return f"{x:>{w}.{p}f}"
    except (TypeError, ValueError):
        return f"{'nan':>{w}}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--refs-dir", default="castm_runs/newset/refs")
    ap.add_argument("--ref-json", default=None)
    a = ap.parse_args()
    out = analyze(a.results, a.refs_dir, a.ref_json)
    print(f"\n=== {a.results} | {out['n_games']} games | discovered {out['num_contexts']} contexts ===")
    print(f"{'game':>16} {'rnd':>8} {'ref':>8} {'peak':>8} {'or_fin':>8} {'inf_fin':>8} "
          f"{'or_prog':>8} {'inf_prog':>8} {'or_ret':>8} {'inf_ret':>8} {'route':>6} ctx")
    for r in out["rows"]:
        tag = " *NEW" if r["is_last"] else ""
        deg = " DEGEN-REF" if r["degenerate_ref"] else ""
        print(f"{r['game']:>16} {_fmt(r['random'])} {_fmt(r['ref'])} {_fmt(r['learned_peak'])} "
              f"{_fmt(r['oracle_final'])} {_fmt(r['inferred_final'])} {_fmt(r['oracle_progress'])} "
              f"{_fmt(r['inferred_progress'])} {_fmt(r['oracle_retention'])} {_fmt(r['inferred_retention'])} "
              f"{_fmt(r['route_acc'],6)} {r['ctx']}{tag}{deg}")
    print(f"\nR_old (inferred, min over old) = {_fmt(out['R_old_inferred'])}   "
          f"R_old (oracle) = {_fmt(out['R_old_oracle'])}")
    print(f"P_new (inferred)               = {_fmt(out['P_new_inferred'])}")
    print(f"A_router (held-out top-1)      = {_fmt(out['A_router'])}")
    print(f"contexts discovered            = {out['num_contexts']} (games={out['n_games']})")
    print("\nSTRICT GATES:")
    for k, v in out["gates"].items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print()
    return out


if __name__ == "__main__":
    main()
