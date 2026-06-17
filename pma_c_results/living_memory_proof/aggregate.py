"""Aggregate the multi-seed ablation sweep: mean retention over LEARNED games, mean±std over seeds."""
import json, glob, os, statistics as st
from collections import defaultdict

LEARNED_MARGIN = float(os.environ.get("LEARNED_MARGIN", "20"))  # best-random must exceed this to count
files = sorted(glob.glob("/root/sweep/*.json"))
by_abl = defaultdict(list)  # ablation -> list of per-seed dicts
for p in files:
    try: d = json.load(open(p))
    except Exception: continue
    if d.get("status") != "done": continue
    name = os.path.basename(p).replace(".json", "")
    abl = name.rsplit("_s", 1)[0]
    by_abl[abl].append(d)

def per_seed_mean_learned(d):
    best, rnd, fin = d.get("best_scores", {}), d.get("random_scores", {}), d.get("final_scores", {})
    rets, learned = [], []
    for g in best:
        b, r, f = float(best[g]), float(rnd.get(g, 0)), float(fin.get(g, 0))
        if b - r > LEARNED_MARGIN:                  # only count games that genuinely learned
            ret = max(0.0, min(1.0, (f - r) / (b - r + 1e-6)))
            rets.append(ret); learned.append(g.split("-")[0])
    return (sum(rets)/len(rets) if rets else float("nan")), learned, rets

print(f"LEARNED_MARGIN={LEARNED_MARGIN} (a game counts only if best-random>{LEARNED_MARGIN})\n")
lines = ["## Multi-seed ablation sweep (5 games, 800k/game, stochastic eval, retention over LEARNED games only)\n",
         "| ablation | seeds | mean retention (over learned games) | per-seed means |",
         "|---|---|---|---|"]
summary = {}
for abl in ["full", "no_memory_read", "plain_ppo"]:
    runs = by_abl.get(abl, [])
    means = []
    for d in runs:
        m, learned, _ = per_seed_mean_learned(d)
        if m == m: means.append(m)
    if means:
        mu = sum(means)/len(means); sd = st.pstdev(means) if len(means) > 1 else 0.0
        summary[abl] = (mu, sd, len(means))
        lines.append(f"| {abl} | {len(means)} | {mu:.3f} ± {sd:.3f} | {[round(x,3) for x in means]} |")
    else:
        lines.append(f"| {abl} | 0 | (no completed runs) | |")
lines.append("")
# per-game retention averaged over seeds, per ablation
games = None
for abl in ["full", "no_memory_read", "plain_ppo"]:
    runs = by_abl.get(abl, [])
    if not runs: continue
    pg = defaultdict(list)
    for d in runs:
        best, rnd, fin = d.get("best_scores", {}), d.get("random_scores", {}), d.get("final_scores", {})
        for g in best:
            b, r, f = float(best[g]), float(rnd.get(g, 0)), float(fin.get(g, 0))
            if b - r > LEARNED_MARGIN:
                pg[g.split("-")[0]].append(max(0.0, min(1.0, (f - r)/(b - r + 1e-6))))
    lines.append(f"**{abl}** per-game retention (mean over seeds, learned games): " +
                 ", ".join(f"{g}={sum(v)/len(v):.2f}(n{len(v)})" for g, v in sorted(pg.items())))
out = "\n".join(lines)
print(out)
dst = "pma_c_results/living_memory_proof/SWEEP_RESULTS.md"
os.makedirs(os.path.dirname(dst), exist_ok=True)
open(dst, "w").write(out)
print(f"\n[written {dst}]")
if summary:
    order = sorted(summary.items(), key=lambda kv: -kv[1][0])
    print("\nORDERING (high->low mean retention):", " > ".join(f"{k}({v[0]:.3f})" for k, v in order))
