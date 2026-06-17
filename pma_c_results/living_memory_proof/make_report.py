"""Assemble Living Memory PMA-C proof JSONs into a markdown report."""
import json, os, sys, glob
SRCS = sys.argv[1:] or sorted(glob.glob("/root/proof_*.json")) + sorted(glob.glob("/root/deep_*.json"))
rows = []
for p in SRCS:
    try: d = json.load(open(p))
    except Exception as e: print(f"skip {p}: {e}"); continue
    if d.get("status") != "done" and "retention" not in d: continue
    rows.append((os.path.basename(p).replace(".json",""), d))

def fmt(d, k):
    v = d.get(k, {})
    return ", ".join(f"{g.split('-')[0]}={round(float(s),1)}" for g, s in v.items())

lines = ["# Living Memory PMA-C — Proof Report\n",
         "Deployment invariant (spec §29/§33): after sequential training, old games are played by the",
         "**live model + bounded compressed memory** (no per-game full checkpoint). Memory = bounded hot bank",
         "(4096 atoms; latent 128-d keys + 18-d teacher policies, ~1.2 MB total) — NOT per-game nets.\n",
         "| run (ablation) | games | retention (per game) | mean | worst | gate_rej | reviews | consol acc/rej |",
         "|---|---|---|---|---|---|---|---|"]
for name, d in rows:
    games = "→".join(g.split("-")[0] for g in d.get("games", []))
    ret = ", ".join(f"{g.split('-')[0]}={round(float(s),3)}" for g, s in d.get("retention", {}).items())
    lines.append(f"| {name} | {games} | {ret} | {round(d.get('mean_retention',0),3)} | "
                 f"{round(d.get('worst_retention',0),3)} | {d.get('gate_rejections','-')} | "
                 f"{d.get('review_counts','-')} | {d.get('consolidation_accepts','-')}/{d.get('consolidation_rejections','-')} |")
lines.append("\n## Per-run scores (best = right after training that game; final = after all games)\n")
for name, d in rows:
    lines.append(f"### {name}")
    lines.append(f"- best:  {fmt(d,'best_scores')}")
    lines.append(f"- final: {fmt(d,'final_scores')}")
    lines.append(f"- random: {fmt(d,'random_scores')}")
    lines.append(f"- return_matrix: {d.get('return_matrix')}\n")
out = "\n".join(lines)
dst = "pma_c_results/living_memory_proof/REPORT.md"
os.makedirs(os.path.dirname(dst), exist_ok=True)
open(dst, "w").write(out)
print(out)
print(f"\n[written to {dst}]")
