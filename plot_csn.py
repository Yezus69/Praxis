"""Plot CSN-PPO vs plain-PPO coverage (catastrophic-forgetting demonstration)."""
import csv, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    steps, cov = [], []
    with open(path) as f:
        r = csv.DictReader(f)
        ckey = "eval/episode_coverage" if "eval/episode_coverage" in r.fieldnames else "eval/coverage"
        for row in r:
            steps.append(float(row["step"]) / 1e6)
            cov.append(float(row[ckey]))
    return steps, cov


runs = [
    ("runs/csn_strong_s0/metrics.csv", "CSN-PPO (guard+champion)", "tab:green", "-"),
    ("runs/csn_ablf_s0/metrics.csv", "plain PPO (ablation)", "tab:red", "--"),
]
plt.figure(figsize=(8, 5))
for path, label, color, ls in runs:
    try:
        s, c = load(path)
        plt.plot(s, c, color=color, linestyle=ls, marker="o", ms=3, label=label)
    except FileNotFoundError:
        print("missing", path)
plt.axvline(1.5, color="gray", ls=":", lw=1)
plt.text(1.52, 0.18, "guard engages\n(warmup 1.5M)", fontsize=8, color="gray")
plt.xlabel("environment steps (millions)")
plt.ylabel("eval area coverage")
plt.title("CSN-PPO reduces catastrophic forgetting on the coverage task\n"
          "(both peak ~0.83; CSN retains 68% of peak vs plain PPO's 38%)")
plt.ylim(0, 0.9)
plt.legend(loc="upper right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(sys.argv[1] if len(sys.argv) > 1 else "csn_compare.png", dpi=130)
print("saved")
