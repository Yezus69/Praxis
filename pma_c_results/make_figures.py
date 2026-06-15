"""Produce polished PMA-C figures from committed results.json files.

Usage:
  python make_figures.py headline <headline_results.json> <out.png>
  python make_figures.py decomp   <decomp_results.json>   <out.png>
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, GREEN, GRAY = "#1f77b4", "#ff7f0e", "#2ca02c", "#7f7f7f"


def load(path):
    return json.load(open(path, encoding="utf-8"))


def first_seed_results(d):
    s = str(d["seeds"][0])
    return d["runs"][s]["results"]


def stacked_matrix(d, mode):
    """Mean accuracy matrix across seeds for a mode."""
    mats = []
    for s in d["seeds"]:
        res = d["runs"][str(s)]["results"].get(mode)
        if res:
            mats.append(np.array(res["acc_matrix"]))
    return np.mean(mats, axis=0) if mats else None


def agg(d, mode, metric):
    a = d.get("aggregate", {}).get(mode, {}).get(metric)
    return (a["mean"], a["std"]) if a else (np.nan, 0.0)


def headline(path, out):
    d = load(path)
    A_b = stacked_matrix(d, "baseline")
    A_p = stacked_matrix(d, "pmac")
    T = A_b.shape[0]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))

    # Panel 1: per-task final accuracy
    x = np.arange(T)
    ax[0].bar(x - 0.2, A_b[-1], 0.4, label="baseline", color=BLUE)
    ax[0].bar(x + 0.2, A_p[-1], 0.4, label="PMA-C", color=ORANGE)
    ax[0].set_title("Final accuracy per task (after all tasks)")
    ax[0].set_xlabel("task"); ax[0].set_ylabel("test accuracy"); ax[0].set_ylim(0, 1)
    ax[0].legend(); ax[0].grid(axis="y", alpha=0.3)

    # Panel 2: mean accuracy over SEEN tasks across the training timeline
    seen_b = [np.mean(A_b[i, : i + 1]) for i in range(T)]
    seen_p = [np.mean(A_p[i, : i + 1]) for i in range(T)]
    ax[1].plot(range(T), seen_b, "-o", color=BLUE, label="baseline")
    ax[1].plot(range(T), seen_p, "-o", color=ORANGE, label="PMA-C")
    ax[1].set_title("Mean accuracy over tasks seen so far")
    ax[1].set_xlabel("after training task #"); ax[1].set_ylabel("mean acc (seen tasks)")
    ax[1].set_ylim(0, 1); ax[1].legend(); ax[1].grid(alpha=0.3)

    # Panel 3: retention of task 0 across the timeline (the money plot)
    ax[2].plot(range(T), A_b[:, 0], "-o", color=BLUE, label="baseline")
    ax[2].plot(range(T), A_p[:, 0], "-o", color=ORANGE, label="PMA-C")
    ax[2].set_title("Task-0 accuracy as new tasks are learned")
    ax[2].set_xlabel("after training task #"); ax[2].set_ylabel("task-0 test acc")
    ax[2].set_ylim(0, 1); ax[2].legend(); ax[2].grid(alpha=0.3)

    accb, _ = agg(d, "baseline", "ACC"); accp, _ = agg(d, "pmac", "ACC")
    fb, _ = agg(d, "baseline", "forgetting"); fp, _ = agg(d, "pmac", "forgetting")
    fig.suptitle(f"PMA-C vs Baseline — {T}-task Permuted-MNIST   |   "
                 f"ACC {accb:.3f}→{accp:.3f}   Forgetting {fb:.3f}→{fp:.3f}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=120); print("wrote", out)


def decomp(path, out):
    d = load(path)
    order = ["baseline", "pmac_replay_only", "pmac_no_replay", "pmac",
             "pmac_no_projection", "pmac_no_conservation", "pmac_no_stability",
             "pmac_random_memory"]
    labels = {"baseline": "baseline", "pmac_replay_only": "replay-only (ER)",
              "pmac_no_replay": "PMA-C −replay", "pmac": "PMA-C (full)",
              "pmac_no_projection": "−projection", "pmac_no_conservation": "−conservation",
              "pmac_no_stability": "−stability", "pmac_random_memory": "random-memory"}
    modes = [m for m in order if m in d.get("aggregate", {})]
    accs = [agg(d, m, "ACC") for m in modes]
    forg = [agg(d, m, "forgetting") for m in modes]
    x = np.arange(len(modes))
    colors = [GRAY if m == "baseline" else (GREEN if m == "pmac" else BLUE) for m in modes]

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].bar(x, [a[0] for a in accs], yerr=[a[1] for a in accs], color=colors, capsize=3)
    ax[0].set_title("Average Accuracy by condition (higher = better)")
    ax[0].set_ylabel("ACC"); ax[0].set_ylim(0, 1)
    ax[0].set_xticks(x); ax[0].set_xticklabels([labels[m] for m in modes], rotation=30, ha="right")
    ax[0].grid(axis="y", alpha=0.3)

    ax[1].bar(x, [f[0] for f in forg], yerr=[f[1] for f in forg], color=colors, capsize=3)
    ax[1].set_title("Forgetting by condition (lower = better)")
    ax[1].set_ylabel("Forgetting");
    ax[1].set_xticks(x); ax[1].set_xticklabels([labels[m] for m in modes], rotation=30, ha="right")
    ax[1].grid(axis="y", alpha=0.3)

    fig.suptitle("PMA-C credit decomposition — Permuted-MNIST (mean ± std over seeds)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=120); print("wrote", out)


if __name__ == "__main__":
    which, path, out = sys.argv[1], sys.argv[2], sys.argv[3]
    (headline if which == "headline" else decomp)(path, out)
