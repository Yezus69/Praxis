"""Plot helpers for PMA-C continual-learning comparisons."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_comparison(results, out_path, title="PMA-C vs Baseline - Permuted-MNIST"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results.keys())
    if not names:
        raise ValueError("results must not be empty")

    first = results[names[0]]
    n_tasks = int(first.acc_matrix.shape[1])
    x = np.arange(n_tasks)
    width = 0.8 / max(len(names), 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(title)

    for i, name in enumerate(names):
        result = results[name]
        offset = (i - (len(names) - 1) / 2.0) * width
        axes[0].bar(x + offset, result.final_acc, width=width, label=name)
    axes[0].set_title("Final Accuracy")
    axes[0].set_xlabel("Task")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].legend(fontsize=8)

    lines = []
    for name in names:
        metrics = results[name].metrics
        lines.append(
            f"{name}\n"
            f"ACC {metrics.get('ACC', 0.0):.3f} | BWT {metrics.get('BWT', 0.0):.3f}\n"
            f"Forget {metrics.get('forgetting', metrics.get('Forgetting', 0.0)):.3f}\n"
            f"Ret {metrics.get('mean_retention', 0.0):.3f} / "
            f"{metrics.get('worst_retention', 0.0):.3f}"
        )
    axes[1].axis("off")
    axes[1].set_title("Summary")
    axes[1].text(
        0.02,
        0.98,
        "\n\n".join(lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "0.8"},
    )

    for name in names:
        result = results[name]
        if result.acc_matrix.shape[1] > 0:
            axes[2].plot(result.acc_matrix[:, 0], marker="o", label=name)
    axes[2].set_title("Task 0 Across Training")
    axes[2].set_xlabel("After Task")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


__all__ = ["plot_comparison"]
