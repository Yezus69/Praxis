"""praxis/plot_curves.py — plot training curves from runs/<run>/metrics.csv.

Reads the eval metrics CSV written by the trainer (Agent-B, praxis/train.py) and
produces a 3-panel figure ``runs/<run>/curves.png``:

  panel 1: eval/episode_reward  vs step   (should RISE)
  panel 2: eval/success_rate    vs step   (should RISE)
  panel 3: eval/collision_rate  vs step   (should FALL)

Column mapping (the CSV columns the trainer writes — see the task spec):
  * x axis        <- "step"
  * reward curve  <- "eval/episode_reward"
  * success curve <- "eval/success_rate"
  * collision     <- "eval/collision_rate"

These success/collision rates originate from the env metrics named
``contract.METRIC_SUCCESS`` ("success") / ``contract.METRIC_COLLISION`` ("collision"),
which Brax surfaces as eval/episode_<name> and the trainer normalizes into the
*_rate columns above.

Robust by design:
  * matplotlib Agg backend (headless; set before importing pyplot).
  * Missing columns are skipped gracefully (the panel shows a "no data" note) rather
    than crashing — a short run may not have every eval/* column yet.
  * Pure stdlib CSV parsing (no pandas dependency).

Run:
    python -m praxis.plot_curves --run-name <run>
    python -m praxis.plot_curves --csv runs/<run>/metrics.csv --out runs/<run>/curves.png
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Optional, Tuple

# Headless: pick Agg BEFORE importing pyplot so this works in containers with no display.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)

from praxis.config import get_config  # noqa: E402


def _read_csv(csv_path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    """Read a metrics CSV into (fieldnames, list-of-row-dicts).

    Raises:
      FileNotFoundError: if csv_path does not exist (with a clear message).
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"metrics CSV not found: {csv_path}\n"
            "Expected the trainer (praxis/train.py) to have written "
            "runs/<run>/metrics.csv. Pass --run-name <run> or --csv <path>."
        )
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(r) for r in reader]
    return fieldnames, rows


def _column(
    rows: List[Dict[str, str]],
    x_col: str,
    y_col: str,
) -> Tuple[List[float], List[float]]:
    """Extract numeric (x, y) pairs for a column, skipping rows with bad/blank values."""
    xs: List[float] = []
    ys: List[float] = []
    for r in rows:
        x_raw = r.get(x_col, "")
        y_raw = r.get(y_col, "")
        if x_raw is None or y_raw is None or x_raw == "" or y_raw == "":
            continue
        try:
            xs.append(float(x_raw))
            ys.append(float(y_raw))
        except (TypeError, ValueError):
            continue
    return xs, ys


def _plot_panel(
    ax: "plt.Axes",
    rows: List[Dict[str, str]],
    fieldnames: List[str],
    x_col: str,
    y_col: str,
    title: str,
    ylabel: str,
    color: str,
    expect: str,
) -> None:
    """Plot one panel; degrade gracefully when the column is absent/empty."""
    ax.set_title(f"{title}\n({expect})", fontsize=10)
    ax.set_xlabel(x_col)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    if y_col not in fieldnames:
        ax.text(
            0.5, 0.5, f"column '{y_col}'\nnot in CSV",
            ha="center", va="center", transform=ax.transAxes, fontsize=9, color="gray",
        )
        return

    xs, ys = _column(rows, x_col, y_col)
    if not xs:
        ax.text(
            0.5, 0.5, "no numeric data",
            ha="center", va="center", transform=ax.transAxes, fontsize=9, color="gray",
        )
        return

    ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.5, color=color)


def plot_curves(csv_path: str, out_path: str, cfg=None) -> str:
    """Render the 3-panel curves figure from a metrics CSV. Returns the output path."""
    if cfg is None:
        cfg = get_config()
    plot_cfg = cfg.plot

    fieldnames, rows = _read_csv(csv_path)

    fig, axes = plt.subplots(
        1, 3,
        figsize=(float(plot_cfg.fig_width), float(plot_cfg.fig_height)),
    )

    _plot_panel(
        axes[0], rows, fieldnames,
        x_col=plot_cfg.x_column, y_col=plot_cfg.reward_column,
        title="Episode reward", ylabel="eval/episode_reward",
        color="#1f77b4", expect="should rise",
    )
    _plot_panel(
        axes[1], rows, fieldnames,
        x_col=plot_cfg.x_column, y_col=plot_cfg.success_column,
        title="Success rate", ylabel="eval/success_rate",
        color="#2ca02c", expect="should rise",
    )
    _plot_panel(
        axes[2], rows, fieldnames,
        x_col=plot_cfg.x_column, y_col=plot_cfg.collision_column,
        title="Collision rate", ylabel="eval/collision_rate",
        color="#d62728", expect="should fall",
    )

    fig.suptitle(f"Praxis training curves — {os.path.basename(os.path.dirname(csv_path)) or csv_path}")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=int(plot_cfg.dpi))
    plt.close(fig)
    return out_path


def _resolve_paths(args, cfg) -> Tuple[str, str]:
    """Resolve (csv_path, out_path) from --csv / --run-name + --out."""
    runs_dir = cfg.paths.runs_dir
    if args.csv:
        csv_path = args.csv
        default_out = os.path.join(os.path.dirname(os.path.abspath(csv_path)), cfg.paths.curves_out)
    elif args.run_name:
        csv_path = os.path.join(runs_dir, args.run_name, "metrics.csv")
        default_out = os.path.join(runs_dir, args.run_name, cfg.paths.curves_out)
    else:
        raise SystemExit("error: one of --run-name or --csv is required")

    out_path = args.out if args.out else default_out
    return csv_path, out_path


def _build_arg_parser(cfg) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m praxis.plot_curves",
        description="Plot reward / success / collision curves from a metrics CSV.",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-name", default=None, help="Run under runs/<run>/metrics.csv.")
    src.add_argument("--csv", default=None, help="Explicit path to a metrics.csv.")
    p.add_argument(
        "--out",
        default=None,
        help="Output PNG (default: runs/<run>/curves.png or alongside --csv).",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point. Importable with no side effects until called."""
    cfg = get_config()
    args = _build_arg_parser(cfg).parse_args(argv)
    csv_path, out_path = _resolve_paths(args, cfg)
    written = plot_curves(csv_path, out_path, cfg=cfg)
    print(f"[plot] wrote {os.path.abspath(written)} (from {csv_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
