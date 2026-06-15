"""Compare CSN-PPO against a plain-PPO ablation from run CSVs.

This is a standalone P10 plotting harness. It reads the trainer CSVs under
``runs/<run>/`` and renders anti-forgetting evidence:

  * eval coverage vs env steps
  * retention R(t) = coverage(t) / max(coverage[0:t])
  * eval collision rate vs env steps
  * CSN memory/kl_p95 vs env steps

Run:
    python -m praxis.plot_comparison \
        --csn-run <name> --ppo-run <name> --out runs/csn_vs_ppo.png
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)
import numpy as np  # noqa: E402


STEP_COLUMNS = ("env_steps", "step", "num_steps")
COVERAGE_COLUMNS = ("eval/episode_coverage", "eval/coverage")
COLLISION_COLUMNS = ("eval/episode_collision", "eval/collision_rate")
KL_P95_COLUMNS = ("memory/kl_p95",)
FINAL_WINDOW = 5


@dataclass
class CsvData:
    path: str
    fieldnames: List[str]
    rows: List[Dict[str, str]]
    note: str = ""


@dataclass
class Series:
    x: np.ndarray
    y: np.ndarray
    x_col: str = ""
    y_col: str = ""
    source: str = ""
    note: str = ""


@dataclass
class Summary:
    peak: float
    final: float
    retention: float


@dataclass
class RunData:
    run_arg: str
    label: str
    run_dir: str
    eval_csv: CsvData
    train_csv: Optional[CsvData]
    coverage: Series
    collision: Series
    kl_p95: Optional[Series]
    retention_curve: Series
    summary: Summary


def _read_csv(csv_path: str) -> CsvData:
    """Read a metrics CSV into fieldnames and row dictionaries.

    Missing files return an empty CsvData with a note, so plotting can degrade
    gracefully for optional inputs such as CSN train metrics.
    """
    if not os.path.isfile(csv_path):
        return CsvData(csv_path, [], [], note=f"missing file: {csv_path}")
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return CsvData(csv_path, fieldnames, rows)


def _to_float(value: object) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _first_numeric_column(
    csv_data: CsvData,
    candidates: Sequence[str],
) -> Optional[str]:
    field_set = set(csv_data.fieldnames)
    for col in candidates:
        if col not in field_set:
            continue
        if any(_to_float(row.get(col)) is not None for row in csv_data.rows):
            return col
    return None


def _extract_series(
    csv_data: CsvData,
    x_candidates: Sequence[str],
    y_candidates: Sequence[str],
) -> Series:
    if csv_data.note:
        return Series(np.array([]), np.array([]), source=csv_data.path, note=csv_data.note)
    if not csv_data.rows:
        return Series(
            np.array([]),
            np.array([]),
            source=csv_data.path,
            note=f"no rows in {csv_data.path}",
        )

    y_col = _first_numeric_column(csv_data, y_candidates)
    if y_col is None:
        tried = ", ".join(y_candidates)
        return Series(
            np.array([]),
            np.array([]),
            source=csv_data.path,
            note=f"missing numeric column; tried: {tried}",
        )

    for x_col in x_candidates:
        if x_col not in csv_data.fieldnames:
            continue
        xs: List[float] = []
        ys: List[float] = []
        for row in csv_data.rows:
            x_val = _to_float(row.get(x_col))
            y_val = _to_float(row.get(y_col))
            if x_val is None or y_val is None:
                continue
            xs.append(x_val)
            ys.append(y_val)
        if xs:
            x_arr = np.asarray(xs, dtype=float)
            y_arr = np.asarray(ys, dtype=float)
            order = np.argsort(x_arr)
            return Series(
                x_arr[order],
                y_arr[order],
                x_col=x_col,
                y_col=y_col,
                source=csv_data.path,
            )

    tried_x = ", ".join(x_candidates)
    return Series(
        np.array([]),
        np.array([]),
        y_col=y_col,
        source=csv_data.path,
        note=f"missing numeric step column; tried: {tried_x}",
    )


def _extract_from_sources(
    sources: Iterable[CsvData],
    x_candidates: Sequence[str],
    y_candidates: Sequence[str],
) -> Series:
    notes: List[str] = []
    for source in sources:
        series = _extract_series(source, x_candidates, y_candidates)
        if series.x.size:
            return series
        if series.note:
            notes.append(os.path.basename(source.path) + ": " + series.note)
    return Series(
        np.array([]),
        np.array([]),
        note="; ".join(notes) if notes else "no sources",
    )


def _summary_stats(coverage: np.ndarray) -> Summary:
    if coverage.size == 0:
        nan = float("nan")
        return Summary(nan, nan, nan)
    peak = float(np.max(coverage))
    window = min(FINAL_WINDOW, int(coverage.size))
    final = float(np.mean(coverage[-window:]))
    retention = final / peak if peak > 0.0 else float("nan")
    return Summary(peak, final, retention)


def _retention_curve(coverage: Series) -> Series:
    if coverage.y.size == 0:
        return Series(
            np.array([]),
            np.array([]),
            x_col=coverage.x_col,
            source=coverage.source,
            note=coverage.note,
        )
    running_peak = np.maximum.accumulate(coverage.y)
    retention = np.full_like(coverage.y, np.nan, dtype=float)
    valid = running_peak > 0.0
    retention[valid] = coverage.y[valid] / running_peak[valid]
    finite = np.isfinite(retention)
    return Series(
        coverage.x[finite],
        retention[finite],
        x_col=coverage.x_col,
        y_col="retention",
        source=coverage.source,
        note="" if np.any(finite) else "retention undefined because coverage never exceeded 0",
    )


def _resolve_run_dir(run_arg: str) -> str:
    if os.path.isdir(run_arg):
        return run_arg
    return os.path.join("runs", run_arg)


def _load_run(run_arg: str, label: str, include_csn_train: bool) -> RunData:
    run_dir = _resolve_run_dir(run_arg)
    eval_csv = _read_csv(os.path.join(run_dir, "metrics.csv"))
    train_csv = _read_csv(os.path.join(run_dir, "train_metrics.csv")) if include_csn_train else None

    coverage = _extract_series(eval_csv, STEP_COLUMNS, COVERAGE_COLUMNS)
    collision = _extract_series(eval_csv, STEP_COLUMNS, COLLISION_COLUMNS)
    kl_p95 = None
    if include_csn_train:
        assert train_csv is not None
        kl_p95 = _extract_from_sources((train_csv, eval_csv), STEP_COLUMNS, KL_P95_COLUMNS)

    retention = _retention_curve(coverage)
    summary = _summary_stats(coverage.y)
    return RunData(
        run_arg=run_arg,
        label=label,
        run_dir=run_dir,
        eval_csv=eval_csv,
        train_csv=train_csv,
        coverage=coverage,
        collision=collision,
        kl_p95=kl_p95,
        retention_curve=retention,
        summary=summary,
    )


def _format_float(value: float) -> str:
    return "nan" if not math.isfinite(value) else f"{value:.4f}"


def _plot_run_series(
    ax: "plt.Axes",
    run: RunData,
    series: Series,
    color: str,
    notes: List[str],
) -> None:
    if series.x.size:
        ax.plot(series.x, series.y, linewidth=2.0, color=color, label=run.label)
        return
    note = series.note or "no numeric data"
    notes.append(f"{run.label}: {note}")


def _write_notes(ax: "plt.Axes", notes: Sequence[str]) -> None:
    if not notes:
        return
    text = "\n".join(notes[:4])
    if len(notes) > 4:
        text += "\n..."
    ax.text(
        0.02,
        0.03,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color="0.35",
        bbox={"facecolor": "white", "edgecolor": "0.85", "alpha": 0.8},
    )


def _finish_panel(
    ax: "plt.Axes",
    title: str,
    ylabel: str,
    notes: Sequence[str],
    show_legend: bool = True,
) -> None:
    ax.set_title(title)
    ax.set_xlabel("env steps")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if show_legend and ax.lines:
        ax.legend()
    if not ax.lines and notes:
        ax.text(
            0.5,
            0.5,
            "\n".join(notes[:4]),
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
            color="0.35",
        )
    else:
        _write_notes(ax, notes)


def plot_comparison(csn: RunData, ppo: RunData, out_path: str) -> str:
    """Render the four-panel CSN-vs-PPO comparison figure."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    ax_cov, ax_ret, ax_col, ax_kl = axes.ravel()
    csn_color = "#1f77b4"
    ppo_color = "#d62728"

    notes: List[str] = []
    _plot_run_series(ax_cov, csn, csn.coverage, csn_color, notes)
    _plot_run_series(ax_cov, ppo, ppo.coverage, ppo_color, notes)
    _finish_panel(ax_cov, "(a) Eval coverage", "coverage", notes)

    notes = []
    _plot_run_series(ax_ret, csn, csn.retention_curve, csn_color, notes)
    _plot_run_series(ax_ret, ppo, ppo.retention_curve, ppo_color, notes)
    ax_ret.set_ylim(0.0, 1.05)
    _finish_panel(ax_ret, "(b) Retention R(t)", "coverage / running peak", notes)

    notes = []
    _plot_run_series(ax_col, csn, csn.collision, csn_color, notes)
    _plot_run_series(ax_col, ppo, ppo.collision, ppo_color, notes)
    _finish_panel(ax_col, "(c) Eval collision rate", "collision rate", notes)

    notes = []
    if csn.kl_p95 is not None and csn.kl_p95.x.size:
        ax_kl.plot(csn.kl_p95.x, csn.kl_p95.y, linewidth=2.0, color=csn_color, label=csn.label)
    else:
        note = "CSN: no memory/kl_p95 data"
        if csn.kl_p95 is not None and csn.kl_p95.note:
            note = f"CSN: {csn.kl_p95.note}"
        notes.append(note)
    _finish_panel(ax_kl, "(d) CSN memory/kl_p95", "memory/kl_p95", notes)

    fig.suptitle("CSN-PPO vs Plain PPO anti-forgetting comparison", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return out_path


def _print_summary(csn: RunData, ppo: RunData) -> None:
    print("run, peak P, final F, retention R")
    for run in (csn, ppo):
        stats = run.summary
        print(
            f"{run.label}, "
            f"{_format_float(stats.peak)}, "
            f"{_format_float(stats.final)}, "
            f"{_format_float(stats.retention)}"
        )
    gap = csn.summary.retention - ppo.summary.retention
    print(f"absolute retention gap (CSN - PPO): {_format_float(gap)}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m praxis.plot_comparison",
        description="Plot CSN-PPO vs plain-PPO anti-forgetting comparison curves.",
    )
    parser.add_argument("--csn-run", required=True, help="CSN run name under runs/ or a run directory.")
    parser.add_argument("--ppo-run", required=True, help="Plain-PPO run name under runs/ or a run directory.")
    parser.add_argument("--out", required=True, help="Output PNG path.")
    parser.add_argument("--label-csn", default="CSN-PPO", help="Legend label for the CSN run.")
    parser.add_argument("--label-ppo", default="Plain PPO", help="Legend label for the PPO run.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    csn = _load_run(args.csn_run, args.label_csn, include_csn_train=True)
    ppo = _load_run(args.ppo_run, args.label_ppo, include_csn_train=False)
    written = plot_comparison(csn, ppo, args.out)
    _print_summary(csn, ppo)
    print(f"[plot] wrote {os.path.abspath(written)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
