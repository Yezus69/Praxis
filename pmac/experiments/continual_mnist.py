"""CLI for matched PMA-C continual MNIST/synthetic experiments."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from pmac.config import ExperimentConfig, PMAConfig
from pmac.continual import ContinualResult, run_baseline, run_pmac
from pmac.data.streams import build_stream
from pmac.plotting import plot_comparison


def _parse_ints(text):
    return tuple(int(part) for part in str(text).split(",") if part)


def _parse_seeds(text):
    return [int(part) for part in str(text).split(",") if part]


def _parse_ablations(text):
    values = [part.strip() for part in str(text).split(",") if part.strip()]
    return [None if value == "none" else value for value in values]


def _result_to_json(result: ContinualResult):
    return {
        "mode": result.mode,
        "source_tag": result.source_tag,
        "acc_matrix": np.asarray(result.acc_matrix).tolist(),
        "learned_acc": np.asarray(result.learned_acc).tolist(),
        "final_acc": np.asarray(result.final_acc).tolist(),
        "peak_acc": np.asarray(result.peak_acc).tolist(),
        "metrics": result.metrics,
        "extra": result.extra,
    }


def _aggregate(results_by_mode):
    aggregate = {}
    for mode, results in results_by_mode.items():
        metric_names = sorted(results[0].metrics.keys())
        metrics = {}
        for name in metric_names:
            values = [result.metrics[name] for result in results]
            if isinstance(values[0], (list, tuple)):
                continue
            arr = np.asarray(values, dtype=np.float64)
            metrics[name] = {"mean": float(np.mean(arr)), "std": float(np.std(arr))}
        aggregate[mode] = metrics
    return aggregate


def _plot_first_seed(first_seed_results, aggregate, out_path):
    plot_results = {}
    for mode, result in first_seed_results.items():
        metrics = dict(result.metrics)
        for name, stats in aggregate.get(mode, {}).items():
            metrics[name] = stats["mean"]
        plot_results[mode] = replace(result, metrics=metrics)
    plot_comparison(plot_results, out_path)


def _print_table(aggregate):
    cols = ("ACC", "BWT", "forgetting", "mean_retention", "worst_retention")
    header = "mode".ljust(24) + " ".join(col.rjust(20) for col in cols)
    print(header)
    print("-" * len(header))
    for mode, metrics in aggregate.items():
        row = mode.ljust(24)
        for col in cols:
            stat = metrics.get(col, {"mean": 0.0, "std": 0.0})
            row += f"{stat['mean']:.4f}+/-{stat['std']:.4f}".rjust(20)
        print(row)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", choices=("permuted_mnist", "split_mnist", "synthetic"), default="permuted_mnist")
    parser.add_argument("--num-tasks", type=int, default=5)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--optimizer", default="sgd")
    parser.add_argument("--hidden", default="256,256")
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--ablations", default="none")
    parser.add_argument("--out", default="runs/pmac_mnist")
    parser.add_argument("--max-eval", type=int, default=2000)
    args = parser.parse_args(argv)

    seeds = _parse_seeds(args.seeds)
    ablations = _parse_ablations(args.ablations)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_cfg = ExperimentConfig(
        hidden_sizes=_parse_ints(args.hidden),
        epochs_per_task=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        optimizer=args.optimizer,
        temperature=args.temperature,
        seed=seeds[0] if seeds else 0,
        max_eval=args.max_eval,
    )
    pma_cfg = PMAConfig()

    results_by_mode = {}
    first_seed_results = {}
    raw = {"seeds": seeds, "runs": {}, "config": {"experiment": asdict(exp_cfg), "pma": asdict(pma_cfg)}}

    for seed in seeds:
        stream_kwargs = {"seed": seed}
        if args.stream in ("permuted_mnist", "synthetic"):
            stream_kwargs["num_tasks"] = args.num_tasks
        tasks, source_tag = build_stream(args.stream, **stream_kwargs)
        seed_cfg = replace(exp_cfg, seed=seed)

        seed_results = {}
        baseline = run_baseline(tasks, seed_cfg, seed)
        seed_results[baseline.mode] = baseline
        results_by_mode.setdefault(baseline.mode, []).append(baseline)

        full = run_pmac(tasks, seed_cfg, pma_cfg, seed, ablation=None)
        seed_results[full.mode] = full
        results_by_mode.setdefault(full.mode, []).append(full)

        for ablation in ablations:
            if ablation is None:
                continue
            result = run_pmac(tasks, seed_cfg, pma_cfg, seed, ablation=ablation)
            seed_results[result.mode] = result
            results_by_mode.setdefault(result.mode, []).append(result)

        if not first_seed_results:
            first_seed_results = dict(seed_results)

        raw["runs"][str(seed)] = {
            "source_tag": source_tag,
            "results": {mode: _result_to_json(result) for mode, result in seed_results.items()},
        }

    aggregate = _aggregate(results_by_mode)
    raw["aggregate"] = aggregate
    results_path = out_dir / "results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2)

    _plot_first_seed(first_seed_results, aggregate, out_dir / "comparison.png")
    _print_table(aggregate)
    print(f"wrote {results_path}")
    print(f"wrote {out_dir / 'comparison.png'}")


if __name__ == "__main__":
    main()
