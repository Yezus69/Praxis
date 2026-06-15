"""CSN-PPO Phase 1b training entrypoint for the Praxis coverage task."""

from __future__ import annotations

import argparse
import csv
import datetime
import os
import sys
import time
from dataclasses import replace
from typing import Any, Dict, Optional, Sequence

from agent.csn_ppo.config import (
    CSNPPOConfig,
    resolve_long_run_config,
    validate_long_run_safety,
)
from praxis import contract
from praxis.train import build_env, reward_overrides_from_args


_ARG_DEST_TO_CONFIG_FIELD = {
    "enable_projection": "enable_gradient_projection",
}


def _explicit_config_fields(
    parser: argparse.ArgumentParser,
    argv: Sequence[str],
) -> set[str]:
    option_to_field = {}
    for action in parser._actions:
        field_name = _ARG_DEST_TO_CONFIG_FIELD.get(action.dest, action.dest)
        for option in action.option_strings:
            option_to_field[option] = field_name

    explicit = set()
    for token in argv:
        if token == "--":
            break
        if not token.startswith("-"):
            continue
        field_name = option_to_field.get(token.split("=", 1)[0])
        if field_name is not None:
            explicit.add(field_name)
    return explicit


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(raw_argv)
    args._explicit_config_fields = _explicit_config_fields(parser, raw_argv)
    return args


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="praxis.train_csn",
        description="CSN-PPO trainer for the Praxis coverage task.",
    )
    p.add_argument("--long-run", "--safe-long-run", dest="long_run",
                   action="store_true", default=False)
    p.add_argument("--num-timesteps", type=float, default=float(CSNPPOConfig.num_timesteps))
    p.add_argument("--num-envs", type=int, default=CSNPPOConfig.num_envs)
    p.add_argument("--num-evals", type=int, default=CSNPPOConfig.num_evals)
    p.add_argument("--deterministic-eval", dest="eval_deterministic",
                   action=argparse.BooleanOptionalAction,
                   default=CSNPPOConfig.eval_deterministic)
    p.add_argument("--seed", type=int, default=CSNPPOConfig.seed)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--episode-length", type=int, default=contract.EPISODE_LENGTH)
    p.add_argument("--unroll-length", type=int, default=CSNPPOConfig.unroll_length)
    p.add_argument("--batch-size", type=int, default=CSNPPOConfig.batch_size)
    p.add_argument("--num-minibatches", type=int, default=CSNPPOConfig.num_minibatches)
    p.add_argument("--learning-rate", type=float, default=CSNPPOConfig.learning_rate)
    p.add_argument("--entropy-cost", type=float, default=CSNPPOConfig.entropy_cost)
    p.add_argument("--discounting", type=float, default=CSNPPOConfig.discounting)
    p.add_argument("--reward-scaling", type=float, default=CSNPPOConfig.reward_scaling)
    p.add_argument("--max-updates-per-batch", type=int, default=CSNPPOConfig.max_updates_per_batch)
    p.add_argument("--holdout-early-stop", dest="enable_holdout_early_stop",
                   action=argparse.BooleanOptionalAction,
                   default=CSNPPOConfig.enable_holdout_early_stop)
    p.add_argument("--memory-size-fast", type=int, default=CSNPPOConfig.memory_size_fast)
    p.add_argument("--memory-size-slow", type=int, default=CSNPPOConfig.memory_size_slow)
    p.add_argument("--memory-batch-size", type=int, default=CSNPPOConfig.memory_batch_size)
    p.add_argument("--guard", "--enable-guard", dest="enable_guard",
                   action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--guard-warmup-steps", type=int, default=CSNPPOConfig.guard_warmup_steps)
    p.add_argument("--guard-kl-budget", type=float, default=CSNPPOConfig.guard_kl_budget)
    p.add_argument("--guard-lambda-mem", type=float, default=CSNPPOConfig.guard_lambda_mem)
    p.add_argument("--guard-lambda-base", type=float, default=CSNPPOConfig.guard_lambda_base)
    p.add_argument("--guard-policy-coef", type=float, default=CSNPPOConfig.guard_policy_coef)
    p.add_argument("--projection", "--enable-projection", dest="enable_projection",
                   action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--enable-sentinel", dest="enable_sentinel",
                   action=argparse.BooleanOptionalAction,
                   default=CSNPPOConfig.enable_sentinel)
    p.add_argument("--sentinel-bank-size", type=int, default=CSNPPOConfig.sentinel_bank_size)
    p.add_argument("--sentinel-eval-interval", type=int,
                   default=CSNPPOConfig.sentinel_eval_interval)
    p.add_argument("--validation-eval-interval", type=int,
                   default=CSNPPOConfig.validation_eval_interval)
    p.add_argument("--synthetic-probe-batch-size", type=int,
                   default=CSNPPOConfig.synthetic_probe_batch_size)
    p.add_argument("--allow-no-sentinel-for-debug", action="store_true",
                   default=CSNPPOConfig.allow_no_sentinel_for_debug)
    p.add_argument("--smoke", action="store_true")

    p.add_argument("--k-cov", type=float, default=None)
    p.add_argument("--k-coll", type=float, default=None)
    p.add_argument("--k-time", type=float, default=None)
    p.add_argument("--k-complete", type=float, default=None)
    p.add_argument("--k-fresh", type=float, default=None)
    p.add_argument("--freshness-decay", type=float, default=None)
    p.add_argument("--collision-penalty-cap", type=float, default=None)
    p.add_argument("--terminate-on-full-coverage", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--patrol", action=argparse.BooleanOptionalAction, default=None)
    return p


def resolve_config(
    argv: argparse.Namespace | Sequence[str] | None = None,
) -> CSNPPOConfig:
    if isinstance(argv, argparse.Namespace):
        args = argv
        explicit_fields = set(getattr(args, "_explicit_config_fields", set()))
    else:
        args = parse_args(argv)
        explicit_fields = set(args._explicit_config_fields)

    values = dict(
        num_timesteps=int(args.num_timesteps),
        num_envs=int(args.num_envs),
        num_evals=int(args.num_evals),
        eval_deterministic=bool(args.eval_deterministic),
        seed=int(args.seed),
        episode_length=int(args.episode_length),
        unroll_length=int(args.unroll_length),
        batch_size=int(args.batch_size),
        num_minibatches=int(args.num_minibatches),
        learning_rate=float(args.learning_rate),
        entropy_cost=float(args.entropy_cost),
        discounting=float(args.discounting),
        reward_scaling=float(args.reward_scaling),
        max_updates_per_batch=int(args.max_updates_per_batch),
        enable_holdout_early_stop=bool(args.enable_holdout_early_stop),
        memory_size_fast=int(args.memory_size_fast),
        memory_size_slow=int(args.memory_size_slow),
        memory_batch_size=int(args.memory_batch_size),
        enable_guard=bool(args.enable_guard),
        guard_warmup_steps=int(args.guard_warmup_steps),
        guard_kl_budget=float(args.guard_kl_budget),
        guard_lambda_mem=float(args.guard_lambda_mem),
        guard_lambda_base=float(args.guard_lambda_base),
        guard_policy_coef=float(args.guard_policy_coef),
        enable_gradient_projection=bool(args.enable_projection),
        enable_sentinel=bool(args.enable_sentinel),
        sentinel_bank_size=int(args.sentinel_bank_size),
        sentinel_eval_interval=int(args.sentinel_eval_interval),
        validation_eval_interval=int(args.validation_eval_interval),
        synthetic_probe_batch_size=int(args.synthetic_probe_batch_size),
        allow_no_sentinel_for_debug=bool(args.allow_no_sentinel_for_debug),
    )
    cfg = CSNPPOConfig(**values)
    if args.long_run:
        cfg = resolve_long_run_config(cfg, explicit_overrides=explicit_fields)
    if args.smoke:
        cfg = replace(
            cfg,
            num_timesteps=300_000,
            num_envs=256,
            num_evals=10,
            batch_size=128,
            num_minibatches=40,
            memory_size_fast=32_768,
            memory_size_slow=8192,
            memory_batch_size=1024,
            synthetic_probe_batch_size=256,
            min_memory_size_before_guard=2048,
        )
    return cfg


def _coerce_scalar(value: Any) -> Any:
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def make_csn_progress_fn(run_dir: str, start_time: float):
    os.makedirs(run_dir, exist_ok=True)
    train_csv_path = os.path.join(run_dir, "train_metrics.csv")
    eval_csv_path = os.path.join(run_dir, "metrics.csv")
    state = {"train_cols": [], "train_rows": [], "eval_cols": [], "eval_rows": []}

    def _write_rectangular(path, rows, cols):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({c: row.get(c, "") for c in cols})

    def progress_fn(num_steps: int, metrics: Dict[str, Any]) -> None:
        wall_s = time.time() - start_time
        row = {"step": int(num_steps), "wall_s": round(wall_s, 2)}
        row.update({k: _coerce_scalar(v) for k, v in metrics.items()})

        train_cols = sorted(set(state["train_cols"]).union(row.keys()))
        state["train_cols"] = train_cols
        state["train_rows"].append(row)
        _write_rectangular(train_csv_path, state["train_rows"], train_cols)

        has_eval = any(k.startswith("eval/") for k in row)
        if has_eval:
            eval_row = {
                "step": row["step"],
                "wall_s": row["wall_s"],
                "eval/episode_reward": row.get("eval/episode_reward", ""),
                "eval/episode_coverage": row.get("eval/episode_coverage", ""),
                "eval/episode_collision": row.get("eval/episode_collision", ""),
            }
            eval_cols = sorted(set(state["eval_cols"]).union(eval_row.keys()))
            state["eval_cols"] = eval_cols
            state["eval_rows"].append(eval_row)
            _write_rectangular(eval_csv_path, state["eval_rows"], eval_cols)

        coverage = row.get("eval/episode_coverage", "")
        guard = row.get("memory/guard_active", "")
        kl_p95 = row.get("memory/kl_p95", "")
        holdout = row.get("ppo/holdout_surrogate", "")
        print(
            f"[csn] step={int(num_steps):>10d} "
            f"coverage={coverage} guard={guard} kl_p95={kl_p95} holdout={holdout}"
        )

    return progress_fn


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    config = resolve_config(args)
    validate_long_run_safety(config)
    reward_overrides = reward_overrides_from_args(args)

    run_name = args.run_name or (
        "csn-" + datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    )
    run_dir = os.path.join("runs", run_name)
    start_time = time.time()
    progress_fn = make_csn_progress_fn(run_dir, start_time)

    print("=" * 78)
    print(f"Praxis CSN-PPO Phase 1b  run={run_name}")
    print(f"  config: {config}")
    if reward_overrides:
        print(f"  reward/env overrides: {reward_overrides}")
    print(f"  metrics -> {os.path.abspath(run_dir)}")
    print("=" * 78)

    from agent.csn_ppo import train as csn_train
    from agent.csn_ppo.env_wrappers import CurriculumBraxTrainingWrapper
    from mujoco_playground import wrapper as mjxp_wrapper

    raw_env = build_env(
        episode_length=config.episode_length,
        reward_overrides=reward_overrides,
    )
    raw_eval_env = build_env(
        episode_length=config.episode_length,
        reward_overrides=reward_overrides,
    )
    env = CurriculumBraxTrainingWrapper(raw_env)
    eval_env = mjxp_wrapper.wrap_for_brax_training(
        raw_eval_env,
        episode_length=config.episode_length,
        action_repeat=1,
    )

    csn_train.train(
        environment=env,
        config=config,
        progress_fn=progress_fn,
        eval_env=eval_env,
    )

    print("=" * 78)
    print(f"[done] trained {config.num_timesteps} requested env steps")
    print(f"[done] metrics dir: {os.path.abspath(run_dir)}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
