"""Smoke-train one envpool Atari game with bounded PPO."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from pmac.agents.atari_eval import evaluate_atari
from pmac.agents.ppo_atari import AtariPPOConfig, train_ppo_atari
from pmac.envs.atari_envpool import ATARI_GAMES


def _downsample(curve, max_points: int = 20):
    curve = list(curve)
    if len(curve) <= int(max_points):
        return [{"update": int(i), "return": float(v)} for i, v in enumerate(curve)]
    idx = np.linspace(0, len(curve) - 1, int(max_points), dtype=np.int32)
    return [{"update": int(i), "return": float(curve[int(i)])} for i in idx]


def _plot_curve(curve, out_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(curve), dtype=np.int32)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, np.asarray(curve, dtype=np.float32), linewidth=2)
    ax.set_title("Atari PPO return")
    ax.set_xlabel("PPO update")
    ax.set_ylabel("completed-episode return")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _is_learnable(curve, final_return: float) -> bool:
    if not curve:
        return False
    initial = float(curve[0])
    improvement = float(final_return) - initial
    return bool(improvement >= max(5.0, 0.25 * abs(initial)) or float(final_return) >= -5.0)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="Pong-v5", choices=ATARI_GAMES)
    parser.add_argument("--total-steps", type=int, default=5_000_000)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="pma_c_results/atari_smoke")
    args = parser.parse_args(argv)

    game_id = ATARI_GAMES.index(args.game)
    cfg = AtariPPOConfig(total_timesteps=int(args.total_steps), num_envs=int(args.num_envs))
    result = train_ppo_atari(
        game=args.game,
        game_id=game_id,
        n_games=len(ATARI_GAMES),
        cfg=cfg,
        seed=int(args.seed),
    )
    eval_score = evaluate_atari(
        result["params"],
        game=args.game,
        game_id=game_id,
        n_games=len(ATARI_GAMES),
        n_episodes=20,
        seed=int(args.seed) + 10_000,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    curve = [float(v) for v in result["returns_curve"]]
    final_return = float(result["final_return"])
    downsampled = _downsample(curve)
    learnable = _is_learnable(curve, final_return)

    payload = {
        "game": args.game,
        "seed": int(args.seed),
        "timesteps": int(result["timesteps"]),
        "config": asdict(cfg),
        "returns_curve": curve,
        "returns_curve_downsampled": downsampled,
        "final_return": final_return,
        "greedy_eval_score": float(eval_score),
        "learnable": bool(learnable),
    }
    returns_path = out_dir / "returns.json"
    with returns_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    plot_path = out_dir / "returns.png"
    _plot_curve(curve, plot_path)

    print("returns_curve_downsampled=" + json.dumps(downsampled))
    print(f"final_return={final_return:.6f}")
    print(f"greedy_eval_score={float(eval_score):.6f}")
    print("LEARNABLE" if learnable else "NOT_LEARNABLE")
    print(f"wrote {returns_path}")
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()


__all__ = ["main"]
