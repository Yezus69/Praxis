"""Smoke-train one MinAtar game with bounded JAX PPO."""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np

from pmac.agents.ppo_minatar import PPOConfig, train_ppo_single
from pmac.envs.minatar_gymnax import GAMES, make_games

warnings.filterwarnings("ignore")


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
    ax.set_title("MinAtar PPO return")
    ax.set_xlabel("PPO update")
    ax.set_ylabel("episodic return")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="Breakout-MinAtar", choices=GAMES)
    parser.add_argument("--total-steps", type=int, default=3_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="/root/minatar_smoke")
    args = parser.parse_args(argv)

    specs = make_games(GAMES)
    by_name = {spec.name: spec for spec in specs}
    game_spec = by_name[args.game]
    cfg = PPOConfig(total_timesteps=int(args.total_steps))
    result = train_ppo_single(
        game_spec=game_spec,
        n_games=len(specs),
        act_max=game_spec.act_max,
        cfg=cfg,
        seed=int(args.seed),
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    curve = result["returns_curve"]
    first_return = float(curve[0]) if curve else 0.0
    final_return = float(result["final_return"])
    learnable = final_return >= first_return + max(1.0, 0.25 * abs(first_return))
    downsampled = _downsample(curve)

    payload = {
        "game": args.game,
        "seed": int(args.seed),
        "timesteps": int(result["timesteps"]),
        "config": asdict(cfg),
        "returns_curve": curve,
        "returns_curve_downsampled": downsampled,
        "first_update_return": first_return,
        "final_return": final_return,
        "learnable": bool(learnable),
    }
    returns_path = out_dir / "returns.json"
    with returns_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    plot_path = out_dir / "returns.png"
    _plot_curve(curve, plot_path)

    print("returns_curve_downsampled=" + json.dumps(downsampled))
    print(f"final_return={final_return:.6f}")
    print("LEARNABLE" if learnable else "NOT_LEARNABLE")
    print(f"wrote {returns_path}")
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()


__all__ = ["main"]
