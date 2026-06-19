"""Minimal CleanRL-style feed-forward PPO for one envpool Atari game."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax


ACT_DIM = 18
FIRE_ACTION = 1
EVAL_INTERVAL = 50


@dataclass(frozen=True)
class PPOConfig:
    game: str = "Breakout-v5"
    total_steps: int = 5_000_000
    num_envs: int = 16
    seed: int = 1
    eval_episodes: int = 30
    fire_reset: bool = True
    out_dir: str = "baseline_ppo_runs"
    num_steps: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_coef: float = 0.1
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr: float = 2.5e-4


class TrainBatch(NamedTuple):
    obs: jnp.ndarray
    actions: jnp.ndarray
    logprobs: jnp.ndarray
    advantages: jnp.ndarray
    returns: jnp.ndarray
    values: jnp.ndarray


class NatureActorCritic(nn.Module):
    @nn.compact
    def __call__(self, obs):
        obs = jnp.asarray(obs)
        if obs.ndim == 3:
            obs = obs[None]
        x = obs.astype(jnp.float32) / 255.0
        init = nn.initializers.orthogonal(math.sqrt(2.0))
        x = nn.relu(nn.Conv(32, (8, 8), (4, 4), padding="VALID", kernel_init=init)(x))
        x = nn.relu(nn.Conv(64, (4, 4), (2, 2), padding="VALID", kernel_init=init)(x))
        x = nn.relu(nn.Conv(64, (3, 3), (1, 1), padding="VALID", kernel_init=init)(x))
        x = nn.relu(nn.Dense(512, kernel_init=init)(x.reshape((x.shape[0], -1))))
        logits = nn.Dense(ACT_DIM, kernel_init=nn.initializers.orthogonal(0.01), bias_init=nn.initializers.zeros)(x)
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0), bias_init=nn.initializers.zeros)(x)
        return logits, jnp.squeeze(value, axis=-1)


def make_env(game: str, num_envs: int, seed: int, *, training: bool):
    import envpool

    return envpool.make(
        game,
        env_type="gymnasium",
        num_envs=int(num_envs),
        seed=int(seed),
        full_action_space=True,
        episodic_life=bool(training),
        reward_clip=bool(training),
    )


def reset_result(result: Any) -> tuple[Any, dict[str, Any]]:
    return result if isinstance(result, tuple) and len(result) == 2 else (result, {})


def nhwc_uint8(obs: Any) -> np.ndarray:
    arr = np.asarray(obs)
    if arr.ndim != 4:
        raise ValueError(f"expected batched Atari obs with 4 dims, got {arr.shape}")
    if arr.shape[-1] == 4:
        out = arr
    elif arr.shape[1] == 4:
        out = np.moveaxis(arr, 1, -1)
    else:
        raise ValueError(f"expected NHWC or NCHW 4-frame obs, got {arr.shape}")
    return np.ascontiguousarray(out.astype(np.uint8, copy=False))


def vec(value: Any, dtype: Any, num_envs: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    if arr.ndim == 0:
        arr = np.full((num_envs,), arr.item(), dtype=dtype)
    arr = arr.reshape(-1)
    if arr.shape[0] != num_envs:
        raise ValueError(f"{name} must have shape ({num_envs},), got {arr.shape}")
    return arr


def log_prob(logits, actions):
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return jnp.take_along_axis(log_probs, actions[..., None], axis=-1)[..., 0]


@jax.jit
def policy_step(params, obs, force_fire, rng):
    rng, key = jax.random.split(rng)
    logits, value = NatureActorCritic().apply({"params": params}, obs)
    sampled = jax.random.categorical(key, logits, axis=-1).astype(jnp.int32)
    actions = jnp.where(force_fire, jnp.full_like(sampled, FIRE_ACTION), sampled)
    return actions, log_prob(logits, actions), value, rng


@jax.jit
def greedy_policy(params, obs, force_fire):
    logits, _ = NatureActorCritic().apply({"params": params}, obs)
    greedy = jnp.argmax(logits, axis=-1).astype(jnp.int32)
    return jnp.where(force_fire, jnp.full_like(greedy, FIRE_ACTION), greedy)


@jax.jit
def value_only(params, obs):
    return NatureActorCritic().apply({"params": params}, obs)[1]


@jax.jit
def compute_gae(rewards, dones, values, last_value, gamma: float, lam: float):
    def step(carry, row):
        gae, next_value = carry
        reward, done, value = row
        not_done = 1.0 - done
        delta = reward + gamma * next_value * not_done - value
        gae = delta + gamma * lam * not_done * gae
        return (gae, value), gae

    _, adv_rev = jax.lax.scan(
        step,
        (jnp.zeros_like(last_value), last_value),
        (rewards[::-1], dones[::-1], values[::-1]),
    )
    adv = adv_rev[::-1]
    return jax.lax.stop_gradient(adv), jax.lax.stop_gradient(adv + values)


def flatten_batch(obs, actions, logprobs, adv, returns, values, size: int) -> TrainBatch:
    return TrainBatch(
        obs=jnp.asarray(obs).reshape((size,) + tuple(obs.shape[2:])),
        actions=jnp.asarray(actions, dtype=jnp.int32).reshape((size,)),
        logprobs=jnp.asarray(logprobs, dtype=jnp.float32).reshape((size,)),
        advantages=jnp.asarray(adv, dtype=jnp.float32).reshape((size,)),
        returns=jnp.asarray(returns, dtype=jnp.float32).reshape((size,)),
        values=jnp.asarray(values, dtype=jnp.float32).reshape((size,)),
    )


def ppo_loss(params, batch: TrainBatch, clip: float, vf_coef: float, ent_coef: float):
    logits, new_values = NatureActorCritic().apply({"params": params}, batch.obs)
    new_logprobs = log_prob(logits, batch.actions)
    logratio = new_logprobs - batch.logprobs
    ratio = jnp.exp(logratio)
    adv = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

    pg_loss = jnp.maximum(-adv * ratio, -adv * jnp.clip(ratio, 1.0 - clip, 1.0 + clip)).mean()
    v_unclipped = jnp.square(new_values - batch.returns)
    v_clipped = batch.values + jnp.clip(new_values - batch.values, -clip, clip)
    v_loss = 0.5 * jnp.maximum(v_unclipped, jnp.square(v_clipped - batch.returns)).mean()
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    entropy = -(jnp.exp(log_probs) * log_probs).sum(axis=-1).mean()
    approx_kl = ((ratio - 1.0) - logratio).mean()
    clipfrac = (jnp.abs(ratio - 1.0) > clip).astype(jnp.float32).mean()
    loss = pg_loss + vf_coef * v_loss - ent_coef * entropy
    return loss, jnp.asarray([loss, pg_loss, v_loss, entropy, approx_kl, clipfrac])


@partial(jax.jit, static_argnames=("epochs", "minibatches", "minibatch_size", "clip", "vf", "ent", "grad_norm"))
def update_ppo(params, opt_state, batch, rng, lr, epochs, minibatches, minibatch_size, clip, vf, ent, grad_norm):
    tx = optax.chain(optax.clip_by_global_norm(grad_norm), optax.adam(lr))
    batch_size = int(minibatches) * int(minibatch_size)

    def epoch_step(carry, _):
        params, opt_state, rng = carry
        rng, perm_key = jax.random.split(rng)
        perm = jax.random.permutation(perm_key, batch_size)

        def shuffle(x):
            return jnp.take(x, perm, axis=0).reshape((minibatches, minibatch_size) + x.shape[1:])

        def mb_step(carry, mb):
            params, opt_state = carry
            (loss, metrics), grads = jax.value_and_grad(ppo_loss, has_aux=True)(params, mb, clip, vf, ent)
            updates, opt_state = tx.update(grads, opt_state, params)
            return (optax.apply_updates(params, updates), opt_state), metrics

        (params, opt_state), metrics = jax.lax.scan(mb_step, (params, opt_state), jax.tree_util.tree_map(shuffle, batch))
        return (params, opt_state, rng), metrics.mean(axis=0)

    (params, opt_state, rng), metrics = jax.lax.scan(epoch_step, (params, opt_state, rng), None, length=epochs)
    return params, opt_state, rng, metrics.mean(axis=0)


def evaluate(params, cfg: PPOConfig, seed: int) -> dict[str, Any]:
    n = min(int(cfg.num_envs), int(cfg.eval_episodes))
    env = make_env(cfg.game, n, seed, training=False)
    obs, _ = reset_result(env.reset())
    obs = nhwc_uint8(obs)
    running = np.zeros((n,), dtype=np.float32)
    fire = np.full((n,), bool(cfg.fire_reset), dtype=np.bool_)
    completed: list[float] = []

    while len(completed) < int(cfg.eval_episodes):
        actions = greedy_policy(params, obs, jnp.asarray(fire))
        obs, reward, terminated, truncated, _ = env.step(np.asarray(jax.device_get(actions), np.int32))
        obs = nhwc_uint8(obs)
        done = np.logical_or(vec(terminated, np.bool_, n, "terminated"), vec(truncated, np.bool_, n, "truncated"))
        running += vec(reward, np.float32, n, "reward")
        for idx in np.flatnonzero(done):
            if len(completed) < int(cfg.eval_episodes):
                completed.append(float(running[idx]))
            running[idx] = 0.0
        fire = np.where(done, bool(cfg.fire_reset), False)

    close = getattr(env, "close", None)
    if callable(close):
        close()
    arr = np.asarray(completed, dtype=np.float32)
    return {
        "episodes": len(completed),
        "mean_return": float(arr.mean()),
        "median_return": float(np.median(arr)),
        "min_return": float(arr.min()),
        "max_return": float(arr.max()),
        "returns": [float(x) for x in completed],
    }


def parse_args() -> tuple[PPOConfig, str]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, default="Breakout-v5")
    parser.add_argument("--total-steps", type=int, default=5_000_000)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--fire-reset", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out-dir", type=str, default="baseline_ppo_runs")
    args = parser.parse_args()
    cmd = subprocess.list2cmdline(
        [
            "python",
            "baseline_ppo.py",
            "--game",
            args.game,
            "--total-steps",
            str(args.total_steps),
            "--num-envs",
            str(args.num_envs),
            "--seed",
            str(args.seed),
            "--eval-episodes",
            str(args.eval_episodes),
            "--out-dir",
            args.out_dir,
            "--fire-reset" if args.fire_reset else "--no-fire-reset",
        ]
    )
    return PPOConfig(**vars(args)), cmd


def main() -> None:
    cfg, cmd = parse_args()
    batch_size = int(cfg.num_envs) * int(cfg.num_steps)
    if batch_size % int(cfg.num_minibatches):
        raise ValueError("num_envs*num_steps must be divisible by num_minibatches")
    num_updates = int(cfg.total_steps) // batch_size
    if num_updates <= 0:
        raise ValueError("total_steps must cover at least one PPO update")

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Baseline PPO | game={cfg.game} total_steps={cfg.total_steps} num_envs={cfg.num_envs} num_steps={cfg.num_steps} seed={cfg.seed} fire_reset={cfg.fire_reset}")
    print("Launch command: " + cmd)

    rng = jax.random.PRNGKey(int(cfg.seed))
    rng, init_key = jax.random.split(rng)
    params = NatureActorCritic().init(init_key, jnp.zeros((1, 84, 84, 4), dtype=jnp.uint8))["params"]
    opt_state = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(cfg.lr)).init(params)

    env = make_env(cfg.game, cfg.num_envs, cfg.seed, training=True)
    obs, _ = reset_result(env.reset())
    next_obs = nhwc_uint8(obs)
    fire = np.full((cfg.num_envs,), bool(cfg.fire_reset), dtype=np.bool_)
    obs_buf = np.zeros((cfg.num_steps, cfg.num_envs, 84, 84, 4), dtype=np.uint8)
    actions_buf = np.zeros((cfg.num_steps, cfg.num_envs), dtype=np.int32)
    logprobs_buf = np.zeros((cfg.num_steps, cfg.num_envs), dtype=np.float32)
    rewards_buf = np.zeros((cfg.num_steps, cfg.num_envs), dtype=np.float32)
    dones_buf = np.zeros((cfg.num_steps, cfg.num_envs), dtype=np.float32)
    values_buf = np.zeros((cfg.num_steps, cfg.num_envs), dtype=np.float32)
    curve: list[dict[str, Any]] = []
    start = time.perf_counter()

    for update in range(1, num_updates + 1):
        for step in range(int(cfg.num_steps)):
            obs_buf[step] = next_obs
            actions, logprobs, values, rng = policy_step(params, next_obs, jnp.asarray(fire), rng)
            actions_np = np.asarray(jax.device_get(actions), dtype=np.int32)
            actions_buf[step] = actions_np
            logprobs_buf[step] = np.asarray(jax.device_get(logprobs), dtype=np.float32)
            values_buf[step] = np.asarray(jax.device_get(values), dtype=np.float32)
            next_obs, reward, terminated, truncated, _ = env.step(actions_np)
            next_obs = nhwc_uint8(next_obs)
            done = np.logical_or(
                vec(terminated, np.bool_, cfg.num_envs, "terminated"),
                vec(truncated, np.bool_, cfg.num_envs, "truncated"),
            )
            rewards_buf[step] = vec(reward, np.float32, cfg.num_envs, "reward")
            dones_buf[step] = done.astype(np.float32)
            fire = np.where(done, bool(cfg.fire_reset), False)

        adv, returns = compute_gae(jnp.asarray(rewards_buf), jnp.asarray(dones_buf), jnp.asarray(values_buf), value_only(params, next_obs), cfg.gamma, cfg.gae_lambda)
        batch = flatten_batch(obs_buf, actions_buf, logprobs_buf, adv, returns, values_buf, batch_size)
        lr = cfg.lr * (1.0 - (update - 1.0) / float(num_updates))
        params, opt_state, rng, metrics = update_ppo(params, opt_state, batch, rng, lr, cfg.update_epochs, cfg.num_minibatches, batch_size // cfg.num_minibatches, cfg.clip_coef, cfg.vf_coef, cfg.ent_coef, cfg.max_grad_norm)
        metrics_np = np.asarray(jax.device_get(metrics), dtype=np.float32)

        if update == 1 or update % EVAL_INTERVAL == 0 or update == num_updates:
            steps = update * batch_size
            record = {
                "update": update,
                "timesteps": steps,
                "sps": float(steps / max(time.perf_counter() - start, 1e-6)),
                **evaluate(params, cfg, cfg.seed + 10_000 + update),
                "loss": float(metrics_np[0]),
                "policy_loss": float(metrics_np[1]),
                "value_loss": float(metrics_np[2]),
                "entropy": float(metrics_np[3]),
                "approx_kl": float(metrics_np[4]),
                "clipfrac": float(metrics_np[5]),
            }
            curve.append(record)
            with (out_dir / "progress.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            print(
                f"eval update={update} steps={steps} mean={record['mean_return']:.2f} "
                f"median={record['median_return']:.2f} min={record['min_return']:.2f} "
                f"max={record['max_return']:.2f} sps={record['sps']:.0f}",
                flush=True,
            )

    close = getattr(env, "close", None)
    if callable(close):
        close()
    best = max(curve, key=lambda x: x["mean_return"]) if curve else None
    final = curve[-1] if curve else None
    final_path = out_dir / "final.json"
    payload = {"config": asdict(cfg), "updates": num_updates, "timesteps": num_updates * batch_size, "learning_curve": curve, "best_eval": best, "final_eval": final}
    final_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if final is not None:
        print(
            f"done final_mean={final['mean_return']:.2f} "
            f"best_mean={best['mean_return']:.2f} final_json={final_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
