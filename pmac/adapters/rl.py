"""Reinforcement-learning adapter for PMA-C section 19.1."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from pmac.adapter import DomainAdapter
from pmac.behavior_distance import kl_categorical, value_abs


def _init_weight(key, in_dim: int, out_dim: int, std: float):
    return jax.random.normal(key, (int(in_dim), int(out_dim)), dtype=jnp.float32) * float(std)


def init_actor_critic(
    key,
    input_dim: int,
    hidden_sizes: tuple[int, ...] = (64, 64),
    num_actions: int = 4,
    scale=None,
):
    """Initialize a shared-trunk actor-critic pytree."""
    hidden_sizes = tuple(int(size) for size in hidden_sizes)
    dims = (int(input_dim), *hidden_sizes)
    n_trunk = max(0, len(dims) - 1)
    keys = jax.random.split(key, n_trunk + 2)
    trunk = []
    for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
        std = float(scale) if scale is not None else (2.0 / float(in_dim)) ** 0.5
        trunk.append(
            {
                "w": _init_weight(keys[i], in_dim, out_dim, std),
                "b": jnp.zeros((out_dim,), dtype=jnp.float32),
            }
        )
    trunk_dim = int(dims[-1])
    head_std = float(scale) if scale is not None else (1.0 / float(trunk_dim)) ** 0.5
    return {
        "trunk": trunk,
        "policy": {
            "w": _init_weight(keys[-2], trunk_dim, int(num_actions), head_std),
            "b": jnp.zeros((int(num_actions),), dtype=jnp.float32),
        },
        "value": {
            "w": _init_weight(keys[-1], trunk_dim, 1, head_std),
            "b": jnp.zeros((1,), dtype=jnp.float32),
        },
    }


def actor_critic_apply(params, x):
    """Apply the shared trunk and return ``(policy_logits, value)``."""
    h = jnp.asarray(x, dtype=jnp.float32)
    for layer in params["trunk"]:
        h = jnp.maximum(h @ layer["w"] + layer["b"], 0.0)
    logits = h @ params["policy"]["w"] + params["policy"]["b"]
    value = jnp.squeeze(h @ params["value"]["w"] + params["value"]["b"], axis=-1)
    return logits, value


def _log_softmax(logits, axis: int = -1):
    shifted = logits - jnp.max(logits, axis=axis, keepdims=True)
    return shifted - jnp.log(jnp.sum(jnp.exp(shifted), axis=axis, keepdims=True))


def _entropy_from_logits(logits):
    logp = _log_softmax(logits)
    p = jnp.exp(logp)
    return -jnp.sum(p * logp, axis=-1)


def _discounted_returns(rewards, mask, gamma: float):
    rewards = jnp.asarray(rewards, dtype=jnp.float32)
    mask = jnp.asarray(mask, dtype=jnp.float32)

    def body(carry, inputs):
        reward, active = inputs
        ret = jnp.where(active > 0.0, reward + float(gamma) * carry, 0.0)
        return ret, ret

    _, rev = jax.lax.scan(body, jnp.zeros_like(rewards[0]), (rewards[::-1], mask[::-1]))
    return rev[::-1]


@dataclass(frozen=True)
class RLLossInfo:
    actor_loss: jnp.ndarray
    value_loss: jnp.ndarray
    entropy: jnp.ndarray
    total_loss: jnp.ndarray


class RLAdapter(DomainAdapter):
    """PMA-C adapter where behavior is policy logits plus scalar value."""

    def __init__(
        self,
        env,
        model_apply=actor_critic_apply,
        gamma: float = 0.99,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        value_distance_coef: float = 1.0,
        temperature: float = 1.0,
        eval_episodes: int = 256,
    ):
        self.env = env
        self.model_apply = model_apply
        self.gamma = float(gamma)
        self.value_loss_coef = float(value_loss_coef)
        self.entropy_coef = float(entropy_coef)
        self.value_distance_coef = float(value_distance_coef)
        self.temperature = float(temperature)
        self.eval_episodes = int(eval_episodes)

    def behavior(self, params, batch):
        x = batch["x"] if isinstance(batch, dict) else batch
        logits, value = self.model_apply(params, x)
        return {
            "policy_logits": logits,
            "value": value,
        }

    def _split_behavior(self, behavior):
        if isinstance(behavior, dict):
            logits = behavior["policy_logits"]
            value = behavior["value"]
        else:
            arr = jnp.asarray(behavior)
            logits = arr[..., : self.env.num_actions]
            value = arr[..., self.env.num_actions]
        return jnp.asarray(logits), jnp.asarray(value)

    def pack_behavior(self, behavior):
        logits, value = self._split_behavior(behavior)
        return jnp.concatenate([logits, jnp.expand_dims(value, axis=-1)], axis=-1)

    def distance(self, cur, teacher, batch=None):
        """Return D_KL(pi* || pi) + lambda_V |V - V*| per example."""
        cur_logits, cur_value = self._split_behavior(cur)
        teacher_logits, teacher_value = self._split_behavior(teacher)
        d_policy = kl_categorical(teacher_logits, cur_logits, self.temperature)
        d_value = value_abs(cur_value, teacher_value)
        return d_policy + self.value_distance_coef * d_value

    def anchor_distance(self, teacher_packed, cur):
        return self.distance(cur, teacher_packed, None)

    def rollout_batch(self, params, key, goal_id, batch_size: int, greedy: bool = False):
        traj = self.env.rollout(
            key,
            self.model_apply,
            params,
            goal_id,
            batch_size=int(batch_size),
            greedy=bool(greedy),
        )
        returns = _discounted_returns(traj.rewards, traj.mask, self.gamma)
        return {
            "obs": traj.obs,
            "actions": traj.actions,
            "returns": returns,
            "mask": traj.mask,
            "reached": traj.reached,
            "rewards": traj.rewards,
        }

    def loss_info(self, params, batch) -> RLLossInfo:
        logits, value = self.model_apply(params, batch["obs"])
        logp = _log_softmax(logits)
        action_logp = jnp.take_along_axis(
            logp, jnp.expand_dims(batch["actions"], axis=-1), axis=-1
        )
        action_logp = jnp.squeeze(action_logp, axis=-1)
        mask = jnp.asarray(batch["mask"], dtype=jnp.float32)
        denom = jnp.maximum(jnp.sum(mask), 1.0)
        returns = jnp.asarray(batch["returns"], dtype=jnp.float32)
        advantage = jax.lax.stop_gradient(returns - value)

        actor_loss = -jnp.sum(action_logp * advantage * mask) / denom
        value_loss = jnp.sum((value - returns) * (value - returns) * mask) / denom
        entropy = jnp.sum(_entropy_from_logits(logits) * mask) / denom
        total = actor_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
        return RLLossInfo(actor_loss, value_loss, entropy, total)

    def current_loss(self, params, batch):
        return self.loss_info(params, batch).total_loss

    def evaluate_skill(self, params, skill_eval_set) -> float:
        goal_id = int(skill_eval_set.get("goal_id", 0))
        num_episodes = int(skill_eval_set.get("num_episodes", self.eval_episodes))
        greedy = bool(skill_eval_set.get("greedy", True))
        key = skill_eval_set.get("key", jax.random.PRNGKey(0))
        traj = self.env.rollout(
            key,
            self.model_apply,
            params,
            goal_id,
            batch_size=num_episodes,
            greedy=greedy,
        )
        return float(jnp.mean(traj.reached[-1].astype(jnp.float32)))

    def make_challenge_batch(self, key, skill_eval_set):
        goal_id = int(skill_eval_set.get("goal_id", 0))
        return {
            "x": self.env.all_observations_for_goal(goal_id),
            "goal_id": goal_id,
            "key": key,
        }

    def grow_capacity(self, key, params, source):
        return params


__all__ = [
    "RLAdapter",
    "RLLossInfo",
    "actor_critic_apply",
    "init_actor_critic",
]
