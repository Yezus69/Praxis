"""Supervised-learning adapter from PMA-C section 19.4."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from pmac.adapter import DomainAdapter
from pmac.behavior_distance import kl_categorical
from pmac.models.mlp import grow_adapter, mlp_apply


class SupervisedAdapter(DomainAdapter):
    """PMA-C adapter where behavior is class logits."""

    def __init__(self, model_apply=mlp_apply, temperature=2.0, noise_std=0.05, growth_rank=64):
        self.model_apply = model_apply
        self.temperature = float(temperature)
        self.noise_std = float(noise_std)
        self.growth_rank = int(growth_rank)

    def behavior(self, params, batch):
        return self.model_apply(params, batch["x"])

    def distance(self, cur_logits, teacher_logits, batch=None):
        return kl_categorical(teacher_logits, cur_logits, self.temperature)

    def current_loss(self, params, batch):
        logits = self.behavior(params, batch)
        losses = optax.softmax_cross_entropy_with_integer_labels(logits, batch["y"])
        return jnp.mean(losses)

    def evaluate_skill(self, params, skill_eval_set) -> float:
        logits = self.behavior(params, skill_eval_set)
        pred = jnp.argmax(logits, axis=-1)
        return float(jnp.mean(pred == skill_eval_set["y"]))

    def make_challenge_batch(self, key, skill_eval_set):
        challenge = dict(skill_eval_set)
        x = jnp.asarray(skill_eval_set["x"])
        noise = self.noise_std * jax.random.normal(key, x.shape, dtype=x.dtype)
        challenge["x"] = x + noise
        return challenge

    def grow_capacity(self, key, params, source):
        hidden_dim = int(params["layers"][-1]["w"].shape[0])
        rank = self.growth_rank
        if isinstance(source, dict):
            hidden_dim = int(source.get("hidden_dim", hidden_dim))
            rank = int(source.get("rank", rank))
        return grow_adapter(key, params, hidden_dim, rank=rank)


__all__ = ["SupervisedAdapter"]
