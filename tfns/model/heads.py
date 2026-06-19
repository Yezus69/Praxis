"""Policy, value, context-key, and auxiliary heads."""

from __future__ import annotations

import math

import flax.linen as nn
import jax.numpy as jnp


def _orthogonal(scale: float):
    return nn.initializers.orthogonal(float(scale))


class PolicyHead(nn.Module):
    act_dim: int = 18

    @nn.compact
    def __call__(self, h: jnp.ndarray) -> jnp.ndarray:
        return nn.Dense(
            features=int(self.act_dim),
            kernel_init=_orthogonal(0.01),
            bias_init=nn.initializers.zeros,
            name="affine",
        )(h)


class ValueHead(nn.Module):
    @nn.compact
    def __call__(self, h: jnp.ndarray) -> jnp.ndarray:
        value = nn.Dense(
            features=1,
            kernel_init=_orthogonal(1.0),
            bias_init=nn.initializers.zeros,
            name="affine",
        )(h)
        return jnp.squeeze(value, axis=-1)


class ContextKeyHead(nn.Module):
    key_dim: int = 128
    key_eps: float = 1e-6

    @nn.compact
    def __call__(self, h: jnp.ndarray) -> jnp.ndarray:
        raw = nn.Dense(
            features=int(self.key_dim),
            kernel_init=_orthogonal(1.0),
            bias_init=nn.initializers.zeros,
            name="affine",
        )(h)
        norm = jnp.linalg.norm(raw, axis=-1, keepdims=True)
        return raw / (norm + float(self.key_eps))


class NextFeatHead(nn.Module):
    dense_dim: int = 512

    @nn.compact
    def __call__(self, h: jnp.ndarray, action_embed: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([h, action_embed], axis=-1)
        x = nn.Dense(
            features=int(self.dense_dim),
            kernel_init=_orthogonal(math.sqrt(2.0)),
            bias_init=nn.initializers.zeros,
            name="hidden",
        )(x)
        x = nn.relu(x)
        return nn.Dense(
            features=int(self.dense_dim),
            kernel_init=_orthogonal(1.0),
            bias_init=nn.initializers.zeros,
            name="out",
        )(x)


class RewardCatHead(nn.Module):
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, h: jnp.ndarray, action_embed: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([h, action_embed], axis=-1)
        x = nn.Dense(
            features=int(self.hidden_dim),
            kernel_init=_orthogonal(math.sqrt(2.0)),
            bias_init=nn.initializers.zeros,
            name="hidden",
        )(x)
        x = nn.relu(x)
        return nn.Dense(
            features=3,
            kernel_init=_orthogonal(1.0),
            bias_init=nn.initializers.zeros,
            name="out",
        )(x)


class TerminalHead(nn.Module):
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, h: jnp.ndarray, action_embed: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([h, action_embed], axis=-1)
        x = nn.Dense(
            features=int(self.hidden_dim),
            kernel_init=_orthogonal(math.sqrt(2.0)),
            bias_init=nn.initializers.zeros,
            name="hidden",
        )(x)
        x = nn.relu(x)
        logit = nn.Dense(
            features=1,
            kernel_init=_orthogonal(1.0),
            bias_init=nn.initializers.zeros,
            name="out",
        )(x)
        return jnp.squeeze(logit, axis=-1)


__all__ = [
    "ContextKeyHead",
    "NextFeatHead",
    "PolicyHead",
    "RewardCatHead",
    "TerminalHead",
    "ValueHead",
]
