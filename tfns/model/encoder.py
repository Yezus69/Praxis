"""Visual encoder for task-free recurrent Atari agents."""

from __future__ import annotations

import math
from typing import Any

import flax.linen as nn
import jax.numpy as jnp


def _orthogonal(scale: float):
    return nn.initializers.orthogonal(float(scale))


def _apply_activation(x: jnp.ndarray, activation: str) -> jnp.ndarray:
    if activation == "relu":
        return nn.relu(x)
    if activation == "crelu":
        return jnp.concatenate([nn.relu(x), nn.relu(-x)], axis=-1)
    raise ValueError(f"unsupported activation {activation!r}")


class Encoder(nn.Module):
    """Nature-style Atari encoder returning a fixed-width visual feature.

    If ``collect_presyn`` is true, affine inputs are returned for projection
    and protected-basis construction.
    """

    dense_dim: int = 512
    activation: str = "relu"
    frame_stack: int = 4
    obs_hw: int = 84
    conv_channels: tuple[int, int, int] = (32, 64, 64)
    conv_kernels: tuple[int, int, int] = (8, 4, 3)
    conv_strides: tuple[int, int, int] = (4, 2, 1)

    @nn.compact
    def __call__(self, obs: jnp.ndarray, collect_presyn: bool = False) -> Any:
        obs = jnp.asarray(obs)
        if jnp.issubdtype(obs.dtype, jnp.integer):
            obs = obs.astype(jnp.float32) / 255.0
        else:
            obs = obs.astype(jnp.float32)
        if (
            obs.ndim != 4
            or obs.shape[1] != self.obs_hw
            or obs.shape[2] != self.obs_hw
            or obs.shape[-1] != self.frame_stack
        ):
            raise ValueError(
                f"expected NHWC obs with shape (B, {self.obs_hw}, {self.obs_hw}, "
                f"{self.frame_stack}), got {obs.shape}"
            )

        conv_init = _orthogonal(math.sqrt(2.0))
        x = obs
        presyn = {"conv1_in": obs} if collect_presyn else None
        for idx, (channels, kernel, stride) in enumerate(
            zip(self.conv_channels, self.conv_kernels, self.conv_strides, strict=True),
            start=1,
        ):
            if collect_presyn and idx > 1:
                assert presyn is not None
                presyn[f"conv{idx}_in"] = x
            x = nn.Conv(
                features=int(channels),
                kernel_size=(int(kernel), int(kernel)),
                strides=(int(stride), int(stride)),
                padding="VALID",
                kernel_init=conv_init,
                bias_init=nn.initializers.zeros,
                name=f"conv{idx}",
            )(x)
            x = _apply_activation(x, self.activation)

        dense_in = x.reshape((x.shape[0], -1))
        dense_features = self.dense_dim
        if self.activation == "crelu":
            if self.dense_dim % 2 != 0:
                raise ValueError("dense_dim must be even when activation='crelu'")
            dense_features = self.dense_dim // 2
        x = nn.Dense(
            features=int(dense_features),
            kernel_init=_orthogonal(math.sqrt(2.0)),
            bias_init=nn.initializers.zeros,
            name="dense",
        )(dense_in)
        e_t = _apply_activation(x, self.activation)
        if collect_presyn:
            assert presyn is not None
            presyn["encoder_dense"] = dense_in
            return e_t, presyn
        return e_t


__all__ = ["Encoder"]
