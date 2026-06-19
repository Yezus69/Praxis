"""Explicit projectable GRU cell."""

from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax.numpy as jnp


def _orthogonal(scale: float):
    return nn.initializers.orthogonal(float(scale))


def gru_presynaptic(x_t: jnp.ndarray, h_prev: jnp.ndarray) -> jnp.ndarray:
    """Return the shared augmented presynaptic vector ``[x_t; h_prev; 1]``."""

    ones = jnp.ones(x_t.shape[:-1] + (1,), dtype=x_t.dtype)
    return jnp.concatenate([x_t, h_prev, ones], axis=-1)


class ExplicitGRU(nn.Module):
    """GRU with separately named gate matrices for projection.

    The candidate uses the cuDNN-style reset-after-recurrent-matmul form:
    ``tanh(W_n x + r * (U_n h) + b_n)``. The incoming hidden state is zeroed
    wherever ``reset`` is true before any gate computation.
    """

    hidden: int = 512

    def presynaptic(self, x_t: jnp.ndarray, h_prev: jnp.ndarray) -> jnp.ndarray:
        """Return the shared augmented presynaptic vector for basis capture."""

        return gru_presynaptic(x_t, h_prev)

    @nn.compact
    def __call__(
        self,
        x_t: jnp.ndarray,
        h_prev: jnp.ndarray,
        reset: jnp.ndarray,
        return_presyn: bool = False,
    ) -> Any:
        x_t = jnp.asarray(x_t, dtype=jnp.float32)
        h_prev = jnp.asarray(h_prev, dtype=jnp.float32)
        reset = jnp.asarray(reset, dtype=bool)
        h_in = jnp.where(reset[..., None], jnp.zeros_like(h_prev), h_prev)
        xi_t = self.presynaptic(x_t, h_in)

        input_dim = int(x_t.shape[-1])
        hidden = int(self.hidden)
        k_init = _orthogonal(1.0)
        r_init = _orthogonal(1.0)
        b_init = nn.initializers.zeros

        W_z = self.param("W_z", k_init, (input_dim, hidden))
        U_z = self.param("U_z", r_init, (hidden, hidden))
        b_z = self.param("b_z", b_init, (hidden,))
        W_r = self.param("W_r", k_init, (input_dim, hidden))
        U_r = self.param("U_r", r_init, (hidden, hidden))
        b_r = self.param("b_r", b_init, (hidden,))
        W_n = self.param("W_n", k_init, (input_dim, hidden))
        U_n = self.param("U_n", r_init, (hidden, hidden))
        b_n = self.param("b_n", b_init, (hidden,))

        z_t = nn.sigmoid(jnp.matmul(x_t, W_z) + jnp.matmul(h_in, U_z) + b_z)
        r_t = nn.sigmoid(jnp.matmul(x_t, W_r) + jnp.matmul(h_in, U_r) + b_r)
        n_t = jnp.tanh(jnp.matmul(x_t, W_n) + r_t * jnp.matmul(h_in, U_n) + b_n)
        h_t = (1.0 - z_t) * n_t + z_t * h_in
        if return_presyn:
            return h_t, xi_t
        return h_t


__all__ = ["ExplicitGRU", "gru_presynaptic"]
