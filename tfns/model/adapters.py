"""Fixed low-rank residual adapter bank."""

from __future__ import annotations

import math

import flax.linen as nn
import jax
import jax.numpy as jnp


def _orthogonal(scale: float):
    return nn.initializers.orthogonal(float(scale))


class ResidualAdapterBank(nn.Module):
    """Sparse top-k low-rank residual adapter bank.

    ``dormant`` is a boolean mask with one entry per adapter. Dormant adapters
    are removed from routing and contribute exactly zero.
    """

    num_adapters: int = 8
    rank: int = 32
    top_k: int = 2
    residual_rank: int = 0

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        router_input: jnp.ndarray,
        dormant: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        x = jnp.asarray(x, dtype=jnp.float32)
        router_input = jnp.asarray(router_input, dtype=jnp.float32)
        dormant = jnp.asarray(dormant, dtype=bool)
        input_dim = int(x.shape[-1])
        num_adapters = int(self.num_adapters)
        rank = int(self.rank)
        k = min(int(self.top_k), num_adapters)

        V = self.param(
            "V",
            _orthogonal(math.sqrt(2.0)),
            (num_adapters, input_dim, rank),
        )
        U = self.param(
            "U",
            nn.initializers.zeros,
            (num_adapters, rank, input_dim),
        )

        # Nullspace-residual gating. ``U_res`` is a stop-gradient buffer holding an
        # orthonormal basis of OLD-task activations (written at consolidation; zero
        # at init). The adapter sees only the residual orthogonal to that basis, so
        # old-task inputs (which lie in the basis) drive the adapter to ~0 and old
        # behavior is preserved regardless of adapter/router weights — while novel
        # inputs (orthogonal to old tasks) pass through and the adapter learns them.
        res_rank = int(self.residual_rank)
        if res_rank > 0:
            U_res = jax.lax.stop_gradient(
                self.param("U_res", nn.initializers.zeros, (input_dim, res_rank))
            )
            x_adapter = x - (x @ U_res) @ U_res.T
        else:
            x_adapter = x

        logits = nn.Dense(
            features=num_adapters,
            kernel_init=_orthogonal(0.01),
            bias_init=nn.initializers.zeros,
            name="router",
        )(router_input)

        active = jnp.logical_not(dormant)
        masked_logits = jnp.where(active[None, :], logits, -jnp.inf)
        top_values, top_indices = jax.lax.top_k(masked_logits, k)
        finite = jnp.isfinite(top_values)
        safe_values = jnp.where(finite, top_values, -1.0e9)
        top_weights = nn.softmax(safe_values, axis=-1) * finite.astype(x.dtype)
        denom = jnp.sum(top_weights, axis=-1, keepdims=True)
        top_weights = jnp.where(denom > 0.0, top_weights / denom, jnp.zeros_like(top_weights))
        router_weights = jnp.zeros(logits.shape, dtype=x.dtype)
        router_weights = router_weights.at[
            jnp.arange(x.shape[0])[:, None], top_indices
        ].add(top_weights)

        down = jnp.einsum("bd,kdr->bkr", x_adapter, V)
        adapter_hidden = nn.relu(down)
        adapter_delta = jnp.einsum("bkr,kro->bko", adapter_hidden, U)
        delta = jnp.sum(router_weights[..., None] * adapter_delta, axis=1)
        return x + delta, router_weights


__all__ = ["ResidualAdapterBank"]
