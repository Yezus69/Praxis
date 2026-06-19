"""Separate delayed-return predictor for causal credit and shaping.

``ReturnPredictor`` consumes time-major feature sequences, not raw pixels and
not task identity. Its indexing is:

- ``H_t`` is the recurrent history before action ``a_t``.
- ``H_{t+1}`` is the history after observing ``a_t``'s consequence.
- ``unroll`` returns heads at ``H_0..H_T``, so ``F_seq`` and ``Phi_seq`` have
  length ``T + 1`` for an input transition sequence of length ``T``.

Features are stop-gradient inputs. Predictor parameters and optimizer state are
separate from policy parameters; callers should freeze a predictor snapshot for
each rollout block before using its outputs for priorities or shaping.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from tfns.model.gru import ExplicitGRU


_MISSING = object()


class ReturnPredictor(nn.Module):
    """Small causal recurrent return predictor over stop-gradient features."""

    act_dim: int = 18
    action_embed_dim: int = 16
    hidden: int = 64

    def init_hidden(self, batch_shape: int | tuple[int, ...] = (), dtype=jnp.float32) -> jnp.ndarray:
        """Return a zero hidden state with shape ``batch_shape + (hidden,)``."""

        if isinstance(batch_shape, int):
            batch_shape = (int(batch_shape),)
        else:
            batch_shape = tuple(int(dim) for dim in batch_shape)
        return jnp.zeros(batch_shape + (int(self.hidden),), dtype=dtype)

    @nn.compact
    def __call__(
        self,
        feat_seq: jnp.ndarray,
        act_seq: jnp.ndarray,
        rew_seq: jnp.ndarray,
        reset_seq: jnp.ndarray,
        h0: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return ``(F_seq, Phi_seq)`` at ``H_0..H_T``.

        ``reset_seq[t]`` zeroes the incoming hidden state before transition
        ``t`` is consumed, matching the policy recurrent convention. The reset
        bit is also part of the predictor input for that transition.
        """

        feat_seq = jax.lax.stop_gradient(jnp.asarray(feat_seq, dtype=jnp.float32))
        act_seq = jnp.asarray(act_seq, dtype=jnp.int32)
        rew_seq = jnp.asarray(rew_seq, dtype=jnp.float32)
        reset_seq = jnp.asarray(reset_seq, dtype=bool)
        h0 = jnp.asarray(h0, dtype=jnp.float32)

        action_embed = nn.Embed(
            num_embeddings=int(self.act_dim),
            features=int(self.action_embed_dim),
            embedding_init=nn.initializers.normal(stddev=0.01),
            name="action_embed",
        )
        gru = ExplicitGRU(hidden=int(self.hidden), name="gru")
        F_head = nn.Dense(1, name="F_head")
        Phi_head = nn.Dense(1, name="Phi_head")

        def heads(hidden):
            F = jnp.squeeze(F_head(hidden), axis=-1)
            Phi = jnp.squeeze(Phi_head(hidden), axis=-1)
            return F, Phi

        h_t = h0
        F_values = []
        Phi_values = []
        for t in range(int(feat_seq.shape[0])):
            feat_t = feat_seq[t]
            act_t = act_seq[t]
            rew_t = rew_seq[t]
            reset_t = reset_seq[t]
            h_in = jnp.where(reset_t[..., None], jnp.zeros_like(h_t), h_t)
            F_t, Phi_t = heads(h_in)
            F_values.append(F_t)
            Phi_values.append(Phi_t)

            emb_t = action_embed(act_t)
            rew_col = jnp.expand_dims(rew_t, axis=-1)
            reset_col = jnp.expand_dims(reset_t.astype(jnp.float32), axis=-1)
            x_t = jnp.concatenate([feat_t, emb_t, rew_col, reset_col], axis=-1)
            h_t = gru(x_t, h_t, reset_t).astype(h_t.dtype)

        F_T, Phi_T = heads(h_t)
        if F_values:
            F_seq = jnp.concatenate([jnp.stack(F_values, axis=0), F_T[None, ...]], axis=0)
            Phi_seq = jnp.concatenate([jnp.stack(Phi_values, axis=0), Phi_T[None, ...]], axis=0)
        else:
            F_seq = F_T[None, ...]
            Phi_seq = Phi_T[None, ...]
        return F_seq, Phi_seq

    def unroll(
        self,
        params: Mapping[str, Any],
        feat_seq: jnp.ndarray,
        act_seq: jnp.ndarray,
        rew_seq: jnp.ndarray,
        reset_seq: jnp.ndarray,
        h0: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Apply the predictor with an explicit params tree."""

        variables = params if "params" in params else {"params": params}
        return self.apply(variables, feat_seq, act_seq, rew_seq, reset_seq, h0)


def discounted_returns(
    rewards: Any,
    gamma: float,
    episode_end_mask: Any | None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``(G0_per_episode_broadcast, remaining_t)``.

    ``remaining_t`` is the discounted return from step ``t`` to the end of the
    same episode. ``G0_per_episode_broadcast`` is that episode's start return
    broadcast to every transition in the episode. ``episode_end_mask[t]`` is
    true on a terminal transition and resets the reverse return scan there.
    """

    rewards = jnp.asarray(rewards, dtype=jnp.float32)
    if episode_end_mask is None:
        episode_end_mask = jnp.zeros(rewards.shape, dtype=bool)
    else:
        episode_end_mask = jnp.broadcast_to(
            jnp.asarray(episode_end_mask, dtype=bool),
            rewards.shape,
        )

    gamma = jnp.asarray(gamma, dtype=rewards.dtype)

    def reverse_step(carry, inputs):
        reward_t, end_t = inputs
        future = jnp.where(end_t, jnp.zeros_like(carry), carry)
        ret_t = reward_t + gamma * future
        return ret_t, ret_t

    _, remaining = jax.lax.scan(
        reverse_step,
        jnp.zeros_like(rewards[0]),
        (rewards, episode_end_mask),
        reverse=True,
    )

    starts = jnp.concatenate(
        [jnp.ones_like(episode_end_mask[:1], dtype=bool), episode_end_mask[:-1]],
        axis=0,
    )

    def forward_step(carry, inputs):
        remaining_t, start_t = inputs
        g0_t = jnp.where(start_t, remaining_t, carry)
        return g0_t, g0_t

    _, G0 = jax.lax.scan(
        forward_step,
        jnp.zeros_like(rewards[0]),
        (remaining, starts),
    )
    return jax.lax.stop_gradient(G0), jax.lax.stop_gradient(remaining)


def _field(batch: Mapping[str, Any], *names: str, default: Any = _MISSING) -> Any:
    for name in names:
        if name in batch:
            return batch[name]
    if default is not _MISSING:
        return default
    raise KeyError(f"batch missing any of fields: {names}")


def _cfg_value(cfg: Any, *names: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        for name in names:
            if name in cfg:
                return cfg[name]
    for name in names:
        if hasattr(cfg, name):
            return getattr(cfg, name)
    return default


def _model_from_batch(batch: Mapping[str, Any]) -> ReturnPredictor:
    model = _field(batch, "model", "predictor", default=None)
    if model is not None:
        return model
    return ReturnPredictor(
        act_dim=int(_field(batch, "act_dim", default=18)),
        action_embed_dim=int(_field(batch, "action_embed_dim", default=16)),
        hidden=int(_field(batch, "hidden", "hidden_dim", default=64)),
    )


def _h0_from_batch(model: ReturnPredictor, batch: Mapping[str, Any], feat_seq: jnp.ndarray) -> jnp.ndarray:
    h0 = _field(batch, "h0", default=None)
    if h0 is not None:
        return jnp.asarray(h0, dtype=jnp.float32)
    return model.init_hidden(tuple(feat_seq.shape[1:-1]), dtype=jnp.float32)


def predictor_loss(params: Mapping[str, Any], batch: Mapping[str, Any]) -> tuple[jnp.ndarray, dict[str, Any]]:
    """Return predictor MSE loss and diagnostics.

    Batch fields are time-major. Accepted aliases are:
    ``features``/``feat_seq``, ``actions``/``act_seq``, ``rewards``/``rew_seq``,
    ``resets``/``reset_seq``, and optional ``episode_end_mask``/``dones``.
    The optional ``model`` field supplies the exact ``ReturnPredictor``
    instance; otherwise a default predictor config is constructed from batch
    metadata.
    """

    model = _model_from_batch(batch)
    feat_seq = jax.lax.stop_gradient(
        jnp.asarray(_field(batch, "features", "feat_seq"), dtype=jnp.float32)
    )
    act_seq = jnp.asarray(_field(batch, "actions", "act_seq"), dtype=jnp.int32)
    rew_seq = jnp.asarray(_field(batch, "rewards", "rew_seq"), dtype=jnp.float32)
    reset_seq = jnp.asarray(
        _field(batch, "resets", "reset_seq", default=jnp.zeros_like(rew_seq, dtype=bool)),
        dtype=bool,
    )
    episode_end_mask = _field(
        batch,
        "episode_end_mask",
        "dones",
        "terminals",
        default=None,
    )
    gamma = float(_field(batch, "gamma", default=0.99))
    h0 = _h0_from_batch(model, batch, feat_seq)

    F_seq, Phi_seq = model.unroll(params, feat_seq, act_seq, rew_seq, reset_seq, h0)
    G0, remaining = discounted_returns(rew_seq, gamma, episode_end_mask)
    G0_for_F = jnp.concatenate([G0, G0[-1:]], axis=0)

    err_F = F_seq - G0_for_F
    err_Phi = Phi_seq[:-1] - remaining
    mse_F = jnp.mean(jnp.square(err_F))
    mse_Phi = jnp.mean(jnp.square(err_Phi))
    loss = mse_F + mse_Phi
    aux = {
        "mse_F": mse_F,
        "mse_Phi": mse_Phi,
        "F_seq": F_seq,
        "Phi_seq": Phi_seq,
        "G0": G0,
        "G0_for_F": G0_for_F,
        "remaining": remaining,
    }
    return loss, aux


def make_predictor_optimizer(cfg: Any = None) -> optax.GradientTransformation:
    """Create the predictor's own Adam optimizer, separate from PPO/policy."""

    learning_rate = float(_cfg_value(cfg, "predictor_lr", "learning_rate", "lr", default=1e-3))
    return optax.adam(learning_rate=learning_rate)


def train_step(
    params: Mapping[str, Any],
    opt_state: optax.OptState,
    batch: Mapping[str, Any],
    tx: optax.GradientTransformation,
) -> tuple[Mapping[str, Any], optax.OptState, dict[str, Any]]:
    """Apply one predictor-only Adam step."""

    def loss_fn(p):
        return predictor_loss(p, batch)

    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, opt_state = tx.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    aux = dict(aux)
    aux["loss"] = loss
    return params, opt_state, aux


def validate(params: Mapping[str, Any], val_batch: Mapping[str, Any]) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return held-out ``F`` MSE and constant-return baseline ``Var(G0)``."""

    _, aux = predictor_loss(params, val_batch)
    G0 = jax.lax.stop_gradient(jnp.asarray(aux["G0"], dtype=jnp.float32))
    return aux["mse_F"], jnp.var(G0)


__all__ = [
    "ReturnPredictor",
    "discounted_returns",
    "make_predictor_optimizer",
    "predictor_loss",
    "train_step",
    "validate",
]
