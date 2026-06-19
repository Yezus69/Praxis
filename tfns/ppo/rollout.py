"""On-policy recurrent PPO rollout and sequence batching utilities.

``ppo_mask`` and ``reset_mask`` intentionally have different meanings:

- ``ppo_mask[t]`` is the PPO terminal/life-loss mask for transition ``t`` and
  is used by GAE only.
- ``reset_mask[t]`` is the true recurrent reset flag fed with ``obs[t]`` and
  is used by recurrent unrolls only.

The environment adapter returns the next true reset flag after stepping; this
module shifts that flag into the next rollout input/carry.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import partial
from typing import Any

from flax import struct
import jax
import jax.numpy as jnp
import numpy as np


@struct.dataclass
class RolloutBatch:
    """Time-major on-policy rollout arrays.

    Required fields match the public PPO data contract. ``hidden_after`` is an
    optional current-policy cache used only to derive on-policy sequence chunk
    starts; replay code must reconstruct hidden state from raw burn-in instead.
    ``last_obs``/``last_reset`` support final-step auxiliary targets and GAE
    bootstrapping.
    """

    obs: Any
    prev_action: Any
    prev_reward_clipped: Any
    action: Any
    logprob: Any
    value: Any
    reward: Any
    ppo_mask: Any
    reset_mask: Any
    last_value: Any
    last_ppo_done: Any
    h0: Any
    hidden_after: Any = None
    last_obs: Any = None
    last_reset: Any = None


@struct.dataclass
class RolloutCarry:
    """State needed to continue a vectorized rollout block."""

    hidden: Any
    prev_action: Any
    prev_reward_clipped: Any
    prev_reset: Any


@struct.dataclass
class SequenceMinibatch:
    """Sequence-major recurrent PPO minibatch.

    Per-step arrays have shape ``(B, L, ...)`` and ``h0_chunk`` has shape
    ``(B, H)``. ``valid_mask`` is all true for current fixed-size chunks but is
    consumed by losses so future padded sequence batches have the same surface.
    """

    obs: Any
    prev_action: Any
    prev_reward_clipped: Any
    action: Any
    old_logprob: Any
    value: Any
    reward: Any
    adv: Any
    ret: Any
    ppo_mask: Any
    reset_mask: Any
    h0_chunk: Any
    valid_mask: Any
    next_obs: Any = None
    next_obs_mask: Any = None
    true_terminal: Any = None
    seq_id: Any = None
    chunk_start: Any = None
    env_index: Any = None


def categorical_log_prob(logits: Any, actions: Any) -> jnp.ndarray:
    """Return categorical log-probabilities for integer ``actions``."""

    log_probs = jax.nn.log_softmax(jnp.asarray(logits, dtype=jnp.float32), axis=-1)
    actions = jnp.asarray(actions, dtype=jnp.int32)
    return jnp.take_along_axis(log_probs, actions[..., None], axis=-1)[..., 0]


def categorical_entropy(logits: Any) -> jnp.ndarray:
    """Return categorical entropy for ``logits``."""

    log_probs = jax.nn.log_softmax(jnp.asarray(logits, dtype=jnp.float32), axis=-1)
    probs = jnp.exp(log_probs)
    return -jnp.sum(probs * log_probs, axis=-1)


@partial(jax.jit, static_argnames=("agent",))
def _policy_step(agent, params, obs, prev_action, prev_reward_clipped, reset, hidden, rng):
    out = agent.apply(
        {"params": params},
        obs,
        prev_action,
        prev_reward_clipped,
        reset,
        hidden,
    )
    action = jax.random.categorical(rng, out.logits, axis=-1).astype(jnp.int32)
    logprob = categorical_log_prob(out.logits, action)
    return action, logprob, out.value.astype(jnp.float32), out.h_next.astype(jnp.float32)


@partial(jax.jit, static_argnames=("agent",))
def _value_only(agent, params, obs, prev_action, prev_reward_clipped, reset, hidden):
    out = agent.apply(
        {"params": params},
        obs,
        prev_action,
        prev_reward_clipped,
        reset,
        hidden,
    )
    return out.value.astype(jnp.float32)


def _current_obs(env_step) -> Any:
    for name in ("obs", "current_obs"):
        if hasattr(env_step, name):
            value = getattr(env_step, name)
            return value() if callable(value) else value
    if hasattr(env_step, "get_obs"):
        return env_step.get_obs()
    raise ValueError("env_step must expose current observation via .obs, .current_obs, or .get_obs()")


def _set_current_obs(env_step, obs: Any) -> None:
    for name in ("obs", "current_obs"):
        if hasattr(env_step, name):
            try:
                setattr(env_step, name, obs)
            except Exception:
                pass
            return


def collect_rollout(env_step, agent, params, carry: RolloutCarry, rollout_len: int, rng):
    """Collect an on-policy recurrent rollout with a Python environment step.

    ``env_step(action)`` must return ``(next_obs, reward_clipped, ppo_done,
    reset, extra)``. The returned ``reset`` is a true recurrent reset for the
    *next* observation, so the recorded ``reset_mask[t]`` is the carry's
    incoming reset flag for ``obs[t]``. ``ppo_done`` is recorded at transition
    ``t`` and is used only by :func:`compute_gae`.
    """

    rollout_len = int(rollout_len)
    if rollout_len <= 0:
        raise ValueError("rollout_len must be positive")

    obs = jnp.asarray(_current_obs(env_step))
    hidden = jnp.asarray(carry.hidden, dtype=jnp.float32)
    prev_action = jnp.asarray(carry.prev_action, dtype=jnp.int32)
    prev_reward = jnp.asarray(carry.prev_reward_clipped, dtype=jnp.float32)
    prev_reset = jnp.asarray(carry.prev_reset, dtype=bool)
    h0 = hidden

    obs_rows = []
    prev_action_rows = []
    prev_reward_rows = []
    action_rows = []
    logprob_rows = []
    value_rows = []
    reward_rows = []
    ppo_mask_rows = []
    reset_mask_rows = []
    hidden_after_rows = []
    extra_rows = []
    last_ppo_done = jnp.zeros(prev_action.shape, dtype=bool)

    for _ in range(rollout_len):
        rng, action_key = jax.random.split(rng)
        action, logprob, value, h_next = _policy_step(
            agent,
            params,
            obs,
            prev_action,
            prev_reward,
            prev_reset,
            hidden,
            action_key,
        )
        next_obs, reward_clipped, ppo_done, reset, extra = env_step(
            np.asarray(jax.device_get(action), dtype=np.int32)
        )

        obs_rows.append(obs)
        prev_action_rows.append(prev_action)
        prev_reward_rows.append(prev_reward)
        action_rows.append(action)
        logprob_rows.append(logprob)
        value_rows.append(value)
        reward_rows.append(jnp.asarray(reward_clipped, dtype=jnp.float32))
        ppo_mask_rows.append(jnp.asarray(ppo_done, dtype=bool))
        reset_mask_rows.append(prev_reset)
        hidden_after_rows.append(h_next)
        extra_rows.append(extra)

        obs = jnp.asarray(next_obs)
        _set_current_obs(env_step, next_obs)
        hidden = h_next
        prev_action = action
        prev_reward = jnp.asarray(reward_clipped, dtype=jnp.float32)
        prev_reset = jnp.asarray(reset, dtype=bool)
        last_ppo_done = jnp.asarray(ppo_done, dtype=bool)

    last_value = _value_only(agent, params, obs, prev_action, prev_reward, prev_reset, hidden)
    batch = RolloutBatch(
        obs=jnp.stack(obs_rows, axis=0).astype(jnp.uint8),
        prev_action=jnp.stack(prev_action_rows, axis=0).astype(jnp.int32),
        prev_reward_clipped=jnp.stack(prev_reward_rows, axis=0).astype(jnp.float32),
        action=jnp.stack(action_rows, axis=0).astype(jnp.int32),
        logprob=jnp.stack(logprob_rows, axis=0).astype(jnp.float32),
        value=jnp.stack(value_rows, axis=0).astype(jnp.float32),
        reward=jnp.stack(reward_rows, axis=0).astype(jnp.float32),
        ppo_mask=jnp.stack(ppo_mask_rows, axis=0).astype(bool),
        reset_mask=jnp.stack(reset_mask_rows, axis=0).astype(bool),
        last_value=last_value.astype(jnp.float32),
        last_ppo_done=last_ppo_done.astype(bool),
        h0=h0.astype(jnp.float32),
        hidden_after=jnp.stack(hidden_after_rows, axis=0).astype(jnp.float32),
        last_obs=obs.astype(jnp.uint8),
        last_reset=prev_reset.astype(bool),
    )
    new_carry = RolloutCarry(
        hidden=hidden.astype(jnp.float32),
        prev_action=prev_action.astype(jnp.int32),
        prev_reward_clipped=prev_reward.astype(jnp.float32),
        prev_reset=prev_reset.astype(bool),
    )
    return batch, new_carry, {"extras": extra_rows, "rng": rng}


@jax.jit
def compute_gae(
    reward: Any,
    value: Any,
    ppo_mask: Any,
    last_value: Any,
    last_ppo_done: Any,
    gamma: float,
    gae_lambda: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute truncated GAE using PPO terminals, never recurrent resets.

    ``ppo_mask[t]`` cuts both the value bootstrap and the advantage recursion
    for transition ``t``. ``reset_mask`` is deliberately absent because true
    recurrent resets and Atari episodic-life PPO terminals are distinct.
    """

    reward = jnp.asarray(reward, dtype=jnp.float32)
    value = jnp.asarray(value, dtype=jnp.float32)
    done = jnp.asarray(ppo_mask, dtype=bool)
    last_value = jnp.asarray(last_value, dtype=jnp.float32)
    last_done = jnp.asarray(last_ppo_done, dtype=bool)
    gamma = jnp.asarray(gamma, dtype=jnp.float32)
    gae_lambda = jnp.asarray(gae_lambda, dtype=jnp.float32)
    done = done.at[-1].set(last_done)

    def step(carry, inputs):
        gae_acc, next_value = carry
        reward_t, value_t, done_t = inputs
        nonterminal = 1.0 - done_t.astype(jnp.float32)
        delta = reward_t + gamma * next_value * nonterminal - value_t
        adv_t = delta + gamma * gae_lambda * nonterminal * gae_acc
        return (adv_t, value_t), adv_t

    init = (jnp.zeros_like(last_value, dtype=jnp.float32), last_value)
    _, adv_rev = jax.lax.scan(step, init, (reward[::-1], value[::-1], done[::-1]))
    adv = adv_rev[::-1].astype(jnp.float32)
    ret = (adv + value).astype(jnp.float32)
    return jax.lax.stop_gradient(adv), jax.lax.stop_gradient(ret)


def reconstruct_hidden(agent, params, obs, prev_action, prev_reward, reset, h_init=None):
    """Reconstruct recurrent state by unrolling raw burn-in inputs.

    This replay/burn-in helper trusts only ``h_init`` (zeros by default) and
    ignores any stored hidden trajectory. Inputs are time-major.
    """

    obs = jnp.asarray(obs)
    if h_init is None:
        h_init = agent.init_hidden(int(obs.shape[1]), dtype=jnp.float32)
    _, h_final = agent.unroll(
        params,
        obs,
        jnp.asarray(prev_action, dtype=jnp.int32),
        jnp.asarray(prev_reward, dtype=jnp.float32),
        jnp.asarray(reset, dtype=bool),
        jnp.asarray(h_init, dtype=jnp.float32),
    )
    return h_final.astype(jnp.float32)


def _time_chunks(x: Any, num_chunks: int, seq_chunk: int) -> jnp.ndarray:
    x = jnp.asarray(x)
    reshaped = x.reshape((num_chunks, seq_chunk, x.shape[1]) + tuple(x.shape[2:]))
    seq_major = jnp.swapaxes(reshaped, 1, 2)
    return seq_major.reshape((num_chunks * x.shape[1], seq_chunk) + tuple(x.shape[2:]))


def _hidden_chunk_starts(
    rollout: RolloutBatch,
    num_chunks: int,
    seq_chunk: int,
    *,
    agent=None,
    params=None,
) -> jnp.ndarray:
    starts = jnp.arange(num_chunks, dtype=jnp.int32) * int(seq_chunk)
    hidden_after = rollout.hidden_after
    if agent is not None and params is not None:
        outputs, _ = agent.unroll(
            params,
            jnp.asarray(rollout.obs),
            jnp.asarray(rollout.prev_action, dtype=jnp.int32),
            jnp.asarray(rollout.prev_reward_clipped, dtype=jnp.float32),
            jnp.asarray(rollout.reset_mask, dtype=bool),
            jnp.asarray(rollout.h0, dtype=jnp.float32),
        )
        hidden_after = outputs.h_next
    if hidden_after is None:
        if num_chunks != 1:
            raise ValueError("rollout.hidden_after is required for multiple recurrent chunks")
        return jnp.asarray(rollout.h0, dtype=jnp.float32)[None, ...]
    boundary_by_t = jnp.concatenate(
        [jnp.asarray(rollout.h0, dtype=jnp.float32)[None, ...], hidden_after[:-1]],
        axis=0,
    )
    return boundary_by_t[starts]


def _take(x: Any, idx: jnp.ndarray) -> Any:
    if x is None:
        return None
    return jnp.take(jnp.asarray(x), idx, axis=0)


def make_sequence_minibatches(
    rollout: RolloutBatch,
    adv: Any,
    ret: Any,
    seq_chunk: int,
    rng: Any,
    *,
    minibatch_size: int | None = None,
    agent=None,
    params=None,
) -> Iterator[SequenceMinibatch]:
    """Yield shuffled contiguous recurrent sequence minibatches.

    The time axis is split into fixed contiguous chunks; each ``(chunk, env)``
    pair remains intact as one sequence. If ``agent`` and ``params`` are
    supplied, chunk hidden states are recomputed once from ``rollout.h0`` over
    the full rollout. Otherwise they come from ``rollout.h0`` and the optional
    ``hidden_after`` cache collected under the same policy. Replay callers
    should use :func:`reconstruct_hidden` instead of trusting stored hidden.
    """

    seq_chunk = int(seq_chunk)
    if seq_chunk <= 0:
        raise ValueError("seq_chunk must be positive")
    T = int(rollout.action.shape[0])
    N = int(rollout.action.shape[1])
    if T % seq_chunk != 0:
        raise ValueError("rollout length must be divisible by seq_chunk")
    num_chunks = T // seq_chunk
    num_seq = num_chunks * N
    if minibatch_size is None:
        minibatch_size = num_seq
    minibatch_size = int(minibatch_size)
    if minibatch_size <= 0:
        raise ValueError("minibatch_size must be positive")

    h0_by_chunk = _hidden_chunk_starts(
        rollout,
        num_chunks,
        seq_chunk,
        agent=agent,
        params=params,
    )
    h0_seq = h0_by_chunk.reshape((num_seq,) + tuple(h0_by_chunk.shape[2:]))

    if rollout.last_obs is None:
        next_obs_time = jnp.concatenate(
            [rollout.obs[1:], jnp.zeros_like(rollout.obs[:1])],
            axis=0,
        )
        next_obs_mask_time = jnp.concatenate(
            [
                jnp.ones((max(T - 1, 0), N), dtype=bool),
                jnp.zeros((1, N), dtype=bool),
            ],
            axis=0,
        )
    else:
        next_obs_time = jnp.concatenate([rollout.obs[1:], rollout.last_obs[None, ...]], axis=0)
        next_obs_mask_time = jnp.ones((T, N), dtype=bool)

    if rollout.last_reset is None:
        true_terminal_time = jnp.asarray(rollout.reset_mask, dtype=bool)
    else:
        true_terminal_time = jnp.concatenate(
            [jnp.asarray(rollout.reset_mask[1:], dtype=bool), rollout.last_reset[None, ...]],
            axis=0,
        )

    fields = {
        "obs": _time_chunks(rollout.obs, num_chunks, seq_chunk),
        "prev_action": _time_chunks(rollout.prev_action, num_chunks, seq_chunk),
        "prev_reward_clipped": _time_chunks(rollout.prev_reward_clipped, num_chunks, seq_chunk),
        "action": _time_chunks(rollout.action, num_chunks, seq_chunk),
        "old_logprob": _time_chunks(rollout.logprob, num_chunks, seq_chunk),
        "value": _time_chunks(rollout.value, num_chunks, seq_chunk),
        "reward": _time_chunks(rollout.reward, num_chunks, seq_chunk),
        "adv": _time_chunks(adv, num_chunks, seq_chunk),
        "ret": _time_chunks(ret, num_chunks, seq_chunk),
        "ppo_mask": _time_chunks(rollout.ppo_mask, num_chunks, seq_chunk),
        "reset_mask": _time_chunks(rollout.reset_mask, num_chunks, seq_chunk),
        "next_obs": _time_chunks(next_obs_time, num_chunks, seq_chunk),
        "next_obs_mask": _time_chunks(next_obs_mask_time, num_chunks, seq_chunk),
        "true_terminal": _time_chunks(true_terminal_time, num_chunks, seq_chunk),
    }
    valid_mask = jnp.ones((num_seq, seq_chunk), dtype=bool)
    chunk_start = jnp.repeat(jnp.arange(num_chunks, dtype=jnp.int32) * int(seq_chunk), N)
    env_index = jnp.tile(jnp.arange(N, dtype=jnp.int32), num_chunks)
    seq_id = jnp.arange(num_seq, dtype=jnp.int32)

    permutation = np.asarray(jax.device_get(jax.random.permutation(rng, num_seq)), dtype=np.int64)
    for start in range(0, num_seq, minibatch_size):
        idx = jnp.asarray(permutation[start : start + minibatch_size], dtype=jnp.int32)
        yield SequenceMinibatch(
            obs=_take(fields["obs"], idx),
            prev_action=_take(fields["prev_action"], idx),
            prev_reward_clipped=_take(fields["prev_reward_clipped"], idx),
            action=_take(fields["action"], idx),
            old_logprob=_take(fields["old_logprob"], idx),
            value=_take(fields["value"], idx),
            reward=_take(fields["reward"], idx),
            adv=_take(fields["adv"], idx),
            ret=_take(fields["ret"], idx),
            ppo_mask=_take(fields["ppo_mask"], idx),
            reset_mask=_take(fields["reset_mask"], idx),
            h0_chunk=_take(h0_seq, idx),
            valid_mask=_take(valid_mask, idx),
            next_obs=_take(fields["next_obs"], idx),
            next_obs_mask=_take(fields["next_obs_mask"], idx),
            true_terminal=_take(fields["true_terminal"], idx),
            seq_id=_take(seq_id, idx),
            chunk_start=_take(chunk_start, idx),
            env_index=_take(env_index, idx),
        )


__all__ = [
    "RolloutBatch",
    "RolloutCarry",
    "SequenceMinibatch",
    "categorical_entropy",
    "categorical_log_prob",
    "collect_rollout",
    "compute_gae",
    "make_sequence_minibatches",
    "reconstruct_hidden",
]
