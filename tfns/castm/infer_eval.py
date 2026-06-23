"""Inferred-address (task-free) routing evaluation — Stage D (spec 9, 21.3, 24).

After the oracle curriculum trains and commits each game to its canonical
address, this module:

1. builds a content-prototype index per discovered context from anchor frames
   (content queries from the shared W0-only encoder; no game/task id);
2. measures router top-1 accuracy on held-out frames (does a frame from game g
   route to the canonical address learned for game g?);
3. runs inferred-address evaluation: at every step the address is chosen by the
   content router (not forced), and the policy uses that address via sparse
   top-1 gather. Reports the inferred score to compare against the oracle score
   (the most important comparison, spec 24).

The address is an internal content-retrieved coordinate; the game identity is
used only by the harness to label held-out frames for the accuracy metric.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import baseline_ppo as bp
from tfns.castm import address as addr
from tfns.castm import ff


def _content_query_fn(cfg_ff):
    @jax.jit
    def q_of(banks, obs, k):
        return ff.content_query(banks, cfg_ff, obs, k)
    return q_of


def collect_anchor_queries(cfg_ff, banks, book, game, ctx_id, *, num_envs, n_frames, seed, fire_reset):
    """Roll out game ``game`` at its oracle address; return normalized content queries."""

    q_of = _content_query_fn(cfg_ff)
    k = addr.code(book, ctx_id)

    @jax.jit
    def act(banks, obs, k, fire, rng):
        rng, key = jax.random.split(rng)
        logits, _ = ff.forward(banks, None, cfg_ff, obs, k, ctx_id=ctx_id, sparse=True)
        a = jax.random.categorical(key, logits, axis=-1).astype(jnp.int32)
        return jnp.where(fire, jnp.full_like(a, bp.FIRE_ACTION), a), rng

    env = bp.make_env(game, num_envs, seed, training=False)
    obs, _ = bp.reset_result(env.reset())
    obs = bp.nhwc_uint8(obs)
    fire = np.full((num_envs,), bool(fire_reset), np.bool_)
    rng = jax.random.PRNGKey(seed + 3)
    queries = []
    collected = 0
    while collected < n_frames:
        q = np.asarray(jax.device_get(q_of(banks, jnp.asarray(obs), k)))
        queries.append(q)
        collected += q.shape[0]
        a, rng = act(banks, jnp.asarray(obs), k, jnp.asarray(fire), rng)
        obs, reward, term, trunc, _ = env.step(np.asarray(jax.device_get(a), np.int32))
        obs = bp.nhwc_uint8(obs)
        done = np.logical_or(bp.vec(term, np.bool_, num_envs, "t"), bp.vec(trunc, np.bool_, num_envs, "tr"))
        fire = np.where(done, bool(fire_reset), False)
    close = getattr(env, "close", None)
    if callable(close):
        close()
    return np.concatenate(queries, axis=0)[:n_frames]


def build_prototypes(queries, m_p=8):
    """Farthest-point prototype selection from normalized queries (spec 9.2)."""

    q = np.asarray(queries, dtype=np.float32)
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-6)
    n = q.shape[0]
    chosen = [0]
    while len(chosen) < min(m_p, n):
        sel = q[chosen]
        nearest = (q @ sel.T).max(axis=1)
        nearest[chosen] = 2.0
        chosen.append(int(np.argmin(nearest)))
    return q[chosen]  # (<=m_p, d_q)


def route_batch(q, prototypes_stack, ctx_ids):
    """Top-1 route each query in ``q`` (B,d_q) to a context via nearest prototype.

    ``prototypes_stack``: list of (M_pi, d_q) arrays; ``ctx_ids`` parallel list.
    Returns routed ctx id per row (B,) and the best similarity (B,).
    """

    best_sim = np.full((q.shape[0],), -2.0, dtype=np.float32)
    best_ctx = np.full((q.shape[0],), -1, dtype=np.int32)
    for protos, c in zip(prototypes_stack, ctx_ids):
        sim = (q @ protos.T).max(axis=1)  # (B,)
        upd = sim > best_sim
        best_sim = np.where(upd, sim, best_sim)
        best_ctx = np.where(upd, c, best_ctx)
    return best_ctx, best_sim


def routing_accuracy(cfg_ff, banks, book, games, ctx_ids, prototypes, *, num_envs, n_frames, seed, fire_reset):
    """Held-out router top-1 accuracy per game (spec 21.3)."""

    proto_stack = [prototypes[c] for c in ctx_ids]
    per_game = {}
    confusion = {}
    total_correct = 0
    total = 0
    for gi, game in enumerate(games):
        true_ctx = ctx_ids[gi]
        q = collect_anchor_queries(cfg_ff, banks, book, game, true_ctx,
                                   num_envs=num_envs, n_frames=n_frames, seed=seed + 5000 + gi, fire_reset=fire_reset)
        routed, _ = route_batch(q, proto_stack, ctx_ids)
        correct = int(np.sum(routed == true_ctx))
        per_game[game] = correct / max(len(routed), 1)
        # confusion row
        row = {int(c): int(np.sum(routed == c)) for c in ctx_ids}
        confusion[game] = row
        total_correct += correct
        total += len(routed)
    return {"per_game": per_game, "overall": total_correct / max(total, 1), "confusion": confusion}


def inferred_eval_game(cfg_ff, banks, book, game, true_ctx, prototypes, ctx_ids, *,
                       num_envs, n_episodes, seed, fire_reset, ema=0.8, max_steps=1_000_000):
    """Run one game with INFERRED addressing; return score + routing accuracy."""

    proto_stack = [prototypes[c] for c in ctx_ids]
    q_of = _content_query_fn(cfg_ff)

    fwd = {}
    for c in ctx_ids:
        kc = addr.code(book, c)

        @jax.jit
        def f(banks, obs, kc=kc, c=c):
            logits, _ = ff.forward(banks, None, cfg_ff, obs, kc, ctx_id=int(c), sparse=True)
            return logits
        fwd[c] = f

    env = bp.make_env(game, num_envs, seed, training=False)
    obs, _ = bp.reset_result(env.reset())
    obs = bp.nhwc_uint8(obs)
    running = np.zeros((num_envs,), np.float32)
    fire = np.full((num_envs,), bool(fire_reset), np.bool_)
    completed = []
    steps = 0
    rng = jax.random.PRNGKey(seed + 11)
    # EMA of per-ctx similarity per env for stable routing (anti-chatter, spec 9.2).
    sim_ema = {int(c): np.full((num_envs,), -2.0, np.float32) for c in ctx_ids}
    route_correct = 0
    route_total = 0
    while len(completed) < n_episodes and steps < max_steps:
        q = np.asarray(jax.device_get(q_of(banks, jnp.asarray(obs), addr.code(book, true_ctx))))
        for c in ctx_ids:
            sim_c = (q @ prototypes[c].T).max(axis=1)
            sim_ema[c] = ema * sim_ema[c] + (1.0 - ema) * sim_c
        stacked = np.stack([sim_ema[c] for c in ctx_ids], axis=1)  # (B, C)
        routed = np.asarray(ctx_ids)[np.argmax(stacked, axis=1)]
        route_correct += int(np.sum(routed == true_ctx))
        route_total += num_envs
        # Apply policy per distinct routed ctx (<= num_contexts forward passes).
        actions = np.zeros((num_envs,), np.int32)
        rng, key = jax.random.split(rng)
        keys = jax.random.split(key, num_envs)
        for c in np.unique(routed):
            mask = routed == c
            logits = np.asarray(jax.device_get(fwd[int(c)](banks, jnp.asarray(obs))))
            # stochastic sample per masked env
            for idx in np.flatnonzero(mask):
                lg = logits[idx]
                a = int(jax.random.categorical(keys[idx], jnp.asarray(lg)))
                actions[idx] = a
        actions = np.where(fire, bp.FIRE_ACTION, actions).astype(np.int32)
        obs, reward, term, trunc, _ = env.step(actions)
        obs = bp.nhwc_uint8(obs)
        done = np.logical_or(bp.vec(term, np.bool_, num_envs, "t"), bp.vec(trunc, np.bool_, num_envs, "tr"))
        running += bp.vec(reward, np.float32, num_envs, "r")
        for idx in np.flatnonzero(done):
            if len(completed) < n_episodes:
                completed.append(float(running[idx]))
            running[idx] = 0.0
            for c in ctx_ids:
                sim_ema[c][idx] = -2.0
        fire = np.where(done, bool(fire_reset), False)
        steps += num_envs
    close = getattr(env, "close", None)
    if callable(close):
        close()
    arr = np.asarray(completed, np.float32)
    return {
        "mean": float(arr.mean()) if arr.size else float("nan"),
        "n": int(arr.size),
        "valid": bool(len(completed) >= n_episodes),
        "route_acc": route_correct / max(route_total, 1),
    }


__all__ = [
    "build_prototypes",
    "collect_anchor_queries",
    "inferred_eval_game",
    "route_batch",
    "routing_accuracy",
]
