"""Task-free context inference: prototype router (spec section 9, steps 10-11).

The router maps a normalized content query ``q_t`` (produced by a recurrent
context encoder from observation/action/reward/termination history) to a
canonical memory address, with no game/task identity input. It implements:

* the prototype index and posterior scoring (9.2),
* EMA score smoothing, minimum dwell, and hysteresis to prevent route
  chattering (9.2),
* unknown-context detection from confidence, nearest-prototype distance, and an
  elevated robust dynamics-error baseline persisting for a window (9.4),
* the route decision (15.1): KNOWN / NOVEL / UNCERTAIN.

The address book mutation (allocate a new canonical code) and the scratch
commit/reset happen host-side in the context-switch transaction
(:func:`tfns.castm.transaction`), not in this jittable per-step function.

State is batched over ``B`` independent environment streams. The prototype index
is global (shared across streams).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from flax import struct
import jax
import jax.numpy as jnp


# Route decision codes.
KNOWN = 0
UNCERTAIN = 1
NOVEL = 2

_NEG_INF = -1e30


@dataclass(frozen=True)
class RouterConfig:
    tau: float = 0.1                 # prototype-score temperature
    lambda_h: float = 0.5            # hysteresis bonus on previous posterior
    score_ema: float = 0.8           # EMA factor on scores (9.2)
    min_dwell: int = 8               # min consecutive steps before a switch (9.2)
    confidence_thresh: float = 0.6   # known-context posterior threshold (9.4.1)
    novelty_dist_thresh: float = 0.5 # nearest-prototype cosine-distance threshold (9.4.2)
    dyn_err_k: float = 3.0           # robust dynamics-error elevation factor (9.4.3)
    dyn_err_ema: float = 0.99        # EMA factor for the dynamics-error baseline
    novel_persist: int = 16          # H_novel persistence window (9.4.4)
    max_contexts: int = 64           # spec 11.3
    prototypes_per_context: int = 8  # M_p (11.3)
    eps: float = 1e-6


@struct.dataclass
class PrototypeIndex:
    """Global content-prototype index (spec 9.2)."""

    prototypes: jnp.ndarray   # (C, M_p, d_q) normalized prototypes
    count: jnp.ndarray        # (C,) int32 number of live prototypes per context
    used: jnp.ndarray         # (C,) bool whether the context is discovered

    @property
    def max_contexts(self) -> int:
        return int(self.prototypes.shape[0])

    @property
    def prototypes_per_context(self) -> int:
        return int(self.prototypes.shape[1])

    @property
    def d_q(self) -> int:
        return int(self.prototypes.shape[2])


@struct.dataclass
class RouterState:
    """Per-stream routing state (B streams)."""

    posterior: jnp.ndarray     # (B, C)
    ema_score: jnp.ndarray     # (B, C)
    locked_ctx: jnp.ndarray    # (B,) int32, -1 if none locked
    pending_ctx: jnp.ndarray   # (B,) int32 candidate awaiting dwell
    pending_streak: jnp.ndarray  # (B,) int32
    novelty_streak: jnp.ndarray  # (B,) int32
    dyn_err_mean: jnp.ndarray  # (B,) running robust mean of dynamics error
    dyn_err_dev: jnp.ndarray   # (B,) running mean abs deviation


def empty_prototype_index(max_contexts: int, prototypes_per_context: int, d_q: int) -> PrototypeIndex:
    C, M, D = int(max_contexts), int(prototypes_per_context), int(d_q)
    return PrototypeIndex(
        prototypes=jnp.zeros((C, M, D), dtype=jnp.float32),
        count=jnp.zeros((C,), dtype=jnp.int32),
        used=jnp.zeros((C,), dtype=bool),
    )


def init_router_state(batch: int, max_contexts: int) -> RouterState:
    B, C = int(batch), int(max_contexts)
    return RouterState(
        posterior=jnp.zeros((B, C), dtype=jnp.float32),
        ema_score=jnp.zeros((B, C), dtype=jnp.float32),
        locked_ctx=jnp.full((B,), -1, dtype=jnp.int32),
        pending_ctx=jnp.full((B,), -1, dtype=jnp.int32),
        pending_streak=jnp.zeros((B,), dtype=jnp.int32),
        novelty_streak=jnp.zeros((B,), dtype=jnp.int32),
        dyn_err_mean=jnp.zeros((B,), dtype=jnp.float32),
        dyn_err_dev=jnp.ones((B,), dtype=jnp.float32),
    )


def _normalize(v: jnp.ndarray, eps: float) -> jnp.ndarray:
    return v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + eps)


def prototype_similarities(index: PrototypeIndex, q: jnp.ndarray) -> jnp.ndarray:
    """Return ``(B, C)`` best prototype cosine similarity per context (q normalized)."""

    # (B,d),(C,M,d) -> (B,C,M)
    sims = jnp.einsum("bd,cmd->bcm", q, index.prototypes)
    # mask prototype slots beyond live count and unused contexts
    m_idx = jnp.arange(index.prototypes_per_context)[None, :]  # (1,M)
    live = (m_idx < index.count[:, None]).astype(jnp.float32)  # (C,M)
    live = live * index.used[:, None].astype(jnp.float32)
    sims = jnp.where(live[None] > 0.0, sims, _NEG_INF)
    return jnp.max(sims, axis=-1)  # (B,C)


def route_step(
    state: RouterState,
    index: PrototypeIndex,
    q: Any,
    dyn_err: Any,
    cfg: RouterConfig,
) -> tuple[RouterState, dict[str, jnp.ndarray]]:
    """One routing step over ``B`` streams (spec 15.1, jittable).

    ``q`` is ``(B, d_q)`` (will be re-normalized), ``dyn_err`` is ``(B,)`` the
    current global dynamics-prediction error. Returns the updated state and an
    info dict with ``decision`` (B,) codes, ``selected_ctx`` (B,), ``posterior``
    (B, C), ``max_post`` (B,), ``nearest_dist`` (B,).
    """

    q = _normalize(jnp.asarray(q, dtype=jnp.float32), cfg.eps)
    dyn_err = jnp.asarray(dyn_err, dtype=jnp.float32).reshape(-1)
    C = index.max_contexts
    used_row = index.used[None, :]  # (1,C)
    any_used = jnp.any(index.used)

    best_sim = prototype_similarities(index, q)  # (B,C)
    raw_score = best_sim / cfg.tau
    hyst = cfg.lambda_h * jnp.log(cfg.eps + state.posterior)
    score = raw_score + hyst
    score = jnp.where(used_row, score, _NEG_INF)

    # Instantaneous content-match posterior (no hysteresis, no EMA) drives the
    # novelty confidence check (9.4.1) so detection is not masked by routing
    # smoothing. The EMA posterior below drives routing/lock stability.
    inst_post = jax.nn.softmax(jnp.where(used_row, raw_score, _NEG_INF), axis=-1)
    max_inst_post = jnp.max(jnp.where(used_row, inst_post, 0.0), axis=-1)

    ema_score = cfg.score_ema * state.ema_score + (1.0 - cfg.score_ema) * score
    ema_score = jnp.where(used_row, ema_score, _NEG_INF)

    posterior = jax.nn.softmax(ema_score, axis=-1)
    posterior = jnp.where(used_row, posterior, 0.0)
    max_post = jnp.max(jnp.where(used_row, posterior, 0.0), axis=-1)  # (B,)
    cand = jnp.argmax(jnp.where(used_row, posterior, _NEG_INF), axis=-1).astype(jnp.int32)

    nearest_sim = jnp.max(jnp.where(used_row, best_sim, _NEG_INF), axis=-1)
    nearest_sim = jnp.where(any_used, nearest_sim, -1.0)
    nearest_dist = 1.0 - nearest_sim  # (B,)

    # Robust dynamics-error baseline (EMA mean + scaled mean-abs-dev).
    dem = cfg.dyn_err_ema * state.dyn_err_mean + (1.0 - cfg.dyn_err_ema) * dyn_err
    ded = cfg.dyn_err_ema * state.dyn_err_dev + (1.0 - cfg.dyn_err_ema) * jnp.abs(
        dyn_err - state.dyn_err_mean
    )
    elevated = dyn_err > (state.dyn_err_mean + cfg.dyn_err_k * state.dyn_err_dev)

    novelty_cond = (
        (max_inst_post < cfg.confidence_thresh)
        & (nearest_dist > cfg.novelty_dist_thresh)
        & elevated
    )
    novelty_streak = jnp.where(novelty_cond, state.novelty_streak + 1, 0)

    # Dwell / hysteresis: a candidate must persist min_dwell steps to switch.
    same_pending = cand == state.pending_ctx
    pending_streak = jnp.where(same_pending, state.pending_streak + 1, 1)
    pending_ctx = cand

    confident = max_post >= cfg.confidence_thresh
    dwell_ok = pending_streak >= cfg.min_dwell
    is_locked = state.locked_ctx >= 0
    keep_lock = is_locked & (cand == state.locked_ctx)

    # Lock transition (spec 9.5): lock only after the hysteresis dwell window;
    # an already-locked context is retained without re-dwelling.
    do_lock = confident & (dwell_ok | keep_lock)
    locked_ctx = jnp.where(do_lock, cand, state.locked_ctx)

    novel = (novelty_streak >= cfg.novel_persist) | (~any_used)
    decision = jnp.where(
        novel,
        NOVEL,
        jnp.where(do_lock & confident, KNOWN, UNCERTAIN),
    ).astype(jnp.int32)

    selected_ctx = jnp.where(decision == KNOWN, locked_ctx, state.locked_ctx).astype(jnp.int32)

    new_state = RouterState(
        posterior=posterior,
        ema_score=ema_score,
        locked_ctx=locked_ctx,
        pending_ctx=pending_ctx,
        pending_streak=pending_streak,
        novelty_streak=novelty_streak,
        dyn_err_mean=dem,
        dyn_err_dev=ded,
    )
    info = {
        "decision": decision,
        "selected_ctx": selected_ctx,
        "posterior": posterior,
        "max_post": max_post,
        "nearest_dist": nearest_dist,
        "novelty_streak": novelty_streak,
        "elevated": elevated.astype(jnp.int32),
    }
    return new_state, info


# --- Prototype index mutation (host-side, at switch/refresh events) -------------


def allocate_context(index: PrototypeIndex, q_window: Any, cfg: RouterConfig) -> tuple[PrototypeIndex, int]:
    """Discover a new context and seed prototypes from a recent query window (9.5.4).

    ``q_window`` is ``(N, d_q)`` of normalized queries from the anchor history.
    Prototypes are the first ``M_p`` farthest-point samples (greedy) for spread.
    Returns the updated index and the new context id (``-1`` if exhausted).
    """

    import numpy as np

    used = np.asarray(index.used)
    free = np.where(~used)[0]
    if free.size == 0:
        return index, -1
    ctx = int(free[0])
    qw = _normalize(jnp.asarray(q_window, dtype=jnp.float32), cfg.eps)
    qw_np = np.asarray(qw)
    M = index.prototypes_per_context
    n = qw_np.shape[0]
    if n == 0:
        return index, -1
    # Greedy farthest-point prototype selection for coverage.
    chosen = [0]
    while len(chosen) < min(M, n):
        sel = qw_np[chosen]                       # (k,d)
        sims = qw_np @ sel.T                       # (n,k)
        nearest = sims.max(axis=1)                 # (n,)
        nearest[chosen] = 2.0
        chosen.append(int(np.argmin(nearest)))
    protos = np.zeros((M, index.d_q), dtype=np.float32)
    k = len(chosen)
    protos[:k] = qw_np[chosen]
    prototypes = index.prototypes.at[ctx].set(jnp.asarray(protos))
    count = index.count.at[ctx].set(int(k))
    used_arr = index.used.at[ctx].set(True)
    return index.replace(prototypes=prototypes, count=count, used=used_arr), ctx


def refresh_prototypes(index: PrototypeIndex, ctx: int, q_window: Any, cfg: RouterConfig) -> PrototypeIndex:
    """Recompute a context's prototypes from re-encoded anchors (spec 10, query drift)."""

    idx2, new_ctx = allocate_context(index.replace(used=index.used.at[int(ctx)].set(False)), q_window, cfg)
    # allocate_context picks the first free slot; force it back to `ctx`.
    if new_ctx == int(ctx):
        return idx2
    # Move the freshly built prototypes into the original ctx slot.
    prototypes = index.prototypes.at[int(ctx)].set(idx2.prototypes[new_ctx])
    count = index.count.at[int(ctx)].set(idx2.count[new_ctx])
    used = index.used.at[int(ctx)].set(True)
    # leave new_ctx unused
    return index.replace(prototypes=prototypes, count=count, used=used)


def num_contexts(index: PrototypeIndex) -> int:
    import numpy as np

    return int(np.sum(np.asarray(index.used)))


__all__ = [
    "KNOWN",
    "NOVEL",
    "UNCERTAIN",
    "PrototypeIndex",
    "RouterConfig",
    "RouterState",
    "allocate_context",
    "empty_prototype_index",
    "init_router_state",
    "num_contexts",
    "prototype_similarities",
    "refresh_prototypes",
    "route_step",
]
