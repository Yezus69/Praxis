"""CASTM feed-forward continual PPO trainer (Atari ladder Stages C-F).

Trains a sequence of Atari games with one live addressed network:
- game 0 is learned into the shared weights W0 (normal PPO, spec 8.1);
- W0 is then frozen and each later game is learned through the LoRA scratchpad and
  committed to its canonical address dual, so prior games' decoded weights are
  unchanged (exact retention under sparse top-1 gather).

Addressing is ORACLE here: the harness forces the correct canonical address per
game (a diagnostic, not a task-free result — spec 21.2). The address is never
fed to the policy through any other path; routing is isolated for Stage D.

After every game, all games seen so far are evaluated (sparse gather at each
game's address) to build the retention matrix. Results persist to JSON.

Run (one process per GPU):
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
      python -m tfns.castm.train_castm --games Alien-v5 Defender-v5 \
        --steps-per-game 2000000 --out-dir castm_runs/oracle/seedA --seed 1
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

import baseline_ppo as bp
from tfns.castm import address as addr
from tfns.castm import ff
from tfns.castm import transaction as tx


class Batch(NamedTuple):
    obs: jnp.ndarray
    actions: jnp.ndarray
    logprobs: jnp.ndarray
    advantages: jnp.ndarray
    returns: jnp.ndarray
    values: jnp.ndarray


@dataclass(frozen=True)
class TrainConfig:
    games: tuple[str, ...] = ("Alien-v5", "Defender-v5")
    steps_per_game: int = 2_000_000
    num_envs: int = 32
    num_steps: int = 128
    seed: int = 1
    eval_episodes: int = 20
    eval_every_updates: int = 25
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_coef: float = 0.1
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr_shared: float = 2.5e-4
    lr_scratch: float = 1.0e-3
    out_dir: str = "castm_runs/oracle/run"
    fire_reset: bool = True
    inferred_eval: bool = True
    anchor_frames: int = 2048
    proto_per_ctx: int = 8
    scratch_mult: float = 1.0
    naive: bool = False


# --- PPO loss (feed-forward) ---------------------------------------------------


def _ppo_terms(logits, values, batch: Batch, clip, vf, ent):
    new_logprobs = bp.log_prob(logits, batch.actions)
    logratio = new_logprobs - batch.logprobs
    ratio = jnp.exp(logratio)
    adv = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)
    pg = jnp.maximum(-adv * ratio, -adv * jnp.clip(ratio, 1.0 - clip, 1.0 + clip)).mean()
    v_unclipped = jnp.square(values - batch.returns)
    v_clipped = batch.values + jnp.clip(values - batch.values, -clip, clip)
    v_loss = 0.5 * jnp.maximum(v_unclipped, jnp.square(v_clipped - batch.returns)).mean()
    logp = jax.nn.log_softmax(logits, axis=-1)
    entropy = -(jnp.exp(logp) * logp).sum(axis=-1).mean()
    approx_kl = ((ratio - 1.0) - logratio).mean()
    loss = pg + vf * v_loss - ent * entropy
    return loss, jnp.asarray([loss, pg, v_loss, entropy, approx_kl])


def loss_shared(trainable, banks_tmpl, cfg_ff, k, batch, clip, vf, ent):
    banks = ff.apply_shared_trainable(banks_tmpl, trainable)
    logits, values = ff.forward(banks, None, cfg_ff, batch.obs, k)
    return _ppo_terms(logits, values, batch, clip, vf, ent)


def loss_scratch(scratch, banks_frozen, cfg_ff, k, batch, clip, vf, ent):
    logits, values = ff.forward(banks_frozen, scratch, cfg_ff, batch.obs, k)
    return _ppo_terms(logits, values, batch, clip, vf, ent)


def _make_update(loss_fn, cfg_ff, cfg: TrainConfig):
    tx_opt = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(cfg.lr_shared))

    @partial(jax.jit, static_argnames=("minibatch_size",))
    def update(params, frozen, k, batch, rng, lr, minibatch_size, opt_state):
        opt = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(lr))
        batch_size = cfg.num_minibatches * minibatch_size

        def epoch(carry, _):
            params, opt_state, rng = carry
            rng, pk = jax.random.split(rng)
            perm = jax.random.permutation(pk, batch_size)

            def shuf(x):
                return jnp.take(x, perm, axis=0).reshape((cfg.num_minibatches, minibatch_size) + x.shape[1:])

            def mb(carry, b):
                params, opt_state = carry
                (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                    params, frozen, cfg_ff, k, b, cfg.clip_coef, cfg.vf_coef, cfg.ent_coef
                )
                updates, opt_state = opt.update(grads, opt_state, params)
                return (optax.apply_updates(params, updates), opt_state), metrics

            (params, opt_state), metrics = jax.lax.scan(mb, (params, opt_state), jax.tree_util.tree_map(shuf, batch))
            return (params, opt_state, rng), metrics.mean(axis=0)

        (params, opt_state, rng), metrics = jax.lax.scan(epoch, (params, opt_state, rng), None, length=cfg.update_epochs)
        return params, opt_state, rng, metrics.mean(axis=0)

    return update


# --- jitted action sampling / greedy ------------------------------------------


def _make_act(cfg_ff):
    @jax.jit
    def sample(banks, scratch, obs, k, fire, rng):
        rng, key = jax.random.split(rng)
        logits, value = ff.forward(banks, scratch, cfg_ff, obs, k)
        a = jax.random.categorical(key, logits, axis=-1).astype(jnp.int32)
        a = jnp.where(fire, jnp.full_like(a, bp.FIRE_ACTION), a)
        return a, bp.log_prob(logits, a), value, rng

    @jax.jit
    def value_only(banks, scratch, obs, k):
        return ff.forward(banks, scratch, cfg_ff, obs, k)[1]

    @partial(jax.jit, static_argnames=("ctx_id",))
    def sample_sparse(banks, obs, k, ctx_id, fire, rng):
        # Stochastic eval (primary, spec 19) via sparse top-1 gather at address k.
        rng, key = jax.random.split(rng)
        logits, _ = ff.forward(banks, None, cfg_ff, obs, k, ctx_id=ctx_id, sparse=True)
        a = jax.random.categorical(key, logits, axis=-1).astype(jnp.int32)
        return jnp.where(fire, jnp.full_like(a, bp.FIRE_ACTION), a), rng

    return sample, value_only, sample_sparse


# --- evaluation ----------------------------------------------------------------


def evaluate_game(sample_fn, banks, game, k, ctx_id, *, num_envs, n_episodes, seed, fire_reset, max_steps=1_000_000):
    env = bp.make_env(game, num_envs, seed, training=False)
    obs, _ = bp.reset_result(env.reset())
    obs = bp.nhwc_uint8(obs)
    running = np.zeros((num_envs,), np.float32)
    fire = np.full((num_envs,), bool(fire_reset), np.bool_)
    completed: list[float] = []
    steps = 0
    rng = jax.random.PRNGKey(int(seed) + 7)
    while len(completed) < n_episodes and steps < max_steps:
        a, rng = sample_fn(banks, jnp.asarray(obs), k, ctx_id, jnp.asarray(fire), rng)
        obs, reward, term, trunc, _ = env.step(np.asarray(jax.device_get(a), np.int32))
        obs = bp.nhwc_uint8(obs)
        done = np.logical_or(bp.vec(term, np.bool_, num_envs, "t"), bp.vec(trunc, np.bool_, num_envs, "tr"))
        running += bp.vec(reward, np.float32, num_envs, "r")
        for idx in np.flatnonzero(done):
            if len(completed) < n_episodes:
                completed.append(float(running[idx]))
            running[idx] = 0.0
        fire = np.where(done, bool(fire_reset), False)
        steps += num_envs
    close = getattr(env, "close", None)
    if callable(close):
        close()
    arr = np.asarray(completed, np.float32)
    valid = len(completed) >= n_episodes
    return {
        "mean": float(arr.mean()) if arr.size else float("nan"),
        "std": float(arr.std()) if arr.size else 0.0,
        "n": int(arr.size),
        "valid": bool(valid),
        "returns": [float(x) for x in completed],
    }


def random_score(game, *, num_envs, n_episodes, seed, fire_reset, max_steps=1_000_000):
    env = bp.make_env(game, num_envs, seed, training=False)
    obs, _ = bp.reset_result(env.reset())
    rng = np.random.default_rng(seed)
    running = np.zeros((num_envs,), np.float32)
    fire = np.full((num_envs,), bool(fire_reset), np.bool_)
    completed: list[float] = []
    steps = 0
    while len(completed) < n_episodes and steps < max_steps:
        a = rng.integers(0, bp.ACT_DIM, size=num_envs, dtype=np.int32)
        a = np.where(fire, bp.FIRE_ACTION, a).astype(np.int32)
        obs, reward, term, trunc, _ = env.step(a)
        done = np.logical_or(bp.vec(term, np.bool_, num_envs, "t"), bp.vec(trunc, np.bool_, num_envs, "tr"))
        running += bp.vec(reward, np.float32, num_envs, "r")
        for idx in np.flatnonzero(done):
            if len(completed) < n_episodes:
                completed.append(float(running[idx]))
            running[idx] = 0.0
        fire = np.where(done, bool(fire_reset), False)
        steps += num_envs
    close = getattr(env, "close", None)
    if callable(close):
        close()
    arr = np.asarray(completed, np.float32)
    return float(arr.mean()) if arr.size else 0.0


# --- per-game training ---------------------------------------------------------


def train_one_game(cfg: TrainConfig, cfg_ff, banks, book, game, ctx_id, *, mode, rng, out_dir, log):
    """Train one game; returns (banks, best_eval_mean, learning_curve, committed)."""

    k = addr.code(book, ctx_id)
    num_envs, num_steps = cfg.num_envs, cfg.num_steps
    batch_size = num_envs * num_steps
    minibatch_size = batch_size // cfg.num_minibatches
    num_updates = max(1, cfg.steps_per_game // batch_size)

    sample, value_only, sample_sparse = _make_act(cfg_ff)

    if mode == "shared":
        params = ff.shared_trainable(banks)
        frozen = banks
        loss_fn = loss_shared
        lr0 = cfg.lr_shared
        scratch = None
    else:
        params = ff.init_scratch(jax.random.PRNGKey(cfg.seed * 7 + ctx_id), cfg_ff)
        frozen = banks
        loss_fn = loss_scratch
        lr0 = cfg.lr_scratch
        scratch = params
    update = _make_update(loss_fn, cfg_ff, cfg)
    opt = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(lr0))
    opt_state = opt.init(params)

    env = bp.make_env(game, num_envs, cfg.seed + ctx_id, training=True)
    obs, _ = bp.reset_result(env.reset())
    next_obs = bp.nhwc_uint8(obs)
    fire = np.full((num_envs,), bool(cfg.fire_reset), np.bool_)
    obs_b = np.zeros((num_steps, num_envs, 84, 84, 4), np.uint8)
    act_b = np.zeros((num_steps, num_envs), np.int32)
    lp_b = np.zeros((num_steps, num_envs), np.float32)
    rew_b = np.zeros((num_steps, num_envs), np.float32)
    done_b = np.zeros((num_steps, num_envs), np.float32)
    val_b = np.zeros((num_steps, num_envs), np.float32)

    def cur_banks():
        return ff.apply_shared_trainable(banks, params) if mode == "shared" else banks

    def cur_scratch():
        return None if mode == "shared" else params

    curve = []
    best = -1e30
    start = time.perf_counter()
    for update_i in range(1, num_updates + 1):
        banks_now = cur_banks()
        scratch_now = cur_scratch()
        for step in range(num_steps):
            obs_b[step] = next_obs
            a, lp, v, rng = sample(banks_now, scratch_now, jnp.asarray(next_obs), k, jnp.asarray(fire), rng)
            a_np = np.asarray(jax.device_get(a), np.int32)
            act_b[step] = a_np
            lp_b[step] = np.asarray(jax.device_get(lp), np.float32)
            val_b[step] = np.asarray(jax.device_get(v), np.float32)
            next_obs, reward, term, trunc, _ = env.step(a_np)
            next_obs = bp.nhwc_uint8(next_obs)
            done = np.logical_or(bp.vec(term, np.bool_, num_envs, "t"), bp.vec(trunc, np.bool_, num_envs, "tr"))
            rew_b[step] = bp.vec(reward, np.float32, num_envs, "r")
            done_b[step] = done.astype(np.float32)
            fire = np.where(done, bool(cfg.fire_reset), False)

        last_v = value_only(banks_now, scratch_now, jnp.asarray(next_obs), k)
        adv, returns = bp.compute_gae(jnp.asarray(rew_b), jnp.asarray(done_b), jnp.asarray(val_b), last_v, cfg.gamma, cfg.gae_lambda)
        batch = Batch(
            obs=jnp.asarray(obs_b).reshape((batch_size,) + obs_b.shape[2:]),
            actions=jnp.asarray(act_b).reshape((batch_size,)),
            logprobs=jnp.asarray(lp_b).reshape((batch_size,)),
            advantages=jnp.asarray(adv).reshape((batch_size,)),
            returns=jnp.asarray(returns).reshape((batch_size,)),
            values=jnp.asarray(val_b).reshape((batch_size,)),
        )
        lr = lr0 * (1.0 - (update_i - 1.0) / float(num_updates))
        params, opt_state, rng, metrics = update(params, frozen, k, batch, rng, lr, minibatch_size, opt_state)

        if update_i == 1 or update_i % cfg.eval_every_updates == 0 or update_i == num_updates:
            banks_eval = cur_banks()
            # During training, eval the live (uncommitted) policy via sparse gather
            # is not possible (scratch not committed); eval with the live forward.
            ev = _eval_live(cfg_ff, banks_eval, cur_scratch(), game, k, num_envs=min(num_envs, cfg.eval_episodes),
                            n_episodes=cfg.eval_episodes, seed=cfg.seed + 9000 + update_i, fire_reset=cfg.fire_reset)
            steps_done = update_i * batch_size
            sps = steps_done / max(time.perf_counter() - start, 1e-6)
            rec = {"game": game, "ctx": ctx_id, "update": update_i, "steps": steps_done,
                   "eval_mean": ev["mean"], "eval_valid": ev["valid"], "sps": float(sps),
                   "loss": float(metrics[0]), "entropy": float(metrics[3]), "approx_kl": float(metrics[4])}
            curve.append(rec)
            best = max(best, ev["mean"] if np.isfinite(ev["mean"]) else best)
            log(f"[{game} ctx{ctx_id} {mode}] upd={update_i}/{num_updates} steps={steps_done} "
                f"eval_mean={ev['mean']:.1f} best={best:.1f} sps={sps:.0f} ent={metrics[3]:.3f}")

    close = getattr(env, "close", None)
    if callable(close):
        close()

    # Fold the learned weights into the live network.
    committed = False
    if mode == "shared":
        banks = ff.apply_shared_trainable(banks, params)  # game 0 lives in W0 (frozen hereafter)
    else:
        banks, report = tx.commit_scratch_bank(banks, params, book, ctx_id,
                                               eps_write=1e-3, eps_intended=5e-2,
                                               energy=0.999, max_elem_tol=1e-2)
        committed = bool(report["accepted"])
        log(f"[{game} ctx{ctx_id}] commit accepted={committed} "
            f"max_ni={report.get('max_noninterference')} reason={report.get('reason')}")
    return banks, float(best), curve, committed


def _eval_live(cfg_ff, banks, scratch, game, k, *, num_envs, n_episodes, seed, fire_reset, max_steps=1_000_000):
    """Greedy eval of the *live* (possibly scratch-carrying) policy at address k."""

    @jax.jit
    def sample(banks, scratch, obs, k, fire, rng):
        rng, key = jax.random.split(rng)
        logits, _ = ff.forward(banks, scratch, cfg_ff, obs, k)
        a = jax.random.categorical(key, logits, axis=-1).astype(jnp.int32)
        return jnp.where(fire, jnp.full_like(a, bp.FIRE_ACTION), a), rng

    env = bp.make_env(game, num_envs, seed, training=False)
    obs, _ = bp.reset_result(env.reset())
    obs = bp.nhwc_uint8(obs)
    running = np.zeros((num_envs,), np.float32)
    fire = np.full((num_envs,), bool(fire_reset), np.bool_)
    completed: list[float] = []
    steps = 0
    rng = jax.random.PRNGKey(int(seed) + 7)
    while len(completed) < n_episodes and steps < max_steps:
        a, rng = sample(banks, scratch, jnp.asarray(obs), k, jnp.asarray(fire), rng)
        obs, reward, term, trunc, _ = env.step(np.asarray(jax.device_get(a), np.int32))
        obs = bp.nhwc_uint8(obs)
        done = np.logical_or(bp.vec(term, np.bool_, num_envs, "t"), bp.vec(trunc, np.bool_, num_envs, "tr"))
        running += bp.vec(reward, np.float32, num_envs, "r")
        for idx in np.flatnonzero(done):
            if len(completed) < n_episodes:
                completed.append(float(running[idx]))
            running[idx] = 0.0
        fire = np.where(done, bool(fire_reset), False)
        steps += num_envs
    close = getattr(env, "close", None)
    if callable(close):
        close()
    arr = np.asarray(completed, np.float32)
    return {"mean": float(arr.mean()) if arr.size else float("nan"), "n": int(arr.size),
            "valid": bool(len(completed) >= n_episodes)}


def run(cfg: TrainConfig):
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    m = float(getattr(cfg, "scratch_mult", 1.0))
    base = ff.FFConfig()
    cfg_ff = ff.FFConfig(
        scratch_rank_conv=min(int(round(base.scratch_rank_conv * m)), base.comp_rank_conv),
        scratch_rank_dense=min(int(round(base.scratch_rank_dense * m)), base.comp_rank_dense),
        scratch_rank_head=min(int(round(base.scratch_rank_head * m)), base.comp_rank_head),
    )
    rng = jax.random.PRNGKey(cfg.seed)
    rng, bkey = jax.random.split(rng)
    banks = ff.init_banks(bkey, cfg_ff)
    book = addr.empty_address_book(d_k=cfg_ff.d_k, n_max=max(8, len(cfg.games) + 2), seed=cfg.seed)
    ctx_ids = []
    for _ in cfg.games:
        book, ctx = addr.allocate_canonical(book)
        ctx_ids.append(ctx)

    log(f"CASTM oracle ladder games={cfg.games} steps/game={cfg.steps_per_game} seed={cfg.seed}")

    # Random scores for normalization.
    random_scores = {}
    for g in cfg.games:
        random_scores[g] = random_score(g, num_envs=min(cfg.num_envs, cfg.eval_episodes),
                                        n_episodes=cfg.eval_episodes, seed=cfg.seed + 777, fire_reset=cfg.fire_reset)
        log(f"random[{g}] = {random_scores[g]:.2f}")

    sample, value_only, sample_sparse = _make_act(cfg_ff)
    best_after_learn = {}
    retention_matrix = []  # list of dicts: after game i, scores for games 0..i
    full_curves = {}

    naive = bool(getattr(cfg, "naive", False))
    if naive:
        log("NAIVE BASELINE: every game fine-tunes the shared net (no freeze, no "
            "addressed memory) — the catastrophic-forgetting control.")

    for gi, game in enumerate(cfg.games):
        ctx_id = ctx_ids[gi]
        # CASTM: game 0 -> shared W0, later games -> isolated scratch committed to
        # their address. Naive control: ALL games fine-tune the same shared net.
        mode = "shared" if (gi == 0 or naive) else "scratch"
        banks, best, curve, committed = train_one_game(
            cfg, cfg_ff, banks, book, game, ctx_id, mode=mode, rng=rng, out_dir=out_dir, log=log
        )
        best_after_learn[game] = best
        full_curves[game] = curve

        # Evaluate every game seen so far with sparse gather at its own address.
        row = {"after_game": game, "after_index": gi, "scores": {}}
        for gj in range(gi + 1):
            gname = cfg.games[gj]
            cj = ctx_ids[gj]
            ev = evaluate_game(sample_sparse, banks, gname, addr.code(book, cj), cj,
                               num_envs=min(cfg.num_envs, cfg.eval_episodes),
                               n_episodes=cfg.eval_episodes, seed=cfg.seed + 12345 + gj,
                               fire_reset=cfg.fire_reset)
            row["scores"][gname] = ev
            log(f"  eval-after[{game}] {gname}: mean={ev['mean']:.1f} valid={ev['valid']} n={ev['n']}")
        retention_matrix.append(row)

        # Persist incrementally.
        payload = _build_payload(cfg, random_scores, best_after_learn, retention_matrix, full_curves)
        (out_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log(f"persisted results after {game}")

    inferred = None
    if cfg.inferred_eval:
        log("=== STAGE D: inferred-address routing eval ===")
        from tfns.castm import infer_eval as ie
        ev_envs = min(cfg.num_envs, cfg.eval_episodes)
        prototypes = {}
        for gi, game in enumerate(cfg.games):
            c = ctx_ids[gi]
            q = ie.collect_anchor_queries(cfg_ff, banks, book, game, c, num_envs=cfg.num_envs,
                                          n_frames=cfg.anchor_frames, seed=cfg.seed + 4000 + gi,
                                          fire_reset=cfg.fire_reset)
            prototypes[c] = ie.build_prototypes(q, m_p=cfg.proto_per_ctx)
            log(f"  built {prototypes[c].shape[0]} prototypes for ctx{c} ({game})")
        racc = ie.routing_accuracy(cfg_ff, banks, book, list(cfg.games), ctx_ids, prototypes,
                                   num_envs=cfg.num_envs, n_frames=cfg.anchor_frames,
                                   seed=cfg.seed + 6000, fire_reset=cfg.fire_reset)
        log(f"  router top-1 accuracy overall={racc['overall']:.4f} per_game={racc['per_game']}")
        inferred_scores = {}
        for gi, game in enumerate(cfg.games):
            ev = ie.inferred_eval_game(cfg_ff, banks, book, game, ctx_ids[gi], prototypes, ctx_ids,
                                       num_envs=ev_envs, n_episodes=cfg.eval_episodes,
                                       seed=cfg.seed + 8000 + gi, fire_reset=cfg.fire_reset)
            inferred_scores[game] = ev
            log(f"  inferred[{game}] mean={ev['mean']:.1f} route_acc={ev['route_acc']:.4f} valid={ev['valid']}")
        inferred = {"routing_accuracy": racc, "scores": inferred_scores}

    log("DONE")
    payload = _build_payload(cfg, random_scores, best_after_learn, retention_matrix, full_curves, inferred)
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _build_payload(cfg, random_scores, best_after_learn, retention_matrix, full_curves, inferred=None):
    return {
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()},
        "random_scores": random_scores,
        "best_after_learn": best_after_learn,
        "retention_matrix": retention_matrix,
        "learning_curves": full_curves,
        "inferred": inferred,
    }


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--games", nargs="+", default=["Alien-v5", "Defender-v5"])
    p.add_argument("--steps-per-game", type=int, default=2_000_000)
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--lr-scratch", type=float, default=1.0e-3)
    p.add_argument("--out-dir", type=str, default="castm_runs/oracle/run")
    p.add_argument("--inferred-eval", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--scratch-mult", type=float, default=1.0)
    p.add_argument("--naive", action="store_true", help="catastrophic-forgetting control: fine-tune shared net on all games")
    args = p.parse_args()
    return TrainConfig(
        games=tuple(args.games), steps_per_game=args.steps_per_game, num_envs=args.num_envs,
        seed=args.seed, eval_episodes=args.eval_episodes, lr_scratch=args.lr_scratch, out_dir=args.out_dir,
        inferred_eval=args.inferred_eval, scratch_mult=args.scratch_mult, naive=args.naive,
    )


if __name__ == "__main__":
    run(parse_args())
