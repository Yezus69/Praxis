"""Fully-plastic CASTM continual PPO (no frozen weights) — spec 8.2/10.

Every game trains the FULL shared weights W0 with raw PPO gradients (maximum
plasticity, no freezing). Old games are protected by a compact per-context
low-rank memory that is *re-solved* after each game to absorb the W0 drift:

    after training game j, with drift  dW0 = W0_new - W0_old,
    for every previously-learned context c:
        M_c <- SVD_R( decode_delta(M_c, k_c) - dW0 )      (bias: beta_c -= db0)

so that  W(k_c) = W0_new + M_c(k_c) ≈ V_c  (the score-achieving weights of game c).
The just-finished game j carries no component (it decodes to the live W0_new);
older games accumulate low-rank drift corrections. Retention is therefore exact
only up to the memory rank R — the honest stability/plasticity/compactness
trilemma. No game/task ID enters the policy forward (the address is an internal
canonical code; ctx ids select the sparse gather only).

Run:
    python -m tfns.castm.train_plastic --games Breakout-v5 ... --mem-rank 64
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

import baseline_ppo as bp
from tfns.castm import address as addr
from tfns.castm import ff
from tfns.castm import synaptic as syn
from tfns.castm.train_castm import Batch, _ppo_terms, evaluate_game, random_score, _eval_live


@dataclass(frozen=True)
class PlasticConfig:
    games: tuple[str, ...] = ("Breakout-v5", "Pong-v5", "SpaceInvaders-v5", "Seaquest-v5", "BeamRider-v5")
    steps_per_game: int = 1_500_000
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
    lr: float = 2.5e-4
    mem_rank: int = 64
    out_dir: str = "castm_runs/plastic/run"
    fire_reset: bool = True


def loss_plastic(trainable, banks_frozen, cfg_ff, k, ctx_id, batch, clip, vf, ent):
    banks = ff.apply_shared_trainable(banks_frozen, trainable)
    # ctx_id is the *current* (component-less) context -> forward decodes pure W0.
    logits, values = ff.forward(banks, None, cfg_ff, batch.obs, k, ctx_id=int(ctx_id), sparse=True)
    return _ppo_terms(logits, values, batch, clip, vf, ent)


def _make_plastic_update(cfg_ff, cfg: PlasticConfig):
    @partial(jax.jit, static_argnames=("minibatch_size", "ctx_id"))
    def update(params, frozen, k, ctx_id, batch, rng, lr, minibatch_size, opt_state):
        # zero_nans guards against a single divergent minibatch destroying W0;
        # adam eps=1e-5 is the standard Atari-PPO stabilizer for continued training.
        opt = optax.chain(optax.zero_nans(), optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(lr, eps=1e-5))
        batch_size = cfg.num_minibatches * minibatch_size

        def epoch(carry, _):
            params, opt_state, rng = carry
            rng, pk = jax.random.split(rng)
            perm = jax.random.permutation(pk, batch_size)

            def shuf(x):
                return jnp.take(x, perm, axis=0).reshape((cfg.num_minibatches, minibatch_size) + x.shape[1:])

            def mb(carry, b):
                params, opt_state = carry
                (loss, metrics), grads = jax.value_and_grad(loss_plastic, has_aux=True)(
                    params, frozen, cfg_ff, k, ctx_id, b, cfg.clip_coef, cfg.vf_coef, cfg.ent_coef
                )
                updates, opt_state = opt.update(grads, opt_state, params)
                return (optax.apply_updates(params, updates), opt_state), metrics

            (params, opt_state), metrics = jax.lax.scan(mb, (params, opt_state), jax.tree_util.tree_map(shuf, batch))
            return (params, opt_state, rng), metrics.mean(axis=0)

        (params, opt_state, rng), metrics = jax.lax.scan(epoch, (params, opt_state, rng), None, length=cfg.update_epochs)
        return params, opt_state, rng, metrics.mean(axis=0)

    return update


def _svd_lowrank(residual, rank):
    """Return (A (r,in), B (out,r)) with B@A ≈ residual, truncated to rank."""

    res = np.asarray(residual, dtype=np.float64)
    u, s, vt = np.linalg.svd(res, full_matrices=False)
    r = int(min(int(rank), s.size))
    sq = np.sqrt(s[:r])
    B = (u[:, :r] * sq).astype(np.float32)
    A = (sq[:, None] * vt[:r]).astype(np.float32)
    return jnp.asarray(A), jnp.asarray(B)


def resolve_memory(banks, book, dW0, db0, old_ctxs, mem_rank):
    """Re-solve per-context low-rank corrections to absorb W0 drift (spec 10).

    For each old context c: new correction = SVD_R(current_correction - dW0),
    so W0_new + correction restores context c's prior decoded weights.
    """

    new_banks = {}
    drift_err = {}
    for name, mem in banks.items():
        m2 = syn.empty_synaptic_memory(mem.W0, mem.b0, comp_rank=mem.comp_rank,
                                       n_slots=mem.n_slots, d_k=mem.d_k)
        layer_err = 0.0
        for c in old_ctxs:
            k_c = addr.code(book, c)
            cur = np.asarray(syn.decode_delta(mem, k_c), dtype=np.float64)        # B_c A_c (only c at k_c)
            cur_beta = np.asarray(syn.decode_bias_delta(mem, k_c), dtype=np.float64)
            residual = cur - np.asarray(dW0[name], dtype=np.float64)
            beta_new = (cur_beta - np.asarray(db0[name], dtype=np.float64)).astype(np.float32)
            A, B = _svd_lowrank(residual, min(mem_rank, mem.comp_rank))
            approx = np.asarray(B @ A, dtype=np.float64)
            layer_err = max(layer_err, float(np.max(np.abs(approx - residual))))
            m2, slot = syn.append_component(m2, A, B, jnp.asarray(beta_new), addr.code(book, c), int(c))
        new_banks[name] = m2
        drift_err[name] = layer_err
    return new_banks, drift_err


def run(cfg: PlasticConfig):
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    # Memory rank governs correction fidelity; size comp_rank to it.
    R = int(cfg.mem_rank)
    cfg_ff = ff.FFConfig(comp_rank_conv=R, comp_rank_dense=R, comp_rank_head=min(R, 32), n_slots=max(8, len(cfg.games) + 2))
    rng = jax.random.PRNGKey(cfg.seed)
    rng, bkey = jax.random.split(rng)
    banks = ff.init_banks(bkey, cfg_ff)
    book = addr.empty_address_book(d_k=cfg_ff.d_k, n_max=max(8, len(cfg.games) + 2), seed=cfg.seed)
    ctx_ids = []
    for _ in cfg.games:
        book, ctx = addr.allocate_canonical(book)
        ctx_ids.append(ctx)

    log(f"FULLY-PLASTIC CASTM (no frozen weights) games={cfg.games} mem_rank={R} seed={cfg.seed}")
    random_scores = {}
    for g in cfg.games:
        random_scores[g] = random_score(g, num_envs=min(cfg.num_envs, cfg.eval_episodes),
                                        n_episodes=cfg.eval_episodes, seed=cfg.seed + 777, fire_reset=cfg.fire_reset)
        log(f"random[{g}] = {random_scores[g]:.2f}")

    # eval helper (stochastic, sparse gather at each context's address)
    @partial(jax.jit, static_argnames=("ctx_id",))
    def sample_sparse(banks, obs, k, ctx_id, fire, rng):
        rng, key = jax.random.split(rng)
        logits, _ = ff.forward(banks, None, cfg_ff, obs, k, ctx_id=ctx_id, sparse=True)
        a = jax.random.categorical(key, logits, axis=-1).astype(jnp.int32)
        return jnp.where(fire, jnp.full_like(a, bp.FIRE_ACTION), a), rng

    update = _make_plastic_update(cfg_ff, cfg)
    num_envs, num_steps = cfg.num_envs, cfg.num_steps
    batch_size = num_envs * num_steps
    minibatch_size = batch_size // cfg.num_minibatches

    best_after_learn = {}
    retention_matrix = []
    drift_log = []

    for gi, game in enumerate(cfg.games):
        ctx_id = ctx_ids[gi]
        k = addr.code(book, ctx_id)
        # snapshot W0 before this game (to compute drift afterward)
        W0_snap = {name: np.asarray(banks[name].W0) for name in banks}
        b0_snap = {name: np.asarray(banks[name].b0) for name in banks}

        params = ff.shared_trainable(banks)
        opt = optax.chain(optax.zero_nans(), optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(cfg.lr, eps=1e-5))
        opt_state = opt.init(params)
        num_updates = max(1, cfg.steps_per_game // batch_size)

        env = bp.make_env(game, num_envs, cfg.seed + ctx_id, training=True)
        obs0, _ = bp.reset_result(env.reset())
        next_obs = bp.nhwc_uint8(obs0)
        fire = np.full((num_envs,), bool(cfg.fire_reset), np.bool_)
        obs_b = np.zeros((num_steps, num_envs, 84, 84, 4), np.uint8)
        act_b = np.zeros((num_steps, num_envs), np.int32)
        lp_b = np.zeros((num_steps, num_envs), np.float32)
        rew_b = np.zeros((num_steps, num_envs), np.float32)
        done_b = np.zeros((num_steps, num_envs), np.float32)
        val_b = np.zeros((num_steps, num_envs), np.float32)

        @jax.jit
        def act(banks, obs, k, fire, rng):
            rng, key = jax.random.split(rng)
            logits, value = ff.forward(banks, None, cfg_ff, obs, k, ctx_id=int(ctx_id), sparse=True)
            a = jax.random.categorical(key, logits, axis=-1).astype(jnp.int32)
            a = jnp.where(fire, jnp.full_like(a, bp.FIRE_ACTION), a)
            return a, bp.log_prob(logits, a), value, rng

        @jax.jit
        def value_only(banks, obs, k):
            return ff.forward(banks, None, cfg_ff, obs, k, ctx_id=int(ctx_id), sparse=True)[1]

        best = -1e30
        start = time.perf_counter()
        for ui in range(1, num_updates + 1):
            banks_now = ff.apply_shared_trainable(banks, params)
            for step in range(num_steps):
                obs_b[step] = next_obs
                a, lp, v, rng = act(banks_now, jnp.asarray(next_obs), k, jnp.asarray(fire), rng)
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
            last_v = value_only(banks_now, jnp.asarray(next_obs), k)
            adv, returns = bp.compute_gae(jnp.asarray(rew_b), jnp.asarray(done_b), jnp.asarray(val_b), last_v, cfg.gamma, cfg.gae_lambda)
            batch = Batch(
                obs=jnp.asarray(obs_b).reshape((batch_size,) + obs_b.shape[2:]),
                actions=jnp.asarray(act_b).reshape((batch_size,)),
                logprobs=jnp.asarray(lp_b).reshape((batch_size,)),
                advantages=jnp.asarray(adv).reshape((batch_size,)),
                returns=jnp.asarray(returns).reshape((batch_size,)),
                values=jnp.asarray(val_b).reshape((batch_size,)),
            )
            lr = cfg.lr * (1.0 - (ui - 1.0) / float(num_updates))
            params, opt_state, rng, metrics = update(params, banks, k, int(ctx_id), batch, rng, lr, minibatch_size, opt_state)
            if ui == 1 or ui % cfg.eval_every_updates == 0 or ui == num_updates:
                banks_eval = ff.apply_shared_trainable(banks, params)
                ev = _eval_live(cfg_ff, banks_eval, None, game, k, num_envs=min(num_envs, cfg.eval_episodes),
                                n_episodes=cfg.eval_episodes, seed=cfg.seed + 9000 + ui, fire_reset=cfg.fire_reset)
                steps_done = ui * batch_size
                sps = steps_done / max(time.perf_counter() - start, 1e-6)
                best = max(best, ev["mean"] if np.isfinite(ev["mean"]) else best)
                log(f"[{game} ctx{ctx_id} PLASTIC] upd={ui}/{num_updates} steps={steps_done} "
                    f"eval_mean={ev['mean']:.1f} best={best:.1f} sps={sps:.0f} ent={metrics[3]:.3f}")
        env_close = getattr(env, "close", None)
        if callable(env_close):
            env_close()

        # commit: W0 is now the just-learned game; absorb drift for OLD contexts.
        banks = ff.apply_shared_trainable(banks, params)
        dW0 = {name: (np.asarray(banks[name].W0) - W0_snap[name]) for name in banks}
        db0 = {name: (np.asarray(banks[name].b0) - b0_snap[name]) for name in banks}
        old_ctxs = ctx_ids[:gi]
        if old_ctxs:
            banks, drift_err = resolve_memory(banks, book, dW0, db0, old_ctxs, R)
            max_drift = max(np.max(np.abs(dW0[n])) for n in dW0)
            max_resid = max(drift_err.values())
            log(f"[{game}] re-solved {len(old_ctxs)} old contexts; max|dW0|={max_drift:.3f} "
                f"max correction residual={max_resid:.4f}")
            drift_log.append({"after_game": game, "max_dW0": float(max_drift), "max_residual": float(max_resid)})
        best_after_learn[game] = float(best)

        # retention eval: every game so far at its own address (sparse gather)
        row = {"after_game": game, "after_index": gi, "scores": {}}
        for gj in range(gi + 1):
            gname = cfg.games[gj]
            cj = ctx_ids[gj]
            ev = evaluate_game(sample_sparse, banks, gname, addr.code(book, cj), cj,
                               num_envs=min(num_envs, cfg.eval_episodes), n_episodes=cfg.eval_episodes,
                               seed=cfg.seed + 12345 + gj, fire_reset=cfg.fire_reset)
            row["scores"][gname] = ev
            log(f"  eval-after[{game}] {gname}: mean={ev['mean']:.1f} valid={ev['valid']} n={ev['n']}")
        retention_matrix.append(row)
        payload = {
            "config": {k2: (list(v) if isinstance(v, tuple) else v) for k2, v in asdict(cfg).items()},
            "method": "fully_plastic_memory_resolve",
            "random_scores": random_scores, "best_after_learn": best_after_learn,
            "retention_matrix": retention_matrix, "drift_log": drift_log,
        }
        (out_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log(f"persisted after {game}")

    log("DONE")
    return payload


def parse_args() -> PlasticConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--games", nargs="+", required=True)
    p.add_argument("--steps-per-game", type=int, default=1_500_000)
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--mem-rank", type=int, default=64)
    p.add_argument("--out-dir", type=str, default="castm_runs/plastic/run")
    args = p.parse_args()
    return PlasticConfig(games=tuple(args.games), steps_per_game=args.steps_per_game, num_envs=args.num_envs,
                         seed=args.seed, eval_episodes=args.eval_episodes, mem_rank=args.mem_rank, out_dir=args.out_dir)


if __name__ == "__main__":
    run(parse_args())
