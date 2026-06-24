"""Genuinely task-free, fully-plastic CASTM continual PPO (the overnight system).

One live neural system. All shared weights (conv encoder, dense, **policy and value
heads**) stay plastic across the whole stream. The learner receives **no** game id,
task id, curriculum index, manual context id, or announced boundary: the active
internal context is *inferred online* from the content of observations by
``OnlineContextManager`` (no game/task identity anywhere in the policy/router/
memory/optimizer path). Old functional knowledge is preserved in compact
content-addressed low-rank synaptic factors that are updated analytically as the
shared weights drift (``online_resolve``):

    during inferred context a:  W0 -> W0 + ΔW0
    every inactive context c:   D_c -> Compress_R(D_c - ΔW0),  β_c -> β_c - Δb0
    active context a:           D_a unchanged  (rides the full update)

so an inactive context's decoded operator ``W0' + D_c'`` is preserved while the
active context learns. New contexts are discovered online (``D=0`` at the current
W0) and become protected once the stream leaves them.

Differences from the historical ``train_plastic.py`` (which is boundary-aware):
no preallocated addresses, no curriculum-index context, **no head re-initialisation**
(replaced by novelty-triggered exploration), online resolve mid-stream, online
prototypes from raw anchors, and **inferred routing is the primary retention eval**
(oracle addressing is reported only as a diagnostic).

Game names live ONLY in the outer harness/evaluator (to fetch matched references and
to *label* held-out frames when scoring routing accuracy). They never enter the
agent, router, optimizer, context manager, or memory.

Run (one process per GPU):
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
      python -m tfns.castm.train_taskfree --games SpaceInvaders-v5 Seaquest-v5 \
        --steps-per-game 500000 --out-dir castm_runs/taskfree/stage1 --seed 1
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
from tfns.castm import context_manager as cm
from tfns.castm import ff
from tfns.castm import online_resolve as orx
from tfns.castm import synaptic as syn
from tfns.castm.train_castm import Batch, _ppo_terms, evaluate_game, random_score, _eval_live


@dataclass(frozen=True)
class TaskFreeConfig:
    games: tuple[str, ...] = ("SpaceInvaders-v5", "Seaquest-v5")
    steps_per_game: int = 500_000
    num_envs: int = 32
    num_steps: int = 128
    seed: int = 1
    eval_episodes: int = 12
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
    resolve_every: int = 20            # PPO updates between periodic resolves
    out_dir: str = "castm_runs/taskfree/run"
    fire_reset: bool = True
    resolve: bool = True               # False = naive control (no memory protection)
    include_value: bool = True         # protect the value head in the resolve (ablatable)
    # online context manager
    proto_per_ctx: int = 12
    known_thresh: float = 0.55
    novel_thresh: float = 0.40
    novel_persist: int = 3
    min_dwell: int = 3
    anchor_cap: int = 2048
    max_contexts: int = 16
    # novelty-triggered exploration (replaces head re-init)
    ent_boost: float = 4.0             # entropy-coef multiplier right after an ALLOC
    ent_boost_updates: int = 40        # updates over which the boost decays to 1
    reset_opt_on_alloc: bool = True    # damp Adam moments at a detected regime change
    intervention: str = "boost_reset"  # {"boost_reset","entropy","reset","none"} — ablation
    calibrate: bool = False            # legacy fixed-threshold probe; adaptive routing self-calibrates


def loss_taskfree(trainable, banks_frozen, cfg_ff, k, ctx_id, batch, free_mask, clip, vf, ent):
    banks = ff.apply_shared_trainable(banks_frozen, trainable)
    logits, values = ff.forward(banks, None, cfg_ff, batch.obs, k, ctx_id=int(ctx_id), sparse=True)
    return _ppo_terms(logits, values, batch, clip, vf, ent, free_mask=free_mask)


def _make_update(cfg_ff, cfg: TaskFreeConfig):
    @partial(jax.jit, static_argnames=("minibatch_size", "ctx_id"))
    def update(params, frozen, k, ctx_id, batch, free_mask, rng, lr, ent, minibatch_size, opt_state):
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
                mb_batch, mb_free = b
                (loss, metrics), grads = jax.value_and_grad(loss_taskfree, has_aux=True)(
                    params, frozen, cfg_ff, k, ctx_id, mb_batch, mb_free, cfg.clip_coef, cfg.vf_coef, ent
                )
                updates, opt_state = opt.update(grads, opt_state, params)
                return (optax.apply_updates(params, updates), opt_state), metrics

            shuffled = jax.tree_util.tree_map(shuf, (batch, free_mask))
            (params, opt_state), metrics = jax.lax.scan(mb, (params, opt_state), shuffled)
            return (params, opt_state, rng), metrics.mean(axis=0)

        (params, opt_state, rng), metrics = jax.lax.scan(epoch, (params, opt_state, rng), None, length=cfg.update_epochs)
        return params, opt_state, rng, metrics.mean(axis=0)

    return update


def _intervention_flags(cfg: TaskFreeConfig):
    boost = cfg.intervention in ("boost_reset", "entropy")
    reset = cfg.intervention in ("boost_reset", "reset") and cfg.reset_opt_on_alloc
    return boost, reset


def run(cfg: TaskFreeConfig):
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    R = int(cfg.mem_rank)
    n_slots = max(8, cfg.max_contexts + 4)
    cfg_ff = ff.FFConfig(comp_rank_conv=R, comp_rank_dense=R, comp_rank_head=min(R, 32), n_slots=n_slots)
    rng = jax.random.PRNGKey(cfg.seed)
    rng, bkey = jax.random.split(rng)
    banks = ff.init_banks(bkey, cfg_ff)

    content_grid = 8
    content_dq = content_grid * content_grid * cfg_ff.frame_stack
    mgr_cfg = cm.ManagerConfig(
        d_q=content_dq, d_k=cfg_ff.d_k, n_max=n_slots, max_contexts=cfg.max_contexts,
        proto_per_ctx=cfg.proto_per_ctx, known_thresh=cfg.known_thresh, novel_thresh=cfg.novel_thresh,
        novel_persist=cfg.novel_persist, min_dwell=cfg.min_dwell, anchor_cap=cfg.anchor_cap, seed=cfg.seed)
    mgr = cm.OnlineContextManager(mgr_cfg)
    boost_on, reset_on = _intervention_flags(cfg)

    log(f"TASK-FREE CASTM games={cfg.games} mem_rank={R} seed={cfg.seed} "
        f"intervention={cfg.intervention} resolve_every={cfg.resolve_every} include_value={cfg.include_value}")

    random_scores = {}
    for g in cfg.games:
        random_scores[g] = random_score(g, num_envs=min(cfg.num_envs, cfg.eval_episodes),
                                        n_episodes=cfg.eval_episodes, seed=cfg.seed + 777, fire_reset=cfg.fire_reset)
        log(f"random[{g}] = {random_scores[g]:.2f}")

    # jitted helpers ----------------------------------------------------------
    @partial(jax.jit, static_argnames=("ctx_id",))
    def act(banks, obs, k, ctx_id, fire, rng):
        rng, key = jax.random.split(rng)
        logits, value = ff.forward(banks, None, cfg_ff, obs, k, ctx_id=ctx_id, sparse=True)
        a = jax.random.categorical(key, logits, axis=-1).astype(jnp.int32)
        a = jnp.where(fire, jnp.full_like(a, bp.FIRE_ACTION), a)
        return a, bp.log_prob(logits, a), value, rng

    @partial(jax.jit, static_argnames=("ctx_id",))
    def value_only(banks, obs, k, ctx_id):
        return ff.forward(banks, None, cfg_ff, obs, k, ctx_id=ctx_id, sparse=True)[1]

    @jax.jit
    def query_of(banks, obs, k):
        return ff.content_features(banks, cfg_ff, obs, k)   # RAW features; manager centers them

    def content_sig(obs):
        """Raw-observation content signature (pooled pixels). Encoder-independent."""
        return cm.pooled_signature(obs, content_grid)

    def query_fn_factory(banks_ref=None):
        """raw frames -> content signatures (pooled pixels; no GPU, drift-free)."""
        def qfn(frames):
            if len(frames) == 0:
                return np.zeros((0, content_dq), np.float32)
            return content_sig(frames)
        return qfn

    @partial(jax.jit, static_argnames=("ctx_id",))
    def sample_sparse(banks, obs, k, ctx_id, fire, rng):
        rng, key = jax.random.split(rng)
        logits, _ = ff.forward(banks, None, cfg_ff, obs, k, ctx_id=ctx_id, sparse=True)
        a = jax.random.categorical(key, logits, axis=-1).astype(jnp.int32)
        return jnp.where(fire, jnp.full_like(a, bp.FIRE_ACTION), a), rng

    update = _make_update(cfg_ff, cfg)
    num_envs, num_steps = cfg.num_envs, cfg.num_steps
    batch_size = num_envs * num_steps
    minibatch_size = batch_size // cfg.num_minibatches

    params = ff.shared_trainable(banks)
    opt = optax.chain(optax.zero_nans(), optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(cfg.lr, eps=1e-5))
    opt_state = opt.init(params)
    snap_W0, snap_b0 = orx.snapshot_shared(ff.apply_shared_trainable(banks, params))

    # optional threshold calibration probe (diagnostic; uses no labels in the agent) --
    if cfg.calibrate and len(cfg.games) >= 2:
        _calibrate_thresholds(cfg, cfg_ff, banks, query_of, mgr, log)

    # buffers
    obs_b = np.zeros((num_steps, num_envs, 84, 84, 4), np.uint8)
    act_b = np.zeros((num_steps, num_envs), np.int32)
    lp_b = np.zeros((num_steps, num_envs), np.float32)
    rew_b = np.zeros((num_steps, num_envs), np.float32)
    done_b = np.zeros((num_steps, num_envs), np.float32)
    val_b = np.zeros((num_steps, num_envs), np.float32)
    forced_b = np.zeros((num_steps, num_envs), np.float32)

    routing_log = []
    resolve_log = []
    drift_log = []
    game_to_ctx = {}             # evaluator label only: dominant ctx during each game segment
    best_after_learn = {}
    retention_matrix = []
    updates_since_resolve = 0
    alloc_age = 10 ** 9          # updates since last ALLOC (for entropy boost)
    global_update = 0

    def cur_banks():
        return ff.apply_shared_trainable(banks, params)

    def do_resolve(active_ctx, tag):
        nonlocal banks, params, opt_state, snap_W0, snap_b0, updates_since_resolve
        if not cfg.resolve:
            # Naive control: no memory protection. Re-snapshot so 'drift since' stays
            # bounded, but never compensate inactive contexts -> catastrophic forgetting.
            snap_W0, snap_b0 = orx.snapshot_shared(ff.apply_shared_trainable(banks, params))
            updates_since_resolve = 0
            return {"budget_ok": True, "max_residual": 0.0, "tag": tag, "disabled": True}
        live = ff.apply_shared_trainable(banks, params)
        new_banks, rep = orx.online_resolve(live, mgr.book, snap_W0, snap_b0, int(active_ctx),
                                            list(mgr.ctx_ids), R, include_value=cfg.include_value)
        banks = new_banks
        params = ff.shared_trainable(banks)             # W0/b0 unchanged by resolve; deltas updated
        # opt_state stays valid (params unchanged); keep momentum across periodic resolves.
        snap_W0, snap_b0 = orx.snapshot_shared(banks)
        updates_since_resolve = 0
        rep["tag"] = tag
        rep["update"] = global_update
        resolve_log.append({k2: rep[k2] for k2 in ("active_ctx", "n_inactive", "max_dW0", "max_residual",
                                                   "max_rel_fro", "budget_ok", "tag", "update")})
        # Refresh prototypes from raw anchors under the (now-current) encoder (drift handling).
        try:
            audit = mgr.refresh_prototypes(query_fn_factory(cur_banks()))
            rep["route_overall"] = audit["overall"]
        except Exception as e:  # never let a refresh hiccup kill a long run
            rep["route_overall"] = None
            log(f"  [resolve {tag}] prototype refresh skipped: {e}")
        log(f"  [resolve {tag} u{global_update}] active={active_ctx} inactive={rep['n_inactive']} "
            f"max|dW0|={rep['max_dW0']:.3f} resid={rep['max_residual']:.4f} budget_ok={rep['budget_ok']} "
            f"route={rep.get('route_overall')}")
        return rep

    start = time.perf_counter()
    for gi, game in enumerate(cfg.games):
        env = bp.make_env(game, num_envs, cfg.seed + 1000 + gi, training=True)
        obs0, _ = bp.reset_result(env.reset())
        next_obs = bp.nhwc_uint8(obs0)
        fire = np.full((num_envs,), bool(cfg.fire_reset), np.bool_)
        num_updates = max(1, cfg.steps_per_game // batch_size)
        best = -1e30
        seg_ctx_counts: dict[int, int] = {}

        for ui in range(1, num_updates + 1):
            global_update += 1
            banks_now = cur_banks()
            # --- routing decision at rollout start (content only) ---
            prev_active = int(mgr.active_ctx)        # outgoing context, captured BEFORE routing
            queries = content_sig(next_obs)          # pooled-pixel content signature
            ev = mgr.update(queries, next_obs)
            active_ctx = mgr.active_ctx
            decision = ev["decision"]
            k_active = jnp.asarray(mgr.active_address())
            ctx_for_forward = int(active_ctx) if active_ctx >= 0 else -999
            seg_ctx_counts[active_ctx] = seg_ctx_counts.get(active_ctx, 0) + 1

            # On a switch/alloc, resolve attributing the just-ended interval to the
            # OUTGOING context (the context active before this rollout). W0 was held
            # during the ambiguous (non-KNOWN) tail, so this credits the right context.
            outgoing = int(ev.get("prev_ctx", prev_active))
            if ev.get("switched") and outgoing >= 0:
                do_resolve(outgoing, tag="switch")
            elif ev.get("allocated") and mgr.num_contexts > 1 and outgoing >= 0:
                do_resolve(outgoing, tag="alloc")
            if ev.get("allocated"):
                alloc_age = 0
                if reset_on:
                    opt_state = opt.init(params)   # damp stale momentum (weights continuous)

            # --- collect a rollout acting at the inferred address ---
            for step in range(num_steps):
                obs_b[step] = next_obs
                forced_b[step] = fire.astype(np.float32)
                a, lp, v, rng = act(banks_now, jnp.asarray(next_obs), k_active, ctx_for_forward,
                                    jnp.asarray(fire), rng)
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

            last_v = value_only(banks_now, jnp.asarray(next_obs), k_active, ctx_for_forward)
            adv, returns = bp.compute_gae(jnp.asarray(rew_b), jnp.asarray(done_b), jnp.asarray(val_b),
                                          last_v, cfg.gamma, cfg.gae_lambda)
            batch = Batch(
                obs=jnp.asarray(obs_b).reshape((batch_size,) + obs_b.shape[2:]),
                actions=jnp.asarray(act_b).reshape((batch_size,)),
                logprobs=jnp.asarray(lp_b).reshape((batch_size,)),
                advantages=jnp.asarray(adv).reshape((batch_size,)),
                returns=jnp.asarray(returns).reshape((batch_size,)),
                values=jnp.asarray(val_b).reshape((batch_size,)),
            )
            free_mask = jnp.asarray(1.0 - forced_b).reshape((batch_size,))

            # --- W0 update: ONLY when confidently dwelling in a known context ---
            apply_update = (decision == "KNOWN") and (active_ctx >= 0)
            if apply_update:
                lr = cfg.lr * max(0.1, 1.0 - global_update / float(num_updates * len(cfg.games)))
                ent_now = cfg.ent_coef
                if boost_on and alloc_age < cfg.ent_boost_updates:
                    frac = 1.0 - alloc_age / float(cfg.ent_boost_updates)
                    ent_now = cfg.ent_coef * (1.0 + (cfg.ent_boost - 1.0) * frac)
                params, opt_state, rng, metrics = update(
                    params, banks, k_active, ctx_for_forward, batch, free_mask, rng, lr, ent_now,
                    minibatch_size, opt_state)
                updates_since_resolve += 1
                alloc_age += 1
                ent_metric = float(metrics[3])
            else:
                ent_metric = float("nan")

            # --- periodic resolve ---
            if updates_since_resolve >= cfg.resolve_every and mgr.num_contexts >= 1:
                do_resolve(int(active_ctx), tag="periodic")

            a_sim = float(getattr(mgr, "last_active_sim", 0.0))
            b_sim = float(getattr(mgr, "last_best_sim", 0.0))
            a_lvl = float(getattr(mgr, "last_active_level", 0.0))
            routing_log.append({"u": global_update, "game_idx": gi, "decision": decision,
                                "active_ctx": int(active_ctx), "num_ctx": mgr.num_contexts,
                                "active_sim": a_sim, "best_sim": b_sim, "level": a_lvl})
            if decision != "KNOWN" or ev.get("allocated") or ev.get("switched"):
                log(f"  [route u{global_update} g{gi}] {decision} ctx={active_ctx} nctx={mgr.num_contexts} "
                    f"active_sim={a_sim:.3f} best_sim={b_sim:.3f} level={a_lvl:.3f} "
                    f"mean={getattr(mgr,'last_active_mean',0.0):.3f} dev={getattr(mgr,'last_active_dev',0.0):.3f}")

            if ui == 1 or ui % cfg.eval_every_updates == 0 or ui == num_updates:
                banks_eval = cur_banks()
                ev_live = _eval_live(cfg_ff, banks_eval, None, game, k_active,
                                     num_envs=min(num_envs, cfg.eval_episodes), n_episodes=cfg.eval_episodes,
                                     seed=cfg.seed + 9000 + global_update, fire_reset=cfg.fire_reset)
                steps_done = global_update * batch_size
                sps = steps_done / max(time.perf_counter() - start, 1e-6)
                if np.isfinite(ev_live["mean"]):
                    best = max(best, ev_live["mean"])
                log(f"[{game} g{gi}] u={ui}/{num_updates} dec={decision} ctx={active_ctx} "
                    f"nctx={mgr.num_contexts} eval={ev_live['mean']:.1f} best={best:.1f} sps={sps:.0f} ent={ent_metric:.3f}")

        env_close = getattr(env, "close", None)
        if callable(env_close):
            env_close()

        # The context the stream SETTLED into for this game (evaluator label only).
        # Use the current active context, not the most-frequent: detection latency at a
        # switch can make a stale context most-frequent even though the segment ended in
        # the correct (newly-discovered) context.
        settled = int(mgr.active_ctx)
        dom = max(seg_ctx_counts, key=lambda c: seg_ctx_counts[c]) if seg_ctx_counts else -1
        game_to_ctx[game] = settled
        best_after_learn[game] = float(best)

        # End-of-segment resolve: attribute the drift since the last resolve to the
        # context that has been active since then (the settled context).
        if mgr.num_contexts >= 1 and updates_since_resolve > 0:
            do_resolve(settled, tag="segment_end")
        log(f"  [segment end] {game} settled_ctx={settled} dominant_ctx={dom} num_contexts={mgr.num_contexts}")

        # --- retention eval after each segment: ORACLE (diagnostic) + INFERRED (primary) ---
        row = {"after_game": game, "after_index": gi, "oracle": {}, "inferred": {}}
        for gj in range(gi + 1):
            gname = cfg.games[gj]
            cj = game_to_ctx.get(gname, -1)
            if cj < 0:
                continue
            ev_or = evaluate_game(sample_sparse, banks, gname, addr.code(mgr.book, cj), cj,
                                  num_envs=min(num_envs, cfg.eval_episodes), n_episodes=cfg.eval_episodes,
                                  seed=cfg.seed + 12345 + gj, fire_reset=cfg.fire_reset)
            row["oracle"][gname] = ev_or
            log(f"  oracle-after[{game}] {gname} ctx{cj}: mean={ev_or['mean']:.1f} valid={ev_or['valid']}")
        retention_matrix.append(row)
        _persist(out_dir, cfg, random_scores, best_after_learn, retention_matrix,
                 routing_log, resolve_log, game_to_ctx, mgr)
        log(f"persisted after {game} (num_contexts={mgr.num_contexts})")

    # --- FINAL inferred-routing evaluation (PRIMARY result) ---
    log("=== FINAL INFERRED-ROUTING EVAL (task-free, primary) ===")
    mgr.refresh_prototypes(query_fn_factory(cur_banks()))
    inferred = _final_inferred_eval(cfg, cfg_ff, banks, mgr, game_to_ctx,
                                    content_sig, sample_sparse, query_fn_factory(), log)
    payload = _persist(out_dir, cfg, random_scores, best_after_learn, retention_matrix,
                       routing_log, resolve_log, game_to_ctx, mgr, inferred=inferred)
    log("DONE")
    return payload


def _calibrate_thresholds(cfg, cfg_ff, banks, query_of, mgr, log):
    """Calibrate known/novel similarity thresholds from a short labelled probe.

    Diagnostic only: rolls a few hundred frames of two games through the *current*
    (untrained) encoder, measures within-vs-cross content similarity, and sets the
    manager's thresholds to the midpoint. The labels are used ONLY here, by the
    harness, to compute the statistic — never fed to the agent. This is a principled
    one-shot calibration, not a hyperparameter sweep.
    """

    import baseline_ppo as bp
    protos = {}
    for gi, game in enumerate(cfg.games[:2]):
        env = bp.make_env(game, cfg.num_envs, cfg.seed + 2000 + gi, training=False)
        obs, _ = bp.reset_result(env.reset())
        obs = bp.nhwc_uint8(obs)
        kq = jnp.zeros((cfg_ff.d_k,))
        qs = []
        for _ in range(8):
            q = np.asarray(query_of(banks, jnp.asarray(obs), kq))
            qs.append(q)
            a = np.random.default_rng(gi).integers(0, bp.ACT_DIM, size=cfg.num_envs).astype(np.int32)
            obs, _, term, trunc, _ = env.step(a)
            obs = bp.nhwc_uint8(obs)
        close = getattr(env, "close", None)
        if callable(close):
            close()
        protos[gi] = cm.farthest_point_prototypes(np.concatenate(qs, 0), cfg.proto_per_ctx)
    # within vs cross best-prototype similarity
    p0, p1 = protos[0], protos[1]
    within = float(np.mean([(p0 @ p0.T).max(axis=1).mean(), (p1 @ p1.T).max(axis=1).mean()])) \
        if p0.size and p1.size else 0.9
    cross = float(max((p0 @ p1.T).max(), (p1 @ p0.T).max())) if p0.size and p1.size else 0.0
    known = 0.5 * (within + cross)
    novel = cross + 0.5 * (within - cross) * 0.5
    mgr.cfg.known_thresh = float(max(0.3, min(0.9, known)))
    mgr.cfg.novel_thresh = float(max(0.2, min(known - 0.05, novel)))
    log(f"  [calibrate] within={within:.3f} cross={cross:.3f} -> known_thresh={mgr.cfg.known_thresh:.3f} "
        f"novel_thresh={mgr.cfg.novel_thresh:.3f}")


def _final_inferred_eval(cfg, cfg_ff, banks, mgr, game_to_ctx, content_sig, sample_sparse, query_fn, log):
    """Inferred (content-routed) eval through the MANAGER's centered prototypes.

    Two metrics, both task-free at inference (the address is chosen by the router from
    content; game labels are used only to score):
      * router top-1 accuracy on the manager's held-out raw-frame anchors (A_router);
      * inferred score per game: roll the game out, route every step via the manager
        (centered), act through the routed context's sparse address.
    """

    ctx_ids = list(mgr.ctx_ids)
    k_neutral = jnp.zeros((cfg_ff.d_k,), jnp.float32)  # placeholder address (acting uses routed addr)
    # dedupe games (an alternation may repeat a game); evaluate each distinct game once
    games, seen = [], set()
    for g in cfg.games:
        if g not in seen and game_to_ctx.get(g, -1) in ctx_ids:
            games.append(g); seen.add(g)

    # A_router: held-out routing accuracy from the manager's own raw anchors (centered).
    audit = mgr.audit_routing(query_fn)
    racc = {"overall": audit["overall"],
            "per_ctx": {int(c): v for c, v in audit["per_ctx"].items()},
            "max_inter_sim": audit["max_inter_sim"]}
    log(f"  router top-1 (held-out anchors) overall={racc['overall']:.4f} per_ctx={racc['per_ctx']} "
        f"max_inter_sim={racc['max_inter_sim']:.3f}")

    scores = {}
    for g in games:
        true_ctx = game_to_ctx[g]
        env = bp.make_env(g, min(cfg.num_envs, cfg.eval_episodes), cfg.seed + 8000, training=False)
        obs, _ = bp.reset_result(env.reset())
        obs = bp.nhwc_uint8(obs)
        n = obs.shape[0]
        running = np.zeros((n,), np.float32)
        fire = np.full((n,), bool(cfg.fire_reset), np.bool_)
        completed, steps = [], 0
        rc = rt_ = 0
        rng = jax.random.PRNGKey(cfg.seed + 8000)
        while len(completed) < cfg.eval_episodes and steps < 1_000_000:
            raw = content_sig(obs)                           # pooled-pixel signature
            routed = mgr.per_stream_route(raw)               # centered routing
            rc += int(np.sum(routed == true_ctx)); rt_ += n
            vals, counts = np.unique(routed, return_counts=True)
            cctx = int(vals[counts.argmax()])                # batch consensus (single game)
            k_c = jnp.asarray(mgr.address_of(cctx)) if cctx >= 0 else k_neutral
            cid = cctx if cctx >= 0 else -999
            a, rng = sample_sparse(banks, jnp.asarray(obs), k_c, cid, jnp.asarray(fire), rng)
            obs, reward, term, trunc, _ = env.step(np.asarray(jax.device_get(a), np.int32))
            obs = bp.nhwc_uint8(obs)
            done = np.logical_or(bp.vec(term, np.bool_, n, "t"), bp.vec(trunc, np.bool_, n, "tr"))
            running += bp.vec(reward, np.float32, n, "r")
            for idx in np.flatnonzero(done):
                if len(completed) < cfg.eval_episodes:
                    completed.append(float(running[idx]))
                running[idx] = 0.0
            fire = np.where(done, bool(cfg.fire_reset), False)
            steps += n
        close = getattr(env, "close", None)
        if callable(close):
            close()
        arr = np.asarray(completed, np.float32)
        ev = {"mean": float(arr.mean()) if arr.size else float("nan"), "n": int(arr.size),
              "valid": bool(len(completed) >= cfg.eval_episodes), "route_acc": rc / max(rt_, 1)}
        scores[g] = ev
        log(f"  inferred[{g}] -> ctx{true_ctx} mean={ev['mean']:.1f} route_acc={ev['route_acc']:.4f} valid={ev['valid']}")
    return {"routing_accuracy": racc, "scores": scores, "num_contexts": mgr.num_contexts}


def _persist(out_dir, cfg, random_scores, best_after_learn, retention_matrix,
             routing_log, resolve_log, game_to_ctx, mgr, inferred=None):
    payload = {
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()},
        "method": "task_free_online_castm",
        "random_scores": random_scores,
        "best_after_learn": best_after_learn,
        "retention_matrix": retention_matrix,
        "resolve_log": resolve_log,
        "routing_log_tail": routing_log[-200:],
        "game_to_ctx_LABEL_ONLY": game_to_ctx,
        "num_contexts": mgr.num_contexts,
        "ctx_ids": list(mgr.ctx_ids),
        "manager_thresholds": {"known": mgr.cfg.known_thresh, "novel": mgr.cfg.novel_thresh},
        "inferred": inferred,
    }
    (Path(out_dir) / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def parse_args() -> TaskFreeConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--games", nargs="+", required=True)
    p.add_argument("--steps-per-game", type=int, default=500_000)
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--eval-episodes", type=int, default=12)
    p.add_argument("--mem-rank", type=int, default=64)
    p.add_argument("--resolve-every", type=int, default=20)
    p.add_argument("--out-dir", type=str, default="castm_runs/taskfree/run")
    p.add_argument("--intervention", type=str, default="boost_reset",
                   choices=["boost_reset", "entropy", "reset", "none"])
    p.add_argument("--no-value", action="store_true", help="ablation: exclude value head from resolve")
    p.add_argument("--no-resolve", action="store_true", help="naive control: disable memory protection")
    p.add_argument("--calibrate", action="store_true", help="legacy fixed-threshold probe (adaptive routing self-calibrates)")
    a = p.parse_args()
    return TaskFreeConfig(games=tuple(a.games), steps_per_game=a.steps_per_game, num_envs=a.num_envs,
                          seed=a.seed, eval_episodes=a.eval_episodes, mem_rank=a.mem_rank,
                          resolve_every=a.resolve_every, out_dir=a.out_dir, intervention=a.intervention,
                          include_value=not a.no_value, resolve=not a.no_resolve, calibrate=a.calibrate)


if __name__ == "__main__":
    run(parse_args())
