import jax.numpy as jnp

from agent.csn_ppo import coverage_probes
from agent.csn_ppo import rollout_mining
from agent.csn_ppo.config import CSNPPOConfig
from agent.csn_ppo.guarded_loss import gaussian_kl, memory_guard_loss
from agent.csn_ppo.memory import BehavioralMemoryBatch
from praxis import contract


def _risky_obstacle_probe():
    obstacles = jnp.zeros((contract.K, contract.PER_OBSTACLE_DIM), dtype=jnp.float32)
    obstacles = obstacles.at[0].set(
        jnp.asarray([0.10, 0.0, 0.0, 0.0], dtype=jnp.float32)
    )
    return coverage_probes.pack_cover_obs(
        agent_feat=jnp.zeros((contract.AGENT_DIM,), dtype=jnp.float32),
        obstacles=obstacles,
        frontier=jnp.asarray([1.0, 0.0, 0.8], dtype=jnp.float32),
        covered=jnp.asarray(0.2, dtype=jnp.float32),
        mask=jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
    )


def _open_probe():
    return coverage_probes.pack_cover_obs(
        agent_feat=jnp.zeros((contract.AGENT_DIM,), dtype=jnp.float32),
        obstacles=jnp.zeros((contract.K, contract.PER_OBSTACLE_DIM), dtype=jnp.float32),
        frontier=jnp.asarray([1.0, 0.0, 0.8], dtype=jnp.float32),
        covered=jnp.asarray(0.2, dtype=jnp.float32),
        mask=jnp.zeros((contract.K,), dtype=jnp.float32),
    )


def _memory_batch(mean, logstd, value=None, kl_budget=None):
    batch_size = mean.shape[0]
    if value is None:
        value = jnp.zeros((batch_size,), dtype=jnp.float32)
    if kl_budget is None:
        kl_budget = jnp.zeros((batch_size,), dtype=jnp.float32)
    return BehavioralMemoryBatch(
        obs=jnp.zeros((batch_size, contract.OBS_DIM), dtype=jnp.float32),
        mean=mean,
        logstd=logstd,
        value=value,
        weight=jnp.ones((batch_size,), dtype=jnp.float32),
        kl_budget=kl_budget,
        value_budget=jnp.full((batch_size,), 0.25, dtype=jnp.float32),
        cluster_id=jnp.zeros((batch_size,), dtype=jnp.int32),
        source_id=jnp.zeros((batch_size,), dtype=jnp.int32),
    )


def _risky_policy_teacher(cfg):
    def apply_policy_value(params, normalizer_params, obs):
        del params, normalizer_params
        n = obs.shape[0]
        mean = jnp.broadcast_to(
            jnp.asarray([1.0, 0.0], dtype=jnp.float32),
            (n, contract.ACT_DIM),
        )
        logstd = jnp.full(
            (n, contract.ACT_DIM),
            cfg.teacher_logstd_floor,
            dtype=jnp.float32,
        )
        value = jnp.zeros((n,), dtype=jnp.float32)
        return mean, logstd, value

    return apply_policy_value


def _labeled_analytic_atom(cfg):
    probe_obs = _risky_obstacle_probe()[None, :]
    labeled = rollout_mining.label_probe_atoms(
        probe_obs,
        params={},
        normalizer_params={},
        apply_policy_value=_risky_policy_teacher(cfg),
        cfg=cfg,
    )
    assert labeled.obs.shape[0] == 1
    analytic_mean = coverage_probes.analytic_coverage_teacher(labeled.obs[0], cfg)
    _, chose_analytic = coverage_probes.safer_of_with_choice(
        labeled.obs[0],
        jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        analytic_mean,
        cfg,
    )
    assert bool(chose_analytic)
    return labeled


def test_analytic_chosen_probe_atom_uses_moderate_logstd():
    cfg = CSNPPOConfig()
    labeled = _labeled_analytic_atom(cfg)

    expected = jnp.full(
        (contract.ACT_DIM,),
        cfg.analytic_teacher_logstd,
        dtype=jnp.float32,
    )
    tight = jnp.full(
        (contract.ACT_DIM,),
        cfg.teacher_logstd_floor,
        dtype=jnp.float32,
    )

    assert bool(jnp.allclose(labeled.logstd[0], expected))
    assert not bool(jnp.allclose(labeled.logstd[0], tight))


def test_policy_chosen_probe_atom_uses_moderate_logstd_and_clips_mean():
    cfg = CSNPPOConfig()
    probe_obs = _open_probe()[None, :]

    def apply_extreme_policy(params, normalizer_params, obs):
        del params, normalizer_params
        n = obs.shape[0]
        mean = jnp.broadcast_to(
            jnp.asarray([2.5, -3.0], dtype=jnp.float32),
            (n, contract.ACT_DIM),
        )
        logstd = jnp.full(
            (n, contract.ACT_DIM),
            cfg.teacher_logstd_floor,
            dtype=jnp.float32,
        )
        value = jnp.zeros((n,), dtype=jnp.float32)
        return mean, logstd, value

    labeled = rollout_mining.label_probe_atoms(
        probe_obs,
        params={},
        normalizer_params={},
        apply_policy_value=apply_extreme_policy,
        cfg=cfg,
    )

    expected_logstd = jnp.full(
        (contract.ACT_DIM,),
        cfg.analytic_teacher_logstd,
        dtype=jnp.float32,
    )
    expected_mean = jnp.asarray([1.0, -1.0], dtype=jnp.float32)

    assert bool(jnp.allclose(labeled.mean[0], expected_mean))
    assert bool(jnp.allclose(labeled.logstd[0], expected_logstd))


def test_guard_kl_clips_means_and_floors_policy_logstd_for_ood_atoms():
    cfg = CSNPPOConfig(guard_mean_clip=0.5, guard_min_logstd=-2.3)
    teacher_mean = jnp.zeros((1, contract.ACT_DIM), dtype=jnp.float32)
    tight_logstd = jnp.full_like(teacher_mean, cfg.teacher_logstd_floor)
    policy_mean = jnp.asarray([[1.0e5, -1.0e5]], dtype=jnp.float32)
    policy_logstd = jnp.full_like(teacher_mean, cfg.teacher_logstd_floor)
    memory_batch = _memory_batch(teacher_mean, tight_logstd)

    def apply_policy_value(params, normalizer_params, obs):
        del params, normalizer_params
        return policy_mean, policy_logstd, jnp.zeros((obs.shape[0],), dtype=jnp.float32)

    loss, metrics = memory_guard_loss(
        None,
        None,
        memory_batch,
        apply_policy_value,
        cfg=cfg,
    )
    t_mean = jnp.clip(teacher_mean, -cfg.guard_mean_clip, cfg.guard_mean_clip)
    p_mean = jnp.clip(policy_mean, -cfg.guard_mean_clip, cfg.guard_mean_clip)
    p_logstd = jnp.maximum(policy_logstd, cfg.guard_min_logstd)
    expected_kl = gaussian_kl(
        t_mean,
        tight_logstd,
        p_mean,
        p_logstd,
    )[0]
    wrong_teacher_floored_kl = gaussian_kl(
        t_mean,
        jnp.maximum(tight_logstd, cfg.guard_min_logstd),
        p_mean,
        p_logstd,
    )[0]

    assert bool(jnp.isfinite(loss))
    assert bool(jnp.isfinite(metrics["memory/kl_p95"]))
    assert bool(jnp.allclose(metrics["memory/kl_p95"], expected_kl, rtol=1e-5, atol=1e-5))
    assert float(metrics["memory/kl_p95"]) <= cfg.max_atom_kl
    assert float(metrics["diag/meandiff_max"]) > cfg.guard_mean_clip
    assert not bool(
        jnp.allclose(
            metrics["memory/kl_p95"],
            wrong_teacher_floored_kl,
            rtol=1e-5,
            atol=1e-5,
        )
    )


def test_guard_kl_for_far_analytic_atom_is_clamped_and_finite():
    cfg = CSNPPOConfig()
    labeled = _labeled_analytic_atom(cfg)
    far_mean = jnp.broadcast_to(
        jnp.asarray([10.0, -10.0], dtype=jnp.float32),
        labeled.mean.shape,
    )
    moderate_logstd = jnp.full_like(labeled.logstd, cfg.analytic_teacher_logstd)

    def apply_far_policy(params, normalizer_params, obs):
        del params, normalizer_params
        return far_mean, moderate_logstd, jnp.zeros((obs.shape[0],), dtype=jnp.float32)

    loss, metrics = memory_guard_loss(
        None,
        None,
        labeled,
        apply_far_policy,
        cfg=cfg,
    )
    bounded_kl = metrics["memory/kl_p95"]
    tight_logstd = jnp.full_like(labeled.logstd, cfg.teacher_logstd_floor)
    tight_raw_kl = gaussian_kl(labeled.mean, tight_logstd, far_mean, tight_logstd)

    assert bool(jnp.isfinite(loss))
    assert bool(jnp.isfinite(bounded_kl))
    assert float(bounded_kl) <= cfg.max_atom_kl
    assert float(bounded_kl) < float(tight_raw_kl[0]) * 1.0e-3
