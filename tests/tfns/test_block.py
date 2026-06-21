from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import optax

import tfns.train.block as block_mod
from tfns.config import (
    AdapterConfig,
    AuxConfig,
    BehaviorConfig,
    CreditConfig,
    MemoryConfig,
    ModelConfig,
    PPOConfig,
    ReplayConfig,
    TFNSConfig,
)
from tfns.consolidate.state import ContinualState
from tfns.detect.change import PageHinkleyDetector
from tfns.memory.bank import SequenceMemoryBank
from tfns.memory.record import frames_from_obs, make_record
from tfns.model.agent import RecurrentAgent
from tfns.protect.projection import build_protected_modules
from tfns.train.block import replay_tube_loss, train_block


def _cfg() -> TFNSConfig:
    return TFNSConfig(
        model=ModelConfig(
            act_dim=3,
            conv_channels=(2, 2, 2),
            dense_dim=16,
            action_embed_dim=4,
            gru_hidden=8,
            key_dim=128,
            ema_decay=0.5,
        ),
        adapter=AdapterConfig(num_adapters=2, rank=2, top_k=1),
        aux=AuxConfig(aux_coef=0.0),
        ppo=PPOConfig(num_envs=2, rollout_len=4, update_epochs=1, seq_chunk=2, max_grad_norm=0.25),
        replay=ReplayConfig(seq_len=4, burn_in=1, protected_region=3, replay_frac_start=0.25),
        memory=MemoryConfig(byte_budget=1 << 24, min_per_cluster=0),
        behavior=BehaviorConfig(kl_tol=1.0e6, value_tol=1.0e6, key_cos_tol=1.0e6),
        credit=CreditConfig(predictor_val_windows=1),
    )


class TinyEnv:
    def __init__(self, num_envs: int, act_dim: int):
        self.num_envs = int(num_envs)
        self.act_dim = int(act_dim)
        self.step_count = 0
        self.obs = np.zeros((self.num_envs, 84, 84, 4), dtype=np.uint8)
        for env in range(self.num_envs):
            for channel in range(4):
                self.obs[env, :, :, channel] = (env * 17 + channel * 11) % 255

    def __call__(self, action):
        action = np.asarray(action, dtype=np.int32)
        self.step_count += 1
        new_frame = np.zeros((self.num_envs, 84, 84), dtype=np.uint8)
        for env in range(self.num_envs):
            value = (self.step_count * 13 + env * 19 + int(action[env]) * 23) % 255
            new_frame[env, :, :] = value
        self.obs = np.concatenate([self.obs[..., 1:], new_frame[..., None]], axis=-1)
        reward = (action.astype(np.float32) - 1.0) * 0.1
        ppo_done = np.zeros((self.num_envs,), dtype=np.bool_)
        reset = np.zeros((self.num_envs,), dtype=np.bool_)
        return self.obs, reward, ppo_done, reset, {}


def _init_state(agent: RecurrentAgent, cfg: TFNSConfig, env: TinyEnv, tx) -> ContinualState:
    obs = jnp.asarray(env.obs)
    prev_action = jnp.zeros((env.num_envs,), dtype=jnp.int32)
    prev_reward = jnp.zeros((env.num_envs,), dtype=jnp.float32)
    reset = jnp.ones((env.num_envs,), dtype=bool)
    hidden = agent.init_hidden(env.num_envs)
    params = agent.init(jax.random.PRNGKey(3), obs, prev_action, prev_reward, reset, hidden)["params"]
    return ContinualState(
        params=params,
        opt_state=tx.init(params),
        ema_params=params,
        bases={},
        memory=SequenceMemoryBank(cfg.memory),
        predictor_params=None,
        predictor_opt_state=None,
        detector_state=PageHinkleyDetector(cfg.detect).init(),
        adapter_dormant=jnp.ones((agent.adapter_config.num_adapters,), dtype=bool),
        robust_stats={},
        protected_clusters=[],
        rng=jax.random.PRNGKey(11),
        block_index=0,
        skills={},
        rollout_carry=None,
    )


def _tree_norm(tree) -> float:
    leaves = jax.tree_util.tree_leaves(tree)
    total = sum(float(jnp.sum(jnp.square(jnp.asarray(leaf, dtype=jnp.float32)))) for leaf in leaves)
    return float(np.sqrt(total))


def _tree_delta(new, old):
    return jax.tree_util.tree_map(lambda a, b: a - b, new, old)


def _tree_changed(a, b) -> bool:
    return _tree_norm(_tree_delta(a, b)) > 1.0e-8


def _overlap_obs(t: int) -> np.ndarray:
    frames = np.arange((t + 4) * 84 * 84, dtype=np.uint8).reshape((t + 4, 84, 84))
    return np.stack([np.moveaxis(frames[i + 1 : i + 5], 0, -1) for i in range(t)], axis=0)


def _memory_record_from_current(agent: RecurrentAgent, params, cfg: TFNSConfig):
    t = int(cfg.replay.seq_len)
    obs = _overlap_obs(t)
    obs_seq = jnp.asarray(obs[:, None, ...])
    act_seq = (jnp.arange(t, dtype=jnp.int32) % cfg.model.act_dim)[:, None]
    rew_seq = jnp.linspace(-0.1, 0.1, t, dtype=jnp.float32)[:, None]
    reset_seq = jnp.zeros((t, 1), dtype=bool).at[0, 0].set(True)
    h0 = agent.init_hidden(1)
    outputs, _ = agent.unroll(params, obs_seq, act_seq, rew_seq, reset_seq, h0)
    init_stack, new_frames = frames_from_obs(obs)
    key = np.asarray(outputs.q_key[:, 0], dtype=np.float32)
    return make_record(
        init_stack=init_stack,
        new_frames=new_frames,
        actions=np.asarray(act_seq[:, 0], dtype=np.int32),
        rewards_clipped=np.asarray(rew_seq[:, 0], dtype=np.float32),
        rewards_raw=np.asarray(rew_seq[:, 0], dtype=np.float32),
        ppo_mask=np.zeros((t,), dtype=np.bool_),
        reset_mask=np.asarray(reset_seq[:, 0], dtype=np.bool_),
        teacher_logits=np.pad(
            np.asarray(outputs.logits[:, 0], dtype=np.float32),
            ((0, 0), (0, 18 - cfg.model.act_dim)),
        ),
        teacher_value=np.asarray(outputs.value[:, 0], dtype=np.float32),
        key_anchor=key,
        causal_contrib=np.zeros((t,), dtype=np.float32),
        credit_trace=np.zeros((t,), dtype=np.float32),
        adv_mag=np.zeros((t,), dtype=np.float32),
        td_mag=np.zeros((t,), dtype=np.float32),
        surprise=np.zeros((t,), dtype=np.float32),
        teacher_entropy=np.zeros((t,), dtype=np.float32),
        episode_id=1,
        chunk_index=0,
    )


def _sentinel_cluster(agent: RecurrentAgent, params, cfg: TFNSConfig):
    t = int(cfg.replay.seq_len)
    obs_seq = jnp.asarray(_overlap_obs(t)[:, None, ...])
    act_seq = (jnp.arange(t, dtype=jnp.int32) % cfg.model.act_dim)[:, None]
    rew_seq = jnp.zeros((t, 1), dtype=jnp.float32)
    reset_seq = jnp.zeros((t, 1), dtype=bool).at[0, 0].set(True)
    h0 = agent.init_hidden(1)
    outputs, _ = agent.unroll(params, obs_seq, act_seq, rew_seq, reset_seq, h0)
    return {
        "obs_seq": obs_seq,
        "act_seq": act_seq,
        "rew_seq": rew_seq,
        "reset_seq": reset_seq,
        "h0": h0,
        "burn_in": 1,
        "teacher_logits": outputs.logits,
        "teacher_value": outputs.value,
        "teacher_key": outputs.q_key,
    }


def _orthonormal(key, rows: int, rank: int) -> jnp.ndarray:
    q, _ = jnp.linalg.qr(jax.random.normal(key, (rows, rank), dtype=jnp.float32), mode="reduced")
    return q[:, :rank]


def _policy_head_null_norm(delta, U) -> float:
    kernel = delta["policy_head"]["affine"]["kernel"]
    bias = delta["policy_head"]["affine"]["bias"]
    kbar = jnp.concatenate([kernel, bias[None, :]], axis=0)
    return float(jnp.linalg.norm(U.T @ kbar))


def test_train_block_empty_protection_runs_and_admits_memory():
    cfg = _cfg()
    agent = RecurrentAgent(model_config=cfg.model, adapter_config=cfg.adapter)
    env = TinyEnv(num_envs=cfg.ppo.num_envs, act_dim=cfg.model.act_dim)
    tx = optax.adam(1.0e-3)
    state = _init_state(agent, cfg, env, tx)
    params_before = state.params
    ema_before = state.ema_params

    state, telemetry = train_block(state, agent, tx, env, cfg)

    numeric = [value for value in telemetry.values() if isinstance(value, (int, float))]
    assert all(np.isfinite(float(value)) for value in numeric)
    assert telemetry["accept_count"] >= 1
    assert _tree_changed(state.params, params_before)
    assert len(state.memory) >= 1
    assert _tree_changed(state.ema_params, ema_before)
    assert _tree_norm(_tree_delta(state.ema_params, ema_before)) < _tree_norm(
        _tree_delta(state.params, params_before)
    )


def test_train_block_uses_ppo_only_with_only_transient_memory(monkeypatch):
    cfg = _cfg()
    agent = RecurrentAgent(model_config=cfg.model, adapter_config=cfg.adapter)
    env = TinyEnv(num_envs=cfg.ppo.num_envs, act_dim=cfg.model.act_dim)
    tx = optax.adam(1.0e-3)
    state = _init_state(agent, cfg, env, tx)
    state, _ = train_block(state, agent, tx, env, cfg)
    assert len(state.memory) >= 1
    assert all(rec.status == "transient" for rec in state.memory.records())

    calls = {"ppo_only": 0}
    original_ppo_only = block_mod._grad_step_ppo_only

    def fail_replay_path(*args, **kwargs):
        raise AssertionError("transient records must not enter replay tube loss")

    def count_ppo_only(*args, **kwargs):
        calls["ppo_only"] += 1
        return original_ppo_only(*args, **kwargs)

    monkeypatch.setattr(block_mod, "_grad_step", fail_replay_path)
    monkeypatch.setattr(block_mod, "_grad_step_ppo_only", count_ppo_only)

    state, telemetry = train_block(state, agent, tx, env, cfg)

    assert calls["ppo_only"] >= 1
    assert telemetry["replay_tube_mean"] == 0.0
    assert telemetry["replay_tube_tail"] == 0.0
    assert telemetry["replay_tube_total"] == 0.0


def test_train_block_protected_path_respects_basis_null_space_and_replay_zero():
    cfg = _cfg()
    agent = RecurrentAgent(model_config=cfg.model, adapter_config=cfg.adapter)
    env = TinyEnv(num_envs=cfg.ppo.num_envs, act_dim=cfg.model.act_dim)
    tx = optax.adam(1.0e-3)
    state = _init_state(agent, cfg, env, tx)
    state, _ = train_block(state, agent, tx, env, cfg)

    rec = _memory_record_from_current(agent, state.params, cfg)
    replay_loss, replay_aux = replay_tube_loss(state.params, agent, [rec], cfg)
    assert float(replay_loss) <= 1.0e-6
    assert float(replay_aux["total"]) <= 1.0e-6

    modules = build_protected_modules(state.params, cfg.model)
    U = _orthonormal(jax.random.PRNGKey(17), modules["policy_head"].d_aug, 1)
    state.bases = {"policy_head": U}
    sentinel = _sentinel_cluster(agent, state.params, cfg)
    before = state.params

    state, telemetry = train_block(state, agent, tx, env, cfg, sentinel_clusters=[sentinel])

    assert telemetry["accept_count"] >= 1
    delta = _tree_delta(state.params, before)
    assert _policy_head_null_norm(delta, U) <= 1.0e-4
