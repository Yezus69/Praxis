from __future__ import annotations

import dataclasses
import inspect
import re

import numpy as np

from tfns.config import MemoryConfig, ReplayConfig, RiskConfig
from tfns.memory.bank import DeletionCertificate, SequenceMemoryBank, can_delete_sentinel
from tfns.memory.record import EpisodeSequence, compress, make_record, nbytes, reconstruct_obs
from tfns.memory.sampling import (
    burn_in_split,
    cluster_probs,
    cluster_risk,
    replay_transition_count,
    sample_sequences,
    split_reconstructed_obs,
)


def _signature(index: int, scale: float = 1.0) -> np.ndarray:
    vec = np.zeros((128,), dtype=np.float32)
    vec[index % 128] = scale
    return vec


def _record(
    *,
    t: int = 1,
    signature: np.ndarray | None = None,
    episode_id: int = 1,
    chunk_index: int = 0,
    causal: float | np.ndarray = 0.0,
    adv: float | np.ndarray = 0.0,
    td: float | np.ndarray = 0.0,
    surprise: float | np.ndarray = 0.0,
    entropy: float | np.ndarray = 0.0,
    ppo_mask: bool | np.ndarray = False,
    init_stack: np.ndarray | None = None,
    new_frames: np.ndarray | None = None,
) -> EpisodeSequence:
    if init_stack is None:
        init_stack = np.zeros((4, 84, 84), dtype=np.uint8)
    if new_frames is None:
        new_frames = np.zeros((t, 84, 84), dtype=np.uint8)
    signature = _signature(0) if signature is None else signature.astype(np.float32)
    key_anchor = np.repeat(signature[None, :], t, axis=0).astype(np.float32)

    def _series(value, dtype=np.float32):
        arr = np.asarray(value, dtype=dtype)
        if arr.ndim == 0:
            arr = np.full((t,), arr.item(), dtype=dtype)
        return arr

    return make_record(
        init_stack=init_stack,
        new_frames=new_frames,
        actions=np.arange(t, dtype=np.int32) % 18,
        rewards_clipped=np.zeros((t,), dtype=np.float32),
        rewards_raw=np.zeros((t,), dtype=np.float32),
        ppo_mask=_series(ppo_mask, np.bool_),
        reset_mask=np.zeros((t,), dtype=np.bool_),
        teacher_logits=np.zeros((t, 18), dtype=np.float32),
        teacher_value=np.zeros((t,), dtype=np.float32),
        key_anchor=key_anchor,
        causal_contrib=_series(causal),
        credit_trace=np.zeros((t,), dtype=np.float32),
        adv_mag=_series(adv),
        td_mag=_series(td),
        surprise=_series(surprise),
        teacher_entropy=_series(entropy),
        episode_id=episode_id,
        chunk_index=chunk_index,
    )


def test_lossless_obs_reconstruction():
    t = 5
    init_stack = np.arange(4 * 84 * 84, dtype=np.uint8).reshape(4, 84, 84)
    new_frames = (np.arange(t * 84 * 84, dtype=np.uint8).reshape(t, 84, 84) + 17).astype(np.uint8)
    rec = _record(t=t, init_stack=init_stack, new_frames=new_frames)

    frames = np.concatenate([init_stack, new_frames], axis=0)
    expected = np.stack([np.moveaxis(frames[i + 1 : i + 5], 0, -1) for i in range(t)], axis=0)
    np.testing.assert_array_equal(reconstruct_obs(rec), expected)
    np.testing.assert_array_equal(reconstruct_obs(compress(rec)), expected)


def test_byte_budget_never_exceeded():
    base = _record(signature=_signature(3))
    cfg = MemoryConfig(byte_budget=nbytes(base) * 3, min_per_cluster=1)
    bank = SequenceMemoryBank(cfg)

    for i in range(20):
        assert bank.add(_record(signature=_signature(3), episode_id=i, causal=float(i)))
        assert bank.bytes_used() <= cfg.byte_budget

    assert len(bank) >= 1


def test_burst_cannot_evict_sole_old_context_rep():
    old = _record(signature=_signature(0), episode_id=1)
    cfg = MemoryConfig(byte_budget=nbytes(old) * 3, min_per_cluster=2)
    bank = SequenceMemoryBank(cfg)
    assert bank.add(old)
    old_cluster = old.cluster_id

    for i in range(20):
        bank.add(_record(signature=_signature(1), episode_id=100 + i, causal=50.0, adv=50.0, td=50.0))
        assert bank.bytes_used() <= cfg.byte_budget

    surviving_old = sum(1 for rec in bank.records() if rec.cluster_id == old_cluster)
    assert surviving_old >= min(cfg.min_per_cluster, 1)


def test_protected_sentinel_needs_deletion_cert():
    rec = _record(signature=_signature(0))
    cfg = MemoryConfig(byte_budget=nbytes(rec) * 3, min_per_cluster=0)
    bank = SequenceMemoryBank(cfg)

    sentinel = _record(signature=_signature(0), episode_id=10)
    sentinel.is_sentinel = True
    sentinel.status = "protected"
    other = _record(signature=_signature(0), episode_id=11)
    assert bank.add(sentinel)
    assert bank.add(other)

    bank.config = dataclasses.replace(cfg, byte_budget=1)
    assert not bank.evict_to_fit()
    assert any(item is sentinel for item in bank.records())

    partial = DeletionCertificate(behavior_covered_by_other_sentinels=True)
    full = DeletionCertificate(
        behavior_covered_by_other_sentinels=True,
        activation_directions_covered_by_retained_bases=True,
        held_out_conservation_not_worsened=True,
        closed_loop_retention_within_threshold=True,
    )
    assert not can_delete_sentinel(sentinel, None)
    assert not can_delete_sentinel(sentinel, partial)
    assert can_delete_sentinel(sentinel, full)

    assert bank.evict_to_fit(sentinel_certs={id(sentinel): full})
    assert all(item is not sentinel for item in bank.records())


def test_admission_formula_and_spike_sensitive_sequence_score():
    bank = SequenceMemoryBank(MemoryConfig())
    low = _record(t=4, causal=0.0, adv=0.0, td=0.0, surprise=0.0, entropy=0.0)
    high = _record(t=4, causal=10.0, adv=0.0, td=0.0, surprise=0.0, entropy=0.0)

    assert np.all(bank.transition_importance(high, drift=1.0) > bank.transition_importance(low, drift=0.0))

    spike_values = np.zeros((10,), dtype=np.float32)
    spike_values[5] = 20.0
    spike = _record(t=10, causal=spike_values)
    importance = bank.transition_importance(spike)
    score = bank.sequence_score(spike)
    assert score > float(np.mean(importance))


def test_eviction_utility_orientation_and_age_penalty():
    rec = _record(signature=_signature(0))
    cfg = MemoryConfig(byte_budget=nbytes(rec) * 10, min_per_cluster=0, lam_red=2.0, lam_age=2.0)
    bank = SequenceMemoryBank(cfg)

    old_redundant = _record(signature=_signature(0), episode_id=1)
    duplicate = _record(signature=_signature(0), episode_id=2)
    unique = _record(signature=_signature(2), episode_id=3, causal=5.0)
    assert bank.add(old_redundant)
    assert bank.add(duplicate)
    assert bank.add(unique)
    old_redundant.seq_importance = 0.0
    duplicate.seq_importance = 0.0
    unique.seq_importance = 1.0

    utilities = bank.eviction_utilities()
    records = bank.records()
    old_idx = next(i for i, item in enumerate(records) if item is old_redundant)
    unique_idx = next(i for i, item in enumerate(records) if item is unique)

    assert utilities[old_idx] < utilities[unique_idx]
    assert bank.age_penalty(unique) == 0.0


def test_no_label_invariance():
    allowed_internal = {"cluster_id", "episode_id"}
    forbidden_field = re.compile(r"game|task|env|onehot|curriculum|(^|_)id($|_)", re.IGNORECASE)
    for field in dataclasses.fields(EpisodeSequence):
        if field.name in allowed_internal:
            continue
        assert not forbidden_field.search(field.name)

    forbidden_api = re.compile(r"game|task|env|onehot|curriculum|(^|_)id($|_)", re.IGNORECASE)
    for fn in (SequenceMemoryBank.add, SequenceMemoryBank.evict_to_fit, sample_sequences):
        for param in inspect.signature(fn).parameters:
            if param == "self":
                continue
            assert not forbidden_api.search(param)


def test_clustering_is_online_and_bounded():
    rec = _record(signature=_signature(0))
    cfg = MemoryConfig(
        byte_budget=nbytes(rec) * 20,
        min_per_cluster=0,
        cluster_sim_thresh=0.95,
        cluster_merge_thresh=0.99,
        max_clusters=3,
    )
    bank = SequenceMemoryBank(cfg)

    for i in range(8):
        assert bank.add(_record(signature=_signature(i), episode_id=i))
        assert len(bank.clusters()) <= cfg.max_clusters

    similar_bank = SequenceMemoryBank(cfg)
    first = _record(signature=_signature(10), episode_id=1)
    almost_first = _record(signature=_signature(10) + 0.01 * _signature(11), episode_id=2)
    assert similar_bank.add(first)
    assert similar_bank.add(almost_first)
    assert first.cluster_id == almost_first.cluster_id


def test_sampling_helpers_are_risk_based_and_label_free():
    risk_cfg = RiskConfig(rho_0=0.1, lam_D=1.0, lam_Q=2.0, lam_R=3.0, lam_A=4.0)
    stats = {
        "behavior_violation": 1.0,
        "high_quantile_drift": 2.0,
        "basis_residual": 3.0,
        "time_since_replay": 4.0,
    }
    assert np.isclose(cluster_risk(stats, risk_cfg), 30.1)
    probs = cluster_probs({10: 1.0, 20: 3.0})
    assert probs[20] == 0.75
    assert replay_transition_count(100, 0.0, ReplayConfig()) == 25
    assert replay_transition_count(100, 100.0, ReplayConfig()) <= 100

    rec = _record(t=64, signature=_signature(4))
    bank = SequenceMemoryBank(MemoryConfig(byte_budget=nbytes(rec) * 2, min_per_cluster=0))
    assert bank.add(rec)
    sampled = sample_sequences(bank, np.random.default_rng(0), 3)
    assert all(item is rec for item in sampled)

    burn_slice, protected_slice = burn_in_split(rec, 16, 64)
    assert (burn_slice.start, burn_slice.stop) == (0, 16)
    assert (protected_slice.start, protected_slice.stop) == (16, 64)
    burn_obs, protected_obs = split_reconstructed_obs(rec, 16, 64)
    assert burn_obs.shape == (16, 84, 84, 4)
    assert protected_obs.shape == (48, 84, 84, 4)
