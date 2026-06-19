"""CPU-resident episodic sequence records; no game/task identity is stored.

The stored observation representation is lossless and frame efficient:
``init_stack`` contains the four frames preceding the first stored transition
and ``new_frames`` contains one new grayscale frame per transition. Optional
zlib compression applies only to those two uint8 frame fields. When compressed,
the same dataclass fields hold compressed ``bytes``; all non-frame arrays remain
plain numpy arrays. ``nbytes`` reports the stored representation exactly by
counting compressed byte lengths for frame bytes and ``.nbytes`` for arrays.
"""

from __future__ import annotations

import dataclasses
import zlib
from typing import Any

import numpy as np

FRAME_STACK = 4
OBS_HW = 84
ACT_DIM = 18
KEY_DIM = 128

VALID_STATUSES = frozenset(
    {
        "transient",
        "candidate",
        "protected",
        "failure_recovery",
        "deletion_pending",
    }
)


@dataclasses.dataclass
class EpisodeSequence:
    init_stack: np.ndarray | bytes
    new_frames: np.ndarray | bytes
    actions: np.ndarray
    rewards_clipped: np.ndarray
    rewards_raw: np.ndarray
    ppo_mask: np.ndarray
    reset_mask: np.ndarray
    teacher_logits: np.ndarray
    teacher_value: np.ndarray
    key_anchor: np.ndarray
    causal_contrib: np.ndarray
    credit_trace: np.ndarray
    adv_mag: np.ndarray
    td_mag: np.ndarray
    surprise: np.ndarray
    teacher_entropy: np.ndarray
    seq_importance: float = 0.0
    cluster_id: int = -1
    episode_id: int = 0
    chunk_index: int = 0
    status: str = "transient"
    is_sentinel: bool = False
    frames_compressed: bool = False


def seq_len(rec: EpisodeSequence) -> int:
    return int(rec.actions.shape[0])


def _as_array(name: str, value: Any, dtype: np.dtype, shape: tuple[int | None, ...]) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    if arr.ndim != len(shape):
        raise ValueError(f"{name} must have {len(shape)} dimensions, got {arr.ndim}.")
    for axis, expected in enumerate(shape):
        if expected is not None and arr.shape[axis] != expected:
            raise ValueError(f"{name} shape must be {shape}, got {arr.shape}.")
    if arr.dtype != np.dtype(dtype):
        raise TypeError(f"{name} must have dtype {np.dtype(dtype)}, got {arr.dtype}.")
    return np.ascontiguousarray(arr)


def _validate_status(status: str) -> str:
    if status not in VALID_STATUSES:
        raise ValueError(f"status must be one of {sorted(VALID_STATUSES)}, got {status!r}.")
    return status


def make_record(
    *,
    init_stack: Any,
    new_frames: Any,
    actions: Any,
    rewards_clipped: Any,
    ppo_mask: Any,
    reset_mask: Any,
    teacher_logits: Any,
    teacher_value: Any,
    key_anchor: Any,
    causal_contrib: Any,
    credit_trace: Any,
    adv_mag: Any,
    td_mag: Any,
    surprise: Any,
    teacher_entropy: Any,
    episode_id: int,
    chunk_index: int,
    rewards_raw: Any | None = None,
    seq_importance: float = 0.0,
    cluster_id: int = -1,
    status: str = "transient",
    is_sentinel: bool = False,
) -> EpisodeSequence:
    """Build a validated CPU numpy sequence record."""

    init_arr = _as_array("init_stack", init_stack, np.uint8, (FRAME_STACK, OBS_HW, OBS_HW))
    frames_arr = _as_array("new_frames", new_frames, np.uint8, (None, OBS_HW, OBS_HW))
    t = int(frames_arr.shape[0])
    if t <= 0:
        raise ValueError("new_frames must contain at least one transition frame.")

    if rewards_raw is None:
        rewards_raw = np.zeros((t,), dtype=np.float32)

    rec = EpisodeSequence(
        init_stack=init_arr,
        new_frames=frames_arr,
        actions=_as_array("actions", actions, np.int32, (t,)),
        rewards_clipped=_as_array("rewards_clipped", rewards_clipped, np.float32, (t,)),
        rewards_raw=_as_array("rewards_raw", rewards_raw, np.float32, (t,)),
        ppo_mask=_as_array("ppo_mask", ppo_mask, np.bool_, (t,)),
        reset_mask=_as_array("reset_mask", reset_mask, np.bool_, (t,)),
        teacher_logits=_as_array("teacher_logits", teacher_logits, np.float32, (t, ACT_DIM)),
        teacher_value=_as_array("teacher_value", teacher_value, np.float32, (t,)),
        key_anchor=_as_array("key_anchor", key_anchor, np.float32, (t, KEY_DIM)),
        causal_contrib=_as_array("causal_contrib", causal_contrib, np.float32, (t,)),
        credit_trace=_as_array("credit_trace", credit_trace, np.float32, (t,)),
        adv_mag=_as_array("adv_mag", adv_mag, np.float32, (t,)),
        td_mag=_as_array("td_mag", td_mag, np.float32, (t,)),
        surprise=_as_array("surprise", surprise, np.float32, (t,)),
        teacher_entropy=_as_array("teacher_entropy", teacher_entropy, np.float32, (t,)),
        seq_importance=float(seq_importance),
        cluster_id=int(cluster_id),
        episode_id=int(episode_id),
        chunk_index=int(chunk_index),
        status=_validate_status(status),
        is_sentinel=bool(is_sentinel),
        frames_compressed=False,
    )
    return rec


def reconstruct_obs(rec: EpisodeSequence) -> np.ndarray:
    """Return exact NHWC four-frame observations for every transition."""

    dec = decompress(rec)
    init_stack = np.asarray(dec.init_stack, dtype=np.uint8)
    new_frames = np.asarray(dec.new_frames, dtype=np.uint8)
    frames = np.concatenate([init_stack, new_frames], axis=0)
    t = seq_len(dec)
    obs = np.empty((t, OBS_HW, OBS_HW, FRAME_STACK), dtype=np.uint8)
    for idx in range(t):
        obs[idx] = np.moveaxis(frames[idx + 1 : idx + 1 + FRAME_STACK], 0, -1)
    return obs


def nbytes(rec: EpisodeSequence) -> int:
    """Return exact bytes for the stored frame representation and arrays."""

    total = 0
    for value in (rec.init_stack, rec.new_frames):
        if isinstance(value, bytes):
            total += len(value)
        else:
            total += int(np.asarray(value).nbytes)

    for value in (
        rec.actions,
        rec.rewards_clipped,
        rec.rewards_raw,
        rec.ppo_mask,
        rec.reset_mask,
        rec.teacher_logits,
        rec.teacher_value,
        rec.key_anchor,
        rec.causal_contrib,
        rec.credit_trace,
        rec.adv_mag,
        rec.td_mag,
        rec.surprise,
        rec.teacher_entropy,
    ):
        total += int(np.asarray(value).nbytes)
    return total


def compress(rec: EpisodeSequence) -> EpisodeSequence:
    """Return a record with only frame arrays zlib-compressed."""

    if rec.frames_compressed:
        return rec
    init_stack = np.asarray(rec.init_stack, dtype=np.uint8)
    new_frames = np.asarray(rec.new_frames, dtype=np.uint8)
    return dataclasses.replace(
        rec,
        init_stack=zlib.compress(np.ascontiguousarray(init_stack).tobytes()),
        new_frames=zlib.compress(np.ascontiguousarray(new_frames).tobytes()),
        frames_compressed=True,
    )


def decompress(rec: EpisodeSequence) -> EpisodeSequence:
    """Return a record with frame fields restored to uint8 numpy arrays."""

    if not rec.frames_compressed:
        return rec
    t = seq_len(rec)
    init = np.frombuffer(zlib.decompress(rec.init_stack), dtype=np.uint8).reshape(
        FRAME_STACK, OBS_HW, OBS_HW
    )
    frames = np.frombuffer(zlib.decompress(rec.new_frames), dtype=np.uint8).reshape(t, OBS_HW, OBS_HW)
    return dataclasses.replace(
        rec,
        init_stack=np.ascontiguousarray(init),
        new_frames=np.ascontiguousarray(frames),
        frames_compressed=False,
    )


__all__ = [
    "ACT_DIM",
    "FRAME_STACK",
    "KEY_DIM",
    "OBS_HW",
    "EpisodeSequence",
    "VALID_STATUSES",
    "compress",
    "decompress",
    "make_record",
    "nbytes",
    "reconstruct_obs",
    "seq_len",
]
