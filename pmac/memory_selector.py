"""Importance-based anchor selection from PMA-C section 13."""

from __future__ import annotations

import numpy as np


def _softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _rng_from_key(key):
    if isinstance(key, np.random.Generator):
        return key
    if key is None:
        return np.random.default_rng()
    arr = np.asarray(key, dtype=np.uint32).reshape(-1)
    seed = 0
    for value in arr:
        seed = (1664525 * seed + int(value) + 1013904223) % (2**32)
    return np.random.default_rng(seed)


def importance_scores(logits, labels, alphas=(1, 1, 1, 1, 1, 1)) -> np.ndarray:
    """Return q(x)=a1 I + a2 N + a3 B + a4 U + a5 F + a6 R."""

    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    probs = _softmax(logits)
    n = labels.shape[0]
    row = np.arange(n)

    ce = -np.log(np.clip(probs[row, labels], 1e-12, 1.0))

    sorted_logits = np.sort(logits, axis=-1)
    margin = sorted_logits[:, -1] - sorted_logits[:, -2]
    boundary = -margin

    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=-1)
    failure = (np.argmax(logits, axis=-1) != labels).astype(np.float64)

    norms = np.linalg.norm(logits, axis=-1)
    novelty = (norms - np.mean(norms)) / (np.std(norms) + 1e-8)

    counts = np.bincount(labels, minlength=int(np.max(labels)) + 1).astype(np.float64)
    rarity = 1.0 / np.maximum(counts[labels], 1.0)
    rarity = rarity / (np.mean(rarity) + 1e-8)

    a = np.asarray(alphas, dtype=np.float64)
    if a.shape[0] != 6:
        raise ValueError("alphas must contain six weights")
    scores = a[0] * ce + a[1] * novelty + a[2] * boundary
    scores = scores + a[3] * entropy + a[4] * failure + a[5] * rarity
    return scores.astype(np.float32)


def select_indices(logits, labels, n_select, mode="importance", key=None) -> np.ndarray:
    logits = np.asarray(logits)
    n = logits.shape[0]
    n_select = min(int(n_select), n)
    if n_select <= 0:
        return np.array([], dtype=np.int64)
    if mode == "importance":
        scores = importance_scores(logits, labels)
        order = np.argsort(-scores, kind="mergesort")
        return order[:n_select].astype(np.int64)
    if mode == "random":
        rng = _rng_from_key(key)
        return rng.choice(n, size=n_select, replace=False).astype(np.int64)
    raise ValueError(f"unknown memory selection mode: {mode}")


__all__ = ["importance_scores", "select_indices"]
