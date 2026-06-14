"""Metric helpers for CSN-PPO."""


def prefix_metrics(d, prefix):
    return {f"{prefix}/{k}": v for k, v in d.items()}


def merge_metrics(*dicts):
    out = {}
    for d in dicts:
        if d:
            out.update(d)
    return out


def to_float_dict(d):
    import numpy as np

    return {k: (float(np.asarray(v)) if hasattr(v, "shape") else v) for k, v in d.items()}


def generalization_gap(train_surrogate, holdout_surrogate):
    return train_surrogate - holdout_surrogate


def should_stop_epoch(
    holdout_score,
    best_holdout_score,
    memory_kl_p95,
    memory_kl_limit,
    approx_kl,
    target_kl,
    eps=1e-4,
    enable_kl_early_stop=False,
):
    """README §21 holdout/memory/KL early stop gate."""
    holdout_bad = holdout_score < best_holdout_score - eps
    memory_bad = memory_kl_p95 > memory_kl_limit
    kl_bad = (approx_kl > 1.5 * target_kl) if enable_kl_early_stop else False
    return bool(holdout_bad | memory_bad | kl_bad)
