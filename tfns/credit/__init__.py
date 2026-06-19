"""Delayed-return prediction, causal credit, and safe shaping helpers."""

from tfns.credit.credit import (
    causal_decomposition,
    eligibility_trace,
    potential_shaping,
    shaping_enabled,
    shaping_eta,
    telescoping_residual,
)
from tfns.credit.predictor import (
    ReturnPredictor,
    discounted_returns,
    make_predictor_optimizer,
    predictor_loss,
    train_step,
    validate,
)

__all__ = [
    "ReturnPredictor",
    "causal_decomposition",
    "discounted_returns",
    "eligibility_trace",
    "make_predictor_optimizer",
    "potential_shaping",
    "predictor_loss",
    "shaping_enabled",
    "shaping_eta",
    "telescoping_residual",
    "train_step",
    "validate",
]
