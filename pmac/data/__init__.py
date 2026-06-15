"""Data streams for PMA-C experiments."""

from pmac.data.streams import (
    Task,
    build_stream,
    iterate_minibatches,
    load_mnist,
    make_permuted_mnist,
    make_split_mnist,
    make_synthetic_permuted,
)

__all__ = [
    "Task",
    "load_mnist",
    "make_permuted_mnist",
    "make_split_mnist",
    "make_synthetic_permuted",
    "build_stream",
    "iterate_minibatches",
]
