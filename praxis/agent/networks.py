"""praxis/agent/networks.py — Brax PPO network factory wiring.

Builds the actor-critic network factory consumed by ``brax.training.agents.ppo.train``.

CORRECTED FACTS (see agent/README.md):
  * Brax PPO uses **separate** policy and value MLPs (NOT a shared trunk).
  * Brax DEFAULT hidden sizes are policy=(32,)*4, value=(256,)*5 — NOT what we want.
    We MUST override to policy=(256,256,256) / value=(256,256,256) for the nav task.

We expose ``make_network_factory`` which returns a ``functools.partial`` over
``ppo_networks.make_ppo_networks`` with the hidden sizes baked in. ``ppo.train``
calls that partial internally with the observation/action sizes it discovers from
the (wrapped) environment, so we deliberately do NOT pass obs/act dims here.
"""

from __future__ import annotations

import functools
from typing import Sequence

# NOTE: brax is imported lazily inside make_network_factory (not at module top) so
# `from praxis.agent.networks import make_network_factory` is importable on hosts
# without brax/jax installed (e.g. for --help / argparse smoke). The import resolves
# at train time, inside the container, exactly when the factory is built.

# Defaults for this task. Kept in one place so train.py and tests agree.
DEFAULT_POLICY_HIDDEN_LAYER_SIZES = (256, 256, 256)
DEFAULT_VALUE_HIDDEN_LAYER_SIZES = (256, 256, 256)


def make_network_factory(
    policy_sizes: Sequence[int] = DEFAULT_POLICY_HIDDEN_LAYER_SIZES,
    value_sizes: Sequence[int] = DEFAULT_VALUE_HIDDEN_LAYER_SIZES,
) -> functools.partial:
    """Return a ``functools.partial`` over ``make_ppo_networks``.

    The returned partial is what ``ppo.train(network_factory=...)`` expects: a
    callable that ``ppo.train`` invokes as
    ``network_factory(observation_size, action_size, preprocess_observations_fn=...)``.
    We only bind the hidden-layer sizes; ``ppo.train`` supplies the rest.

    Args:
      policy_sizes: hidden layer widths for the policy (actor) MLP.
      value_sizes:  hidden layer widths for the value (critic) MLP.

    Returns:
      ``functools.partial`` suitable to pass as ``network_factory``.
    """
    # Brax 0.14.x: training agents are maintained even though Brax physics is
    # deprecated. Imported here (not at module top) to keep this module importable
    # without brax/jax present.
    from brax.training.agents.ppo import networks as ppo_networks

    # Cast to tuples of int so the values are JIT/hashable-stable and immune to
    # callers passing lists or numpy ints.
    policy_sizes = tuple(int(x) for x in policy_sizes)
    value_sizes = tuple(int(x) for x in value_sizes)

    return functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=policy_sizes,
        value_hidden_layer_sizes=value_sizes,
    )


__all__ = [
    "make_network_factory",
    "DEFAULT_POLICY_HIDDEN_LAYER_SIZES",
    "DEFAULT_VALUE_HIDDEN_LAYER_SIZES",
]
