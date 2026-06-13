"""praxis/envs/randomize.py — Brax-style continuous domain randomization.

``domain_randomize(model, rng) -> (batched_model, in_axes)`` follows the
Brax / MuJoCo-Playground ``randomization_fn`` convention consumed by
``BraxDomainRandomizationVmapWrapper`` (a.k.a. ``get_domain_randomizer``):

  * Build a *batched* copy of the ``mjx.Model`` where ONLY the randomized
    fields carry a leading batch axis (one row per parallel env).
  * Return an ``in_axes`` pytree shaped like the model: ``0`` on the leaf paths
    that were batched, ``None`` everywhere else. The wrapper ``jax.vmap``s the
    env over ``(batched_model, in_axes)``.

CRITICAL CONSTRAINT (MJX fixed topology): we randomize ONLY continuous fields —
never geom COUNT, never add/remove bodies. Here we randomize:
  * ``geom_size``   — agent + obstacle radii (continuous)
  * ``geom_friction`` — floor / contact friction (continuous)

The randomized START / GOAL / per-obstacle PATROL params live in
``state.info`` and are sampled inside :meth:`NavEnv.reset` from the per-env
``rng`` (see nav_env.py) — they are NOT model fields, so they do not belong in
the batched model here. This split is deliberate: model-level randomization
(sizes/friction) goes through ``in_axes``; task-level randomization
(positions/phases) goes through the reset rng. Both are continuous-only.

NOTE(orchestrator): the exact set of randomized model leaves and the in_axes
tree must match what the installed ``BraxDomainRandomizationVmapWrapper``
expects (it vmaps env.reset/step with ``in_axes=(model_in_axes, 0)``). We batch
``geom_size`` and ``geom_friction`` and mark them with ``0``; verify the wrapper
threads a per-leaf in_axes Model the same way (this is the documented Playground
pattern, e.g. the locomotion ``domain_randomize`` fns).
"""

from __future__ import annotations

from typing import Any, Tuple

import jax
import jax.numpy as jp


# Randomization ranges (continuous). Multipliers/additive perturbations applied
# to the nominal model values so the env stays learnable day-one. Kept SMALL
# (size +/-5%) so the geometric collision check in nav_env.py (which uses the
# nominal config radii, not these per-env model sizes) stays within ~0.0125 m of
# the physics contact — an acceptable approximation for the MVP.
_SIZE_FACTOR_RANGE = (0.95, 1.05)       # multiplier on each geom's radius column
_FRICTION_SLIDE_RANGE = (0.8, 1.2)      # multiplier on tangential friction


def domain_randomize(model: Any, rng: jax.Array) -> Tuple[Any, Any]:
    """Return ``(batched_model, in_axes)`` for Brax-style vmapped randomization.

    Args:
        model: an ``mjx.Model`` (single, unbatched).
        rng: a ``jax.random.PRNGKey``. Its leading batch dimension determines
            how many parallel envs to produce: this fn is called by the wrapper
            with ``rng`` already split to shape ``(num_envs, 2)``.

    Returns:
        (batched_model, in_axes) where ``batched_model`` has a leading axis on
        the randomized leaves and ``in_axes`` is a Model-shaped pytree with
        ``0`` on those leaves and ``None`` elsewhere.
    """
    # The wrapper passes a batched rng of shape (num_envs, 2). Derive num_envs
    # from its shape (a static dim under vmap-free construction).
    rng = jp.asarray(rng)
    if rng.ndim == 1:
        # A single key was passed; treat as a batch of 1.
        rng = rng[None, :]
    num_envs = rng.shape[0]

    geom_size = model.geom_size            # (ngeom, 3)
    geom_friction = model.geom_friction    # (ngeom, 3)
    ngeom = geom_size.shape[0]

    def _per_env(key: jax.Array):
        """Produce per-env randomized (geom_size, geom_friction)."""
        k_size, k_fric = jax.random.split(key)

        # --- sizes: scale every geom's primary radius (column 0) by a small
        #     per-geom continuous factor. Cylinders use size[0]=radius. This
        #     never changes geom COUNT — only continuous size values. ---
        size_factor = jax.random.uniform(
            k_size, (ngeom, 1),
            minval=_SIZE_FACTOR_RANGE[0], maxval=_SIZE_FACTOR_RANGE[1],
        )
        # Only perturb the radius column (0); leave half-length / unused cols.
        size_scale = jp.concatenate(
            [size_factor, jp.ones((ngeom, 2))], axis=-1
        )
        new_size = geom_size * size_scale

        # --- friction: scale the tangential (column 0) friction by a continuous
        #     factor; keep torsional/rolling columns. ---
        fric_factor = jax.random.uniform(
            k_fric, (ngeom, 1),
            minval=_FRICTION_SLIDE_RANGE[0], maxval=_FRICTION_SLIDE_RANGE[1],
        )
        fric_scale = jp.concatenate(
            [fric_factor, jp.ones((ngeom, 2))], axis=-1
        )
        new_friction = geom_friction * fric_scale

        return new_size, new_friction

    sizes, frictions = jax.vmap(_per_env)(rng)  # (num_envs, ngeom, 3) each

    # Build the batched model: replace ONLY the randomized leaves with their
    # batched versions; everything else stays shared (in_axes None).
    batched_model = model.tree_replace({
        "geom_size": sizes,
        "geom_friction": frictions,
    })

    # in_axes pytree: same structure as model, 0 on randomized leaves, None else.
    in_axes = jax.tree_util.tree_map(lambda _: None, model)
    in_axes = in_axes.tree_replace({
        "geom_size": 0,
        "geom_friction": 0,
    })

    return batched_model, in_axes
