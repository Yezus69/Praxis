"""Task-identity leakage tests (spec 17.8, 26).

Static introspection proving that the policy, router, memory-write, transaction,
consolidation, and serialization APIs expose no game / task / label / curriculum
/ environment-id argument. Internal *context* coordinates (``ctx``, ``context``)
are allowed: they are allocated from experience and retrieved by content, not
supplied as external identity (spec 9.3).
"""

from __future__ import annotations

import inspect

import pytest

from tfns.castm import (
    address,
    audit,
    consolidate,
    layers,
    router,
    scratch,
    state,
    synaptic,
    transaction,
)

# Substrings that would indicate an external task/game identity leak.
FORBIDDEN = ("game", "task", "label", "curriculum", "env_id", "envid", "onehot", "one_hot", "game_id")
# Explicitly allowed tokens that merely *contain* nothing forbidden but are
# adjacent in meaning; listed for documentation. Internal context is allowed.
ALLOWED_CONTEXT_TOKENS = ("ctx", "context", "ctx_id")

MODULES = [address, audit, consolidate, layers, router, scratch, state, synaptic, transaction]


def _public_callables(module):
    for name in getattr(module, "__all__", dir(module)):
        obj = getattr(module, name, None)
        if inspect.isfunction(obj):
            yield f"{module.__name__}.{name}", obj


def test_no_task_identity_in_any_castm_api():
    offenders = []
    for module in MODULES:
        for qualname, fn in _public_callables(module):
            try:
                sig = inspect.signature(fn)
            except (ValueError, TypeError):
                continue
            for pname in sig.parameters:
                low = pname.lower()
                if any(tok in low for tok in FORBIDDEN):
                    offenders.append(f"{qualname}({pname})")
    assert not offenders, f"task-identity leak in API parameters: {offenders}"


def test_route_step_signature_is_content_only():
    # The router consumes only a content query and a dynamics-error scalar.
    sig = inspect.signature(router.route_step)
    params = set(sig.parameters)
    assert params == {"state", "index", "q", "dyn_err", "cfg"}
    for forbidden in FORBIDDEN:
        assert all(forbidden not in p.lower() for p in params)


def test_commit_api_uses_internal_context_not_task():
    sig = inspect.signature(transaction.commit_scratch_bank)
    params = list(sig.parameters)
    # The write addresses an internal context id, never a game/task label.
    assert "ctx_id" in params
    for forbidden in FORBIDDEN:
        assert all(forbidden not in p.lower() for p in params)


def test_suite_module_isolated_from_agent_apis():
    # Game names live only in the suite/harness module. None of the agent-facing
    # CASTM modules import it.
    import importlib

    for module in MODULES:
        src = inspect.getsource(module)
        assert "from tfns.castm.suite" not in src
        assert "import tfns.castm.suite" not in src
