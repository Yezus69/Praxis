from pmac.rollback_gate import GateConfig, GateDecision, evaluate_gate, on_reject_actions


def _protected_pass():
    return {"pong": {"current": 10.0, "best": 10.0, "random": 0.0}}


def _current_pass():
    return {
        "progress": 0.5,
        "val_current": 10.0,
        "val_best": 10.0,
        "random": 0.0,
        "best": 10.0,
    }


def test_evaluate_gate_accepts_when_all_conditions_pass():
    decision = evaluate_gate(
        protected=_protected_pass(),
        current=_current_pass(),
        violation_rate=0.0,
        retrieval_alignment=1.0,
        cfg=GateConfig(),
    )

    assert decision.accept is True
    assert decision.regressed_games == []
    assert decision.reasons == []


def test_evaluate_gate_rejects_protected_retention_regression():
    protected = {"pong": {"current": 8.0, "best": 10.0, "random": 0.0}}
    decision = evaluate_gate(
        protected=protected,
        current=_current_pass(),
        violation_rate=0.0,
        retrieval_alignment=1.0,
        cfg=GateConfig(r_min=0.9, delta_abs=3.0),
    )

    assert decision.accept is False
    assert decision.regressed_games == ["pong"]
    assert "protected_regression" in decision.reasons


def test_evaluate_gate_rejects_violation_retrieval_progress_and_val_regression_in_isolation():
    violation = evaluate_gate(
        protected=_protected_pass(),
        current=_current_pass(),
        violation_rate=0.2,
        retrieval_alignment=1.0,
        cfg=GateConfig(max_violation_rate=0.1),
    )
    assert violation.accept is False
    assert violation.reasons == ["violation_rate"]

    retrieval = evaluate_gate(
        protected=_protected_pass(),
        current=_current_pass(),
        violation_rate=0.0,
        retrieval_alignment=0.2,
        cfg=GateConfig(retrieval_floor=0.5),
    )
    assert retrieval.accept is False
    assert retrieval.reasons == ["retrieval_alignment"]

    progress_current = _current_pass()
    progress_current["progress"] = 0.0
    progress = evaluate_gate(
        protected=_protected_pass(),
        current=progress_current,
        violation_rate=0.0,
        retrieval_alignment=1.0,
        cfg=GateConfig(min_new_progress=0.1),
    )
    assert progress.accept is False
    assert progress.reasons == ["new_game_progress"]

    val_current = _current_pass()
    val_current["val_current"] = 8.5
    val_regression = evaluate_gate(
        protected=_protected_pass(),
        current=val_current,
        violation_rate=0.0,
        retrieval_alignment=1.0,
        cfg=GateConfig(current_regress_frac=0.1),
    )
    assert val_regression.accept is False
    assert val_regression.reasons == ["current_val_regression"]


def test_on_reject_actions_returns_regressed_games_and_followup_flags():
    decision = GateDecision(
        accept=False,
        regressed_games=["pong", "breakout"],
        reasons=["protected_regression"],
    )

    actions = on_reject_actions(decision)

    assert actions == {
        "increase_risk_games": ["pong", "breakout"],
        "increase_review_games": ["pong", "breakout"],
        "write_failure_memories": True,
        "raise_retrieval_confidence": True,
    }
