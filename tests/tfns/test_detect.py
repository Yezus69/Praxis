from __future__ import annotations

import numpy as np

from tfns.config import DetectConfig
from tfns.detect.change import PageHinkleyDetector, signature_window


def test_signature_window_normalizes_mean_key():
    keys = np.array([[3.0, 0.0], [0.0, 4.0]], dtype=np.float32)
    sig = signature_window(keys)

    np.testing.assert_allclose(np.linalg.norm(sig), 1.0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(sig, np.array([3.0, 4.0], dtype=np.float32) / 5.0)


def test_detector_does_not_fire_on_stationary_stream_or_single_spike():
    detector = PageHinkleyDetector(DetectConfig(ph_delta=0.05, ph_lambda=5.0, cooldown_blocks=2))
    state = detector.init()

    for x in [0.0, 0.02, -0.01, 0.01, 0.0] * 8:
        state, changed = detector.update(state, x)
        assert changed is False

    state, changed = detector.update(state, 20.0)
    assert changed is False


def test_detector_fires_once_after_sustained_shift_and_respects_cooldown():
    detector = PageHinkleyDetector(DetectConfig(ph_delta=0.05, ph_lambda=5.0, cooldown_blocks=3))
    state = detector.init()

    for _ in range(20):
        state, changed = detector.update(state, 0.0)
        assert changed is False

    changed_at = None
    for idx in range(6):
        state, changed = detector.update(state, 10.0)
        if changed:
            changed_at = idx
            break

    assert changed_at is not None
    assert changed_at >= 1

    for _ in range(3):
        state, changed = detector.update(state, 10.0)
        assert changed is False
