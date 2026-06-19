"""Task-free context-change detection."""

from tfns.detect.change import DetectorState, PageHinkleyDetector, signature_window

__all__ = [
    "DetectorState",
    "PageHinkleyDetector",
    "signature_window",
]
