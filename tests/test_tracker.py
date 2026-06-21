"""
Basic smoke tests for the ObjectTracker class.
These don't require GPU and only check that the pipeline wires together
correctly on a synthetic frame.

Run with: pytest tests/test_tracker.py
"""


import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tracker import ObjectTracker  # noqa: E402


@pytest.fixture(scope="module")
def tracker():
    return ObjectTracker(model_path="yolov8n.pt", confidence_threshold=0.4, device="cpu")


def test_tracker_initializes(tracker):
    assert tracker.model is not None
    assert tracker.tracker is not None
    assert tracker.device == "cpu"


def test_detect_returns_list(tracker):
    # Black synthetic frame — model should run without error, likely 0 detections
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = tracker.detect(frame)
    assert isinstance(detections, list)


def test_track_handles_empty_detections(tracker):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    tracks = tracker.track(frame, [])
    assert isinstance(tracks, list)


def test_color_for_id_is_deterministic():
    c1 = ObjectTracker._color_for_id(5)
    c2 = ObjectTracker._color_for_id(5)
    assert c1 == c2
