"""
Unit tests for hand_service's pure helpers.

We don't load MediaPipe here - it's heavy to import and the actual model is
exercised manually on hardware. What we DO test is the pure
``_landmarks_to_pose`` converter, with mock landmark objects, so the
NormalizedLandmark -> HandPose mapping is locked down.

Run with::

    cd backend && python -m pytest tests/test_hand_service.py
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from services.hand_service import HandPose, _landmarks_to_pose


@dataclass
class _MockLM:
    """Mimic MediaPipe NormalizedLandmark (we only use .x and .y)."""
    x: float
    y: float
    z: float = 0.0


def _twenty_one_landmarks(wrist=(0.5, 0.95), tip=(0.5, 0.2)):
    """Build a 21-landmark list with controllable wrist (idx 0) and
    index-finger-tip (idx 8); the rest are arbitrary but inside [0, 1]."""
    lms = [_MockLM(0.5, 0.5)] * 21
    lms[0] = _MockLM(*wrist)
    lms[8] = _MockLM(*tip)
    # A few spread points so the bbox isn't degenerate.
    lms[4] = _MockLM(0.4, 0.8)
    lms[20] = _MockLM(0.6, 0.3)
    return lms


def test_basic_landmarks_yield_pixel_pose():
    lms = _twenty_one_landmarks(wrist=(0.5, 0.9), tip=(0.5, 0.2))
    pose = _landmarks_to_pose(lms, frame_w=640, frame_h=480)
    assert isinstance(pose, HandPose)
    assert pose.wrist == pytest.approx((320.0, 432.0))   # 0.5 * 640, 0.9 * 480
    assert pose.fingertip == pytest.approx((320.0, 96.0))  # 0.5 * 640, 0.2 * 480
    assert pose.score == pytest.approx(1.0)


def test_bbox_wraps_all_landmarks():
    lms = _twenty_one_landmarks(wrist=(0.1, 0.95), tip=(0.9, 0.05))
    pose = _landmarks_to_pose(lms, frame_w=100, frame_h=100)
    x1, y1, x2, y2 = pose.bbox
    assert x1 <= 10.0   # min x from wrist
    assert x2 >= 90.0   # max x from tip
    assert y1 <= 5.0    # min y from tip
    assert y2 >= 95.0   # max y from wrist


def test_empty_landmarks_returns_none():
    assert _landmarks_to_pose([], 100, 100) is None
    assert _landmarks_to_pose(None, 100, 100) is None


def test_too_few_landmarks_returns_none():
    # Need at least 9 landmarks (we read indices 0 and 8).
    lms = [_MockLM(0.5, 0.5)] * 5
    assert _landmarks_to_pose(lms, 100, 100) is None


def test_score_override_recorded():
    lms = _twenty_one_landmarks()
    pose = _landmarks_to_pose(lms, 100, 100, score=0.73)
    assert pose.score == pytest.approx(0.73)


def test_hand_pose_is_frozen():
    pose = HandPose(fingertip=(1, 2), wrist=(3, 4), bbox=(0, 0, 10, 10), score=0.9)
    with pytest.raises((AttributeError, Exception)):
        pose.fingertip = (5, 6)  # frozen dataclass
