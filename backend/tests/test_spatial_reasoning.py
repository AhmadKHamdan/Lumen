"""
Unit tests for spatial_reasoning.

Covers the region (left/center/right) and distance (near/medium/far) bucketing
under the per-class size-prior model, the threshold boundaries, degenerate
frames, and the most_centered / closest selectors.

Run with::

    cd backend && python -m pytest tests/test_spatial_reasoning.py
"""
from __future__ import annotations

import pytest

from services.spatial_reasoning import (
    DEFAULT_NEAR_APPARENT,
    FAR_RATIO_MAX,
    NEAR_RATIO_MIN,
    PER_CLASS_NEAR_APPARENT,
    REGION_LEFT_MAX,
    REGION_RIGHT_MIN,
    closest,
    locate,
    most_centered,
    near_threshold_for,
)
from services.yolo_service import Detection


# ---------- region bucketing ----------

@pytest.mark.parametrize("box, expected_region", [
    ((0, 0, 20, 80), "left"),       # cx=10/100=0.10
    ((40, 40, 60, 60), "center"),   # cx=50/100=0.50
    ((80, 0, 100, 80), "right"),    # cx=90/100=0.90
])
def test_region_buckets(box, expected_region):
    info = locate(box, 100, 100)
    assert info.region == expected_region


def test_region_boundaries_are_center():
    # cx_frac exactly at the split points falls into "center" (strict <, >).
    w = 100
    left_edge_cx = REGION_LEFT_MAX * w   # 35
    right_edge_cx = REGION_RIGHT_MIN * w  # 65
    assert locate((left_edge_cx, 0, left_edge_cx, 10), w, 100).region == "center"
    assert locate((right_edge_cx, 0, right_edge_cx, 10), w, 100).region == "center"


# ---------- distance bucketing (default / no label) ----------
#
# DEFAULT_NEAR_APPARENT is 0.22. With NEAR_RATIO_MIN=0.85 and FAR_RATIO_MAX=0.35
# the default-class breakpoints are approximately:
#     apparent_frac >= 0.19  -> near
#     apparent_frac <= 0.08  -> far
#     in between             -> medium

def test_distance_near_default():
    # 60x60 box on a 100x100 frame -> apparent_frac = 0.60 -> near.
    info = locate((20, 20, 80, 80), 100, 100)
    assert info.distance == "near"
    assert info.apparent_frac == pytest.approx(0.60)


def test_distance_medium_default():
    # 12x12 box on a 100x100 frame -> apparent_frac = 0.12 -> ratio 0.55 -> medium.
    info = locate((45, 45, 57, 57), 100, 100)
    assert info.distance == "medium"


def test_distance_far_default():
    # 5x5 box on a 100x100 frame -> apparent_frac = 0.05 -> ratio 0.23 -> far.
    info = locate((48, 48, 53, 53), 100, 100)
    assert info.distance == "far"


# ---------- distance bucketing (per-class priors) ----------

def test_same_apparent_size_classifies_differently_per_class():
    """A 18x18 box on a 100x100 frame is 'near' for a cup but 'medium' for a laptop."""
    box = (40, 40, 58, 58)   # apparent_frac = 0.18
    cup = locate(box, 100, 100, label="cup")        # cup near at 0.18 -> ratio 1.0 -> near
    laptop = locate(box, 100, 100, label="laptop")  # laptop near at 0.40 -> ratio 0.45 -> medium
    assert cup.distance == "near"
    assert laptop.distance == "medium"


def test_laptop_filling_40_percent_is_near():
    """The exact symptom the user reported: laptop right in front of you."""
    # 40x30 box on a 100x100 frame -> apparent_frac = 0.40 -> ratio 1.0 against
    # the laptop threshold -> near. Old area-only code put this at medium.
    info = locate((30, 35, 70, 65), 100, 100, label="laptop")
    assert info.distance == "near"


def test_cell_phone_at_phone_threshold_is_near():
    # cell phone near at 0.12.
    info = locate((44, 44, 56, 56), 100, 100, label="cell phone")  # apparent 0.12
    assert info.distance == "near"


def test_far_far_away_cup():
    info = locate((49, 49, 53, 51), 100, 100, label="cup")  # apparent 0.04
    assert info.distance == "far"


def test_unknown_label_falls_back_to_default_threshold():
    # giraffe is not in the table; should fall back to DEFAULT_NEAR_APPARENT.
    assert near_threshold_for("giraffe") == DEFAULT_NEAR_APPARENT


def test_apparent_uses_max_of_width_and_height():
    """A tall narrow bottle should be classified by its height, not flattened by width."""
    # 6 wide x 30 tall on a 100x100 frame -> w_frac 0.06, h_frac 0.30,
    # apparent 0.30. With bottle threshold 0.18, ratio 1.67 -> near.
    info = locate((47, 35, 53, 65), 100, 100, label="bottle")
    assert info.apparent_frac == pytest.approx(0.30)
    assert info.distance == "near"


def test_distance_threshold_constants_sane():
    assert 0.0 < FAR_RATIO_MAX < NEAR_RATIO_MIN <= 1.0
    assert all(0.05 < v < 1.0 for v in PER_CLASS_NEAR_APPARENT.values())


# ---------- degenerate frames ----------

@pytest.mark.parametrize("w, h", [(0, 100), (100, 0), (0, 0)])
def test_degenerate_frame_falls_back(w, h):
    info = locate((10, 10, 20, 20), w, h)
    assert info.region == "center"
    assert info.distance == "medium"


def test_cx_frac_clamped():
    info = locate((90, 0, 200, 50), 100, 100)
    assert 0.0 <= info.cx_frac <= 1.0


# ---------- SpatialInfo carries the new fields ----------

def test_spatial_info_records_apparent_and_label():
    info = locate((30, 35, 70, 65), 100, 100, label="laptop")
    assert info.label == "laptop"
    assert info.apparent_frac == pytest.approx(0.40)


# ---------- selectors ----------

def test_most_centered_picks_nearest_to_middle():
    frame_w = 100
    left = Detection("cup", 0.9, (0, 0, 10, 10))     # cx=5
    mid = Detection("cup", 0.9, (45, 0, 55, 10))     # cx=50
    right = Detection("cup", 0.9, (90, 0, 100, 10))  # cx=95
    assert most_centered([left, mid, right], frame_w) is mid


def test_most_centered_empty_returns_none():
    assert most_centered([], 100) is None


def test_closest_picks_largest_area():
    small = Detection("cup", 0.9, (0, 0, 10, 10))    # area 100
    big = Detection("cup", 0.9, (0, 0, 50, 50))      # area 2500
    assert closest([small, big]) is big


def test_closest_empty_returns_none():
    assert closest([]) is None
