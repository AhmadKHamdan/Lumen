"""
Unit tests for reach_guidance.

Covers every state (touching / almost / approach), each named direction
(left / right / up / down), the almost-vs-direction threshold, and edge
cases (degenerate frames, off-frame fingertip).

Run with::

    cd backend && python -m pytest tests/test_reach_guidance.py
"""
from __future__ import annotations

import pytest

from services.reach_guidance import (
    ALMOST_CENTER_FRAC,
    ReachInfo,
    assess_reach,
    reach_phrase,
)

W, H = 100, 100


# ---------- touching ----------

def test_fingertip_inside_box_is_touching():
    info = assess_reach((30, 30, 70, 70), (50, 50), W, H)
    assert info.state == "touching"
    assert info.direction == "center"


@pytest.mark.parametrize("tip", [(30, 30), (70, 70), (30, 70), (70, 30)])
def test_box_corners_are_inside(tip):
    # Touch test is inclusive of the box edges.
    info = assess_reach((30, 30, 70, 70), tip, W, H)
    assert info.state == "touching"


# ---------- approach (named direction) ----------

def test_target_to_the_right_says_right():
    # Target box centred at (70, 50), fingertip at (20, 50) -> dx large +ve.
    info = assess_reach((60, 40, 80, 60), (20, 50), W, H)
    assert info.state == "approach"
    assert info.direction == "right"


def test_target_to_the_left_says_left():
    info = assess_reach((10, 40, 30, 60), (80, 50), W, H)
    assert info.state == "approach"
    assert info.direction == "left"


def test_target_above_finger_says_up():
    # Target centred at (50, 20), finger at (50, 80) -> dy = -60 -> raise.
    info = assess_reach((40, 10, 60, 30), (50, 80), W, H)
    assert info.state == "approach"
    assert info.direction == "up"


def test_target_below_finger_says_down():
    info = assess_reach((40, 70, 60, 90), (50, 20), W, H)
    assert info.state == "approach"
    assert info.direction == "down"


def test_dominant_axis_wins_for_diagonal_offset():
    # dx = 30 (right), dy = 10 (down) -> X dominates -> "right".
    info = assess_reach((50, 40, 70, 60), (30, 40), W, H)
    assert info.direction == "right"


# ---------- almost ----------

def test_close_to_center_in_both_axes_is_almost():
    # Box centred at (50, 50), fingertip 5px away in each axis -> < 10% frame.
    info = assess_reach((40, 40, 60, 60), (35, 45), W, H)  # 5px outside box left
    # Actually 35 < 40 so finger is outside the box on the left.
    # dx = 50-35 = 15 -> 0.15 frac > ALMOST_CENTER_FRAC -> approach, not almost.
    # Make a tighter case for "almost":
    info = assess_reach((40, 40, 60, 60), (38, 45), W, H)
    # dx = 50-38=12 -> 0.12 > 0.10 -> still approach. Need fingertip closer.
    info = assess_reach((40, 40, 60, 60), (39, 45), W, H)
    # Box now contains (39, 45)? No: x1=40 so 39 < 40 -> outside. dx = 50-39=11
    # -> 0.11 > 0.10 still approach.
    # Note: by construction, anywhere inside the centroid's 10% window is
    # inside a 20x20 box at center (40,40,60,60). So "almost" really only
    # applies when the BOX is smaller than the almost-radius. Construct that.
    info = assess_reach((48, 48, 52, 52), (45, 51), W, H)  # tiny target
    # Box centre (50, 50). dx = 5 -> 0.05 < 0.10. dy = -1 -> 0.01 < 0.10.
    # Finger (45, 51) is outside the 48-52 box, so not touching.
    assert info.state == "almost"
    assert info.direction == "center"


def test_almost_threshold_constant_sane():
    assert 0.0 < ALMOST_CENTER_FRAC < 0.5


# ---------- phrase rendering ----------

@pytest.mark.parametrize("direction, needle", [
    ("left", "to the left"),
    ("right", "to the right"),
    ("up", "raise"),
    ("down", "lower"),
])
def test_phrase_per_direction(direction, needle):
    info = ReachInfo(state="approach", direction=direction, dx_frac=0.2, dy_frac=0.2)
    assert needle in reach_phrase("cup", info).lower()


def test_phrase_almost_says_reach_forward():
    info = ReachInfo(state="almost", direction="center", dx_frac=0.05, dy_frac=0.05)
    assert "reach forward" in reach_phrase("cup", info).lower()


def test_phrase_touching_says_grasp_and_names_target():
    info = ReachInfo(state="touching", direction="center", dx_frac=0.0, dy_frac=0.0)
    p = reach_phrase("cup", info)
    assert "cup" in p
    assert "grasp" in p.lower()


def test_phrase_handles_empty_target():
    info = ReachInfo(state="touching", direction="center", dx_frac=0.0, dy_frac=0.0)
    assert isinstance(reach_phrase("", info), str)


# ---------- degenerate input ----------

@pytest.mark.parametrize("w, h", [(0, 100), (100, 0), (0, 0)])
def test_degenerate_frame_falls_back(w, h):
    info = assess_reach((10, 10, 20, 20), (15, 15), w, h)
    assert info.state == "approach"
    assert info.direction == "center"


def test_off_frame_fingertip_still_classifies():
    # Fingertip at (-50, -50) is above AND left of the target -> target is to
    # the RIGHT and BELOW the finger, so the named direction (whichever axis
    # wins) is one of "right" / "down".
    info = assess_reach((40, 40, 60, 60), (-50, -50), W, H)
    assert info.state == "approach"
    assert info.direction in ("right", "down")
