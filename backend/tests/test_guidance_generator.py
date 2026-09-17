"""
Unit tests for guidance_generator.

Checks every region x distance combination produces a sensible, target-named
phrase, plus the situational phrases (first-seen, scanning, lost, timeout,
complete, cancel).

Run with::

    cd backend && python -m pytest tests/test_guidance_generator.py
"""
from __future__ import annotations

import pytest

from services import guidance_generator as gg
from services.spatial_reasoning import SpatialInfo


def _info(region, distance):
    return SpatialInfo(region=region, distance=distance, cx_frac=0.5, area_frac=0.1)


# ---------- main guidance ----------

def test_center_near_invites_the_hand():
    phrase = gg.guidance_phrase("cup", _info("center", "near"))
    assert "right in front of you" in phrase.lower()
    assert "raise your hand" in phrase.lower()   # concrete cue, never "reach forward"
    assert "cup" in phrase


@pytest.mark.parametrize("region, clause", [
    ("left", "to your left"),
    ("right", "to your right"),
    ("center", "straight ahead"),
])
def test_direction_clause_present(region, clause):
    phrase = gg.guidance_phrase("bottle", _info(region, "medium"))
    assert clause in phrase.lower()
    assert "bottle" in phrase


@pytest.mark.parametrize("distance, needle", [
    ("near", "within arm's reach"),   # proximity hands off to hand guidance
    ("medium", "a few steps away"),
    ("far", "far away"),
])
def test_distance_clause_present(distance, needle):
    phrase = gg.guidance_phrase("book", _info("left", distance))
    assert needle in phrase.lower()


def test_target_name_interpolated():
    phrase = gg.guidance_phrase("dining table", _info("right", "far"))
    assert "dining table" in phrase


# ---------- situational phrases ----------

def test_first_seen_leads_with_found():
    phrase = gg.first_seen_phrase("cup", _info("left", "medium"))
    assert phrase.lower().startswith("found")
    assert "to your left" in phrase.lower()


def test_first_seen_center_near():
    phrase = gg.first_seen_phrase("cup", _info("center", "near"))
    assert "found" in phrase.lower()
    assert "right in front" in phrase.lower()


def test_scanning_phrase_prompts_turn():
    phrase = gg.scanning_phrase("remote")
    assert "remote" in phrase
    assert "turn" in phrase.lower()


@pytest.mark.parametrize("region, needle", [
    ("left", "to your left"),
    ("right", "to your right"),
    ("center", "straight ahead"),
])
def test_lost_phrase_recalls_last_region(region, needle):
    phrase = gg.lost_phrase("cup", region)
    assert "lost sight" in phrase.lower()
    assert needle in phrase.lower()


def test_lost_phrase_unknown_region():
    phrase = gg.lost_phrase("cup", None)
    assert "lost sight" in phrase.lower()
    assert "camera" in phrase.lower()


def test_timeout_phrase():
    phrase = gg.timeout_phrase("cup")
    assert "couldn't find" in phrase.lower()
    assert "cup" in phrase


def test_complete_phrase_names_target():
    phrase = gg.complete_phrase("cup")
    assert "cup" in phrase


def test_cancel_phrase():
    assert "stop" in gg.cancel_phrase("cup").lower()


def test_handles_empty_target_gracefully():
    # Should not raise and should still produce a string.
    assert isinstance(gg.guidance_phrase("", _info("center", "far")), str)
    assert isinstance(gg.scanning_phrase(""), str)
