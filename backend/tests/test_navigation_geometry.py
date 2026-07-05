"""Tests for services.navigation.geometry - compass/bearing math.

Pure functions plus the NavState-anchored helpers. The nasty cases are all
wrap-around: a door directly behind the user straddles the +/-180 seam, and
turn directions must always take the short way round.
"""
from __future__ import annotations

import pytest

from services.navigation.geometry import (
    _cluster_bearings,
    _direction,
    _signed_from_ref_deg,
    _track_turn,
    _turn_to,
)
from services.navigation.state import NavState


@pytest.fixture()
def st():
    s = NavState()
    s["ref_heading"] = 90.0   # scan started facing east
    return s


class TestTurnTo:
    def test_short_way_right(self):
        assert _turn_to(120.0, 90.0) == pytest.approx(30.0)

    def test_short_way_left(self):
        assert _turn_to(60.0, 90.0) == pytest.approx(-30.0)

    def test_wraps_across_north(self):
        # Facing 350, target 10 -> +20 (right), never -340.
        assert _turn_to(10.0, 350.0) == pytest.approx(20.0)


class TestDirection:
    @pytest.mark.parametrize("deg,expected", [
        (0.0, "ahead"),
        (25.0, "ahead"),
        (60.0, "on your right"),
        (120.0, "behind you, to the right"),
        (179.0, "behind you"),
        (-179.0, "behind you"),
        (-120.0, "behind you, to the left"),
        (-60.0, "on your left"),
    ])
    def test_buckets(self, deg, expected):
        assert _direction(deg) == expected


class TestSignedFromRef:
    def test_right_of_start(self, st):
        assert _signed_from_ref_deg(st, 120.0) == pytest.approx(30.0)

    def test_left_of_start_wraps(self, st):
        assert _signed_from_ref_deg(st, 350.0) == pytest.approx(-100.0)


class TestClusterBearings:
    def test_one_object_many_sightings(self, st):
        clusters = _cluster_bearings(st, [100.0, 105.0, 110.0])
        assert len(clusters) == 1
        mean, n = clusters[0]
        assert n == 3
        assert mean == pytest.approx(15.0, abs=1.0)   # ~105 abs -> +15 from ref 90

    def test_two_distinct_objects(self, st):
        clusters = _cluster_bearings(st, [100.0, 102.0, 200.0, 205.0])
        assert len(clusters) == 2
        assert sorted(n for _m, n in clusters) == [2, 2]

    def test_seam_merge_directly_behind(self, st):
        # Sightings at 269 and 273 with ref 90 -> signed +179 / -177: one door
        # straddling the +/-180 seam must merge into ONE cluster, mean ~180.
        clusters = _cluster_bearings(st, [269.0, 273.0])
        assert len(clusters) == 1
        mean, n = clusters[0]
        assert n == 2
        assert abs(mean) == pytest.approx(179.0, abs=2.0)


class TestTrackTurn:
    def test_accumulates_rotation(self, st):
        st["last_heading"] = 90.0
        for h in (120.0, 150.0, 180.0):
            _track_turn(st, h)
        assert st["net_rotation"] == pytest.approx(90.0)

    def test_wraps_through_north(self, st):
        st["last_heading"] = 350.0
        _track_turn(st, 10.0)
        assert st["net_rotation"] == pytest.approx(20.0)

    def test_ignores_compass_glitches(self, st):
        st["last_heading"] = 0.0
        _track_turn(st, 170.0)   # a >120 deg jump in one frame = sensor glitch
        assert st["net_rotation"] == 0.0

    def test_no_heading_is_noop(self, st):
        st["last_heading"] = None
        _track_turn(st, 90.0)    # first reading only sets the anchor elsewhere
        assert st["net_rotation"] == 0.0
