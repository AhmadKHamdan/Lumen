"""End-to-end tests for the navigation exploration controller.

Drives the ported decision layer (NavState + controller) through scripted
frames the way engine.py does at runtime - no models, no I/O, no clock. This
is the same layering as test_object_allocation: the perception results are
faked, the *policy* is what's under test.

The long test walks a full journey:
  room 1: guided 360 scan -> door found on the right -> face it -> call-out ->
          path-clear verdict -> approach -> at-door -> transit ->
  room 2: fresh scan -> fridge+oven sighted -> directed confirm -> ARRIVAL.
"""
from __future__ import annotations

import pytest

from services.navigation import controller
from services.navigation.goals import indicators_for
from services.navigation.state import NavState


@pytest.fixture()
def st():
    s = NavState()
    s["goal"] = "kitchen"
    return s


PRIM, SEC = indicators_for("kitchen")
INDICATORS = set(PRIM) | set(SEC)


def tick(st, heading, *, seen=frozenset(), door=False, door_cx=None, corro=False,
         dist=None, region=None, motion=5.0, near_box=False, cur_frac=0.0,
         obst=("", False, False)):
    """One engine tick: evidence -> transit -> controller.step, like engine.run."""
    confirmed = controller.accumulate_indicator_evidence(st, set(seen), INDICATORS)
    transit, just_near = controller.detect_transit(st, near_box, door, cur_frac, motion)
    return controller.step(
        st, goal="kitchen", heading=heading, motion=motion, w=640,
        seen=set(seen), indicators=INDICATORS,
        obj_dets=[(c, 0.8, [300.0, 200.0, 400.0, 400.0]) for c in seen],
        door_confirmed=door, door_cx_frac=door_cx, door_corro=corro,
        region=region, door_dist=dist, transit=transit, just_near=just_near,
        confirmed=confirmed, obst_guidance=obst[0], obst_priority=obst[1],
        obst_blocking=obst[2])


def run_360_scan(st, start_heading, sight_fn, max_steps=40):
    """Advance the compass through a full circle, calling sight_fn(heading) ->
    dict of tick kwargs per frame. Returns the spoken lines."""
    lines = []
    g, *_ = tick(st, start_heading)   # anchors ref_heading, speaks the instruction
    if g:
        lines.append(g)
    h = start_heading
    for _ in range(max_steps):
        h = (h + 10.0) % 360.0
        g, *_ = tick(st, h, **sight_fn(h))
        if g:
            lines.append(g)
        if st["mode"] != "discover":
            break
    return lines


class TestDiscoverScan:
    def test_first_frame_speaks_scan_instruction(self, st):
        g, priority, arrived, _, _ = tick(st, 0.0)
        assert priority
        assert "slowly turn" in g.lower()
        assert not arrived

    def test_no_compass_first_frame_uses_fallback_prompt(self, st):
        g, priority, *_ = tick(st, None)
        assert priority
        assert "scan the room" in g.lower() or "pan" in g.lower()

    def test_full_circle_with_one_door_targets_it(self, st):
        def sights(h):
            if 85.0 <= h <= 115.0:   # door while facing ~right of start
                return dict(door=True, door_cx=0.5, corro=True,
                            region="ahead", dist=4.0)
            return {}
        lines = run_360_scan(st, 0.0, sights)
        assert st["mode"] == "face_target"
        assert st["target_kind"] == "door"
        summary = lines[-1]
        assert "door" in summary and "right" in summary

    def test_empty_room_rescans(self, st):
        lines = run_360_scan(st, 0.0, lambda h: {})
        assert st["mode"] == "discover"       # went back to a fresh scan
        assert "one more time" in lines[-1].lower() or "scan" in lines[-1].lower()


class TestFullJourney:
    def test_two_rooms_to_arrival(self, st):
        # --- Room 1: scan; a single door on the right, no indicators.
        def room1(h):
            if 85.0 <= h <= 115.0:
                return dict(door=True, door_cx=0.5, corro=True,
                            region="ahead", dist=4.0)
            return {}
        run_360_scan(st, 0.0, room1)
        assert st["mode"] == "face_target"
        target_h = st["target_heading"]

        # Face the door -> silent handoff to go_door.
        for _ in range(5):
            tick(st, target_h, door=True, door_cx=0.5, region="ahead", dist=4.0)
            if st["mode"] == "go_door":
                break
        assert st["mode"] == "go_door"

        # One-shot call-out, then the path-clear verdict with walking cue.
        spoken = []
        for _ in range(10):
            g, *_ = tick(st, target_h, door=True, door_cx=0.5,
                         region="ahead", dist=3.0)
            if g:
                spoken.append(g)
        assert any("let me check the path" in g.lower() for g in spoken)
        assert any("path is clear" in g.lower() for g in spoken)

        # Approach until the door fills the frame -> at-door instruction.
        at_door = None
        for _ in range(6):
            g, *_ = tick(st, target_h, door=True, door_cx=0.5, region="ahead",
                         dist=1.0, cur_frac=0.9, motion=20.0)
            if g and "right at the door" in g.lower():
                at_door = g
                break
        assert at_door is not None
        assert "walk through" in at_door.lower()

        # Door gone + sustained camera motion -> transit -> new discover.
        transit_line = None
        for _ in range(12):
            g, *_ = tick(st, target_h, door=False, motion=30.0)
            if g:
                transit_line = g
            if st["mode"] == "discover":
                break
        assert st["mode"] == "discover"
        assert transit_line and "through" in transit_line.lower()

        # --- Room 2: fridge + oven cluster behind the entry direction.
        ref = st["ref_heading"]   # transit anchored the new scan to that heading

        def room2(h):
            rel = (h - ref) % 360.0
            if 160.0 <= rel <= 200.0:
                return dict(seen={"refrigerator", "oven"})
            return {}
        lines = run_360_scan(st, ref, room2)
        assert st["mode"] == "face_target"
        assert st["target_kind"] == "indicator"
        assert "kitchen" in lines[-1]

        # Face the sighting -> go_indicator -> confirm settles -> arrival.
        target_h = st["target_heading"]
        for _ in range(3):
            tick(st, target_h, seen={"refrigerator", "oven"})
            if st["mode"] == "go_indicator":
                break
        assert st["mode"] == "go_indicator"

        arrival = None
        for _ in range(20):
            g, p, arrived, phrase, matched = tick(
                st, target_h, seen={"refrigerator", "oven"})
            if arrived:
                arrival = phrase
                break
        assert arrival is not None
        assert "we've reached the kitchen" in arrival.lower()
        assert "fridge" in arrival.lower()


class TestObstaclePreemption:
    def test_blocking_obstacle_overrides_door_guidance(self, st):
        st.enter_go_door()
        st["door_announced"] = True     # call-out already made
        st["path_checked"] = True
        g, priority, *_ = tick(
            st, 0.0, door=True, door_cx=0.5, region="ahead", dist=3.0,
            obst=("There's a chair in your path. Step to your left, where it's clear.",
                  True, True))
        assert "chair" in g.lower()
        assert priority


class TestNavStateIsolation:
    def test_sessions_do_not_share_state(self):
        a, b = NavState(), NavState()
        a["goal"] = "kitchen"
        a["mode"] = "go_door"
        a.scan_counts.update({"refrigerator": 3})
        a.door_hist.append(("ahead", 2.0))
        assert b["goal"] is None
        assert b["mode"] == "discover"
        assert not b.scan_counts
        assert not b.door_hist

    def test_enter_discover_resets_room_evidence(self):
        s = NavState()
        s.scan_counts.update({"oven": 5})
        s["door_bearings"].extend([10.0, 20.0])
        s["net_rotation"] = 270.0
        s.enter_discover()
        assert not s.scan_counts
        assert s["door_bearings"] == []
        assert s["net_rotation"] == 0.0
