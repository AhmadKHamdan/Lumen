"""End-to-end tests for the navigation exploration controller.

Drives the ported decision layer (NavState + controller) through scripted
frames the way engine.py does at runtime - no models, no I/O, no clock. This
is the same layering as test_object_allocation: the perception results are
faked, the *policy* is what's under test.

Pacing constants are SECONDS of observed-frame time; every tick here passes
DT = 1/3 s (the engine's nominal rate), so "6 ticks" below means "2 seconds
of frames".

The long test walks a full journey:
  room 1: guided 360 scan -> door found on the right -> face it -> call-out ->
          path-clear verdict -> approach -> at-door -> transit ->
  room 2: fresh scan -> fridge+oven sighted -> directed confirm -> ARRIVAL.
"""
from __future__ import annotations

import pytest

from services.navigation import controller
from services.navigation.config import (REPROMPT_SEC, ROOM_CAP, ROOM_CAP_REMIND,
                                        TRANSIT_GONE_SEC)
from services.navigation.goals import indicators_for
from services.navigation.state import NavState

DT = 1.0 / 3.0   # nominal engine tick


@pytest.fixture()
def st():
    s = NavState()
    s["goal"] = "kitchen"
    return s


PRIM, SEC = indicators_for("kitchen")
INDICATORS = set(PRIM) | set(SEC)


def tick(st, heading, *, seen=frozenset(), door=False, door_cx=None, corro=False,
         dist=None, region=None, motion=5.0, near_box=False, cur_frac=0.0,
         obst=("", False, False), dt=DT):
    """One engine tick: evidence -> transit -> controller.step, like engine.run."""
    confirmed = controller.accumulate_indicator_evidence(st, set(seen), INDICATORS)
    transit, just_near = controller.detect_transit(st, near_box, door, cur_frac,
                                                   motion, dt)
    return controller.step(
        st, goal="kitchen", heading=heading, motion=motion, w=640,
        seen=set(seen), indicators=INDICATORS,
        obj_dets=[(c, 0.8, [300.0, 200.0, 400.0, 400.0]) for c in seen],
        door_confirmed=door, door_cx_frac=door_cx, door_corro=corro,
        region=region, door_dist=dist, transit=transit, just_near=just_near,
        confirmed=confirmed, obst_guidance=obst[0], obst_priority=obst[1],
        obst_blocking=obst[2], dt=dt)


def force_transit(st):
    """Drive controller.step with transit=True directly (the doorway-crossing
    outcome), skipping the physical approach - for room-cap tests."""
    return controller.step(
        st, goal="kitchen", heading=90.0, motion=5.0, w=640,
        seen=set(), indicators=INDICATORS, obj_dets=[],
        door_confirmed=False, door_cx_frac=None, door_corro=False,
        region=None, door_dist=None, transit=True, just_near=False,
        confirmed=set(), obst_guidance="", obst_priority=False,
        obst_blocking=False, dt=DT)


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

        # One-shot call-out, then (after the ~2 s holdoff) the path-clear
        # verdict with the walking cue.
        spoken = []
        for _ in range(12):
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

        # Door gone + sustained camera motion -> after TRANSIT_GONE_SEC of
        # doorless frames, transit -> new discover.
        transit_line = None
        for _ in range(int(TRANSIT_GONE_SEC / DT) + 6):
            g, *_ = tick(st, target_h, door=False, motion=30.0)
            if g:
                transit_line = g
            if st["mode"] == "discover":
                break
        assert st["mode"] == "discover"
        assert transit_line and "through" in transit_line.lower()
        assert st["rooms_visited"] == 1

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


class TestRoomCap:
    def test_normal_rooms_get_plain_transit_line(self, st):
        for n in range(1, ROOM_CAP):
            g, priority, *_ = force_transit(st)
            assert st["rooms_visited"] == n
            assert g.startswith("You're through.")
            assert "say stop" not in g.lower()
            assert priority

    def test_check_in_at_cap_and_reminders(self, st):
        lines = []
        for _ in range(ROOM_CAP + ROOM_CAP_REMIND):
            g, *_ = force_transit(st)
            lines.append(g)
        # At the cap: the check-in, merged into ONE utterance with the scan cmd.
        cap_line = lines[ROOM_CAP - 1]
        assert f"{ROOM_CAP} rooms" in cap_line
        assert "say stop" in cap_line.lower()
        assert "slowly turn" in cap_line.lower()   # scan instruction still there
        # The room right after the cap: back to the plain line.
        assert "say stop" not in lines[ROOM_CAP].lower()
        # And the reminder fires ROOM_CAP_REMIND rooms later.
        remind_line = lines[ROOM_CAP + ROOM_CAP_REMIND - 1]
        assert "say stop" in remind_line.lower()

    def test_rooms_visited_survives_new_room_reset(self, st):
        force_transit(st)                 # transit -> enter_discover ran inside
        assert st["rooms_visited"] == 1
        st.enter_discover()               # explicit re-scan must not erase it
        assert st["rooms_visited"] == 1


class TestPacing:
    def test_find_door_reprompt_is_time_based(self, st):
        """In go_door with no door in sight, the first 'no door yet' nudge must
        wait out the entry delay (~REPROMPT_SEC of frames), not fire instantly."""
        st.enter_go_door()
        early, later = [], []
        n_early = int((REPROMPT_SEC - 1.0) / DT)         # ~3 s of frames
        for i in range(int(REPROMPT_SEC / DT) + 8):
            g, *_ = tick(st, 0.0, door=False)
            (early if i < n_early else later).append(g)
        assert not any(early), f"nudge fired too early: {[g for g in early if g]}"
        assert any("door" in (g or "").lower() for g in later)

    def test_transit_needs_full_absence_window(self, st):
        """The 'you're through' inference needs TRANSIT_GONE_SEC of doorless
        FRAMES - a couple of ticks (detection flicker) must never fire it."""
        st.enter_go_door()
        st["near_latch"] = True
        st["walk_frames"] = 10            # user definitely walked
        few = int(TRANSIT_GONE_SEC / DT) - 2
        for _ in range(few):
            tick(st, 0.0, door=False, motion=30.0)
        assert st["mode"] == "go_door"    # not yet
        for _ in range(4):
            tick(st, 0.0, door=False, motion=30.0)
        assert st["mode"] == "discover"   # now it fired


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
        a["rooms_visited"] = 3
        a.scan_counts.update({"refrigerator": 3})
        a.door_hist.append(("ahead", 2.0))
        assert b["goal"] is None
        assert b["mode"] == "discover"
        assert b["rooms_visited"] == 0
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
