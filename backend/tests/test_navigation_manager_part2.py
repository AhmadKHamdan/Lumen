"""Manager-level integration tests for Sprint 4 Part 2.

These tests exercise the manager's NEW responsibilities:
- B1+B4: eager resolution; partial-recognition announcement.
- A4: unmappable waypoint -> unknown-landmark dialog.
- E5+E6: detector callback triggers spoken "you've reached the X".
- C7: disambiguation dialog (prompt -> answer -> commit -> reached).
- G1+G2+G3: detection timeout -> recovery dialog -> skip path.

Uses the same fakes (FakeTTS / FakeFSM) as test_navigation_manager.py.
"""

from __future__ import annotations

from typing import Optional

from navigation.context import WaypointStatus
from navigation.detector import LANDMARK_DETECTION_TIMEOUT_SECONDS
from navigation.manager import (
    NavigationTaskManager,
    WAYPOINT_CONFIRMATION_TEMPLATE,
    WAYPOINT_PARTIAL_RECOGNITION_TEMPLATE,
    WAYPOINT_UNKNOWN_TEXT,
    WAYPOINT_REACHED_TEMPLATE,
    NEXT_WAYPOINT_PROMPT_TEXT,
    DETECTION_TIMEOUT_TEXT,
    DISAMBIGUATION_PROMPT_TEMPLATE,
    DISAMBIGUATION_CONFIRMATION_TEMPLATE,
)


# --- shared fakes ------------------------------------------------------

class FakeTTS:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def synthesize(self, text: str) -> bytes:
        self.calls.append(text)
        return f"<AUDIO:{text}>".encode("utf-8")


class FakeSendTTS:
    def __init__(self) -> None:
        self.sent: list[bytes] = []

    def __call__(self, audio: bytes) -> None:
        self.sent.append(audio)


class FakeFSM:
    def __init__(self) -> None:
        self.events: list[tuple[str, Optional[dict]]] = []

    def handle_event(self, event: str, payload: Optional[dict] = None) -> None:
        self.events.append((event, payload))


def _make_manager() -> tuple[NavigationTaskManager, FakeTTS, FakeSendTTS, FakeFSM]:
    tts, send, fsm = FakeTTS(), FakeSendTTS(), FakeFSM()
    return NavigationTaskManager(tts, send, fsm), tts, send, fsm


def _detect(cls: str, conf: float = 0.85,
            region: str = "center", distance: str = "near") -> dict:
    return {
        "class_name": cls, "confidence": conf,
        "region": region, "distance_category": distance,
        "bbox": [100, 100, 400, 400],
    }


# ======================================================================
# B1 + B4: eager resolution + partial-recognition announcement
# ======================================================================

class TestEagerResolution:
    def test_single_mappable_waypoint_stores_classes(self):
        mgr, _tts, _send, _fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        mgr.on_user_response("the doorway")
        wp = mgr.context.current_waypoint()
        assert wp.target_classes == ["door"]
        assert mgr.context.frame_processing_enabled is True

    def test_seat_alias_resolves_to_multiple_classes(self):
        mgr, _tts, _send, _fsm = _make_manager()
        mgr.on_navigation_command("living room")
        mgr.on_user_response("the seat")
        wp = mgr.context.current_waypoint()
        assert set(wp.target_classes) == {"chair", "couch", "bench"}

    def test_partial_recognition_drops_unmappable_and_announces(self):
        # B4 = C: "doorway then turn left" -> keep doorway, drop "turn left",
        # announce the drop BEFORE "Looking for ..." (sub-decision 3).
        mgr, tts, _send, _fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        mgr.on_user_response("doorway then turn left")
        assert [w.normalized_text for w in mgr.context.waypoints] == ["doorway"]
        # Find indices of the two messages.
        partial_idx = next(
            (i for i, c in enumerate(tts.calls)
             if "didn't recognize" in c and "turn left" in c),
            -1,
        )
        looking_idx = next(
            (i for i, c in enumerate(tts.calls) if c.startswith("Looking for")),
            -1,
        )
        assert partial_idx != -1, f"partial-recognition not spoken; tts={tts.calls}"
        assert looking_idx != -1
        # B4 sub 3: rejection FIRST.
        assert partial_idx < looking_idx


# ======================================================================
# A4: all-unmappable -> unknown-landmark prompt
# ======================================================================

class TestAllUnmappable:
    def test_unmappable_single_waypoint_triggers_unknown_dialog(self):
        mgr, tts, _send, _fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        mgr.on_user_response("the unicorn")
        # Still awaiting waypoint, prompt was the unknown-landmark text.
        assert mgr.context.awaiting_waypoint is True
        assert mgr.context.retry_count == 1
        assert WAYPOINT_UNKNOWN_TEXT in tts.calls
        assert mgr.context.frame_processing_enabled is False

    def test_three_unmappable_in_a_row_cancels(self):
        mgr, tts, _send, fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        for _ in range(3):
            mgr.on_user_response("the unicorn")
        # Same give-up path as empty input.
        assert any(ev[0] == "task_cancelled" for ev in fsm.events)
        assert mgr.is_active() is False


# ======================================================================
# E5 + E6: reached announcement via detector callback
# ======================================================================

class TestReachedAnnouncement:
    def test_two_centered_near_frames_trigger_reached_speech(self):
        mgr, tts, _send, _fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        mgr.on_user_response("the chair")
        # 2-of-3 hits, both center+near.
        mgr.on_detections([_detect("chair")])
        mgr.on_detections([_detect("chair")])
        spoken = WAYPOINT_REACHED_TEMPLATE.format(waypoint="chair")
        assert spoken in tts.calls
        assert mgr.context.waypoints[0].status == WaypointStatus.REACHED

    def test_reached_auto_advances_to_next_queued_waypoint(self):
        mgr, tts, _send, _fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        # Two valid waypoints stored.
        mgr.on_user_response("the chair and the table")
        mgr.on_detections([_detect("chair")])
        mgr.on_detections([_detect("chair")])
        assert mgr.context.current_waypoint().normalized_text == "table"
        # Next "Looking for ..." should have been announced.
        assert WAYPOINT_CONFIRMATION_TEMPLATE.format(waypoint="table") in tts.calls

    def test_reached_only_once_per_waypoint(self):
        # E6 = A: don't re-fire reached on subsequent matching frames.
        mgr, tts, _send, _fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        mgr.on_user_response("the chair")
        for _ in range(5):
            mgr.on_detections([_detect("chair")])
        reached_msgs = [c for c in tts.calls if c.startswith("You've reached")]
        assert len(reached_msgs) == 1


# ======================================================================
# C7: full disambiguation dialog
# ======================================================================

class TestDisambiguationDialog:
    def test_two_alias_classes_trigger_prompt(self):
        mgr, tts, _send, _fsm = _make_manager()
        mgr.on_navigation_command("living room")
        mgr.on_user_response("the seat")  # -> ["chair","couch","bench"]
        # Two frames each with chair AND couch (2-of-3 each, on a 3-window).
        mgr.on_detections([
            _detect("chair", region="left", distance="medium"),
            _detect("couch", region="right", distance="medium"),
        ])
        mgr.on_detections([
            _detect("chair", region="center", distance="near"),
            _detect("couch", region="right", distance="medium"),
        ])
        # Disambiguation prompt should have been spoken.
        prompts = [c for c in tts.calls if c.startswith("I see a")]
        assert prompts, f"expected disambig prompt; tts={tts.calls}"
        assert mgr.context.awaiting_disambiguation is True

    def test_user_answer_commits_and_resumes(self):
        mgr, tts, _send, _fsm = _make_manager()
        mgr.on_navigation_command("living room")
        mgr.on_user_response("the seat")
        # Trigger the prompt.
        for _ in range(2):
            mgr.on_detections([
                _detect("chair", region="left", distance="medium"),
                _detect("couch", region="right", distance="medium"),
            ])
        # User picks chair.
        mgr.on_user_response("chair")
        # Confirmation spoken, awaiting_disambiguation cleared.
        assert mgr.context.awaiting_disambiguation is False
        assert mgr.context.current_waypoint().locked_class == "chair"
        confirmation = DISAMBIGUATION_CONFIRMATION_TEMPLATE.format(waypoint="chair")
        assert confirmation in tts.calls

    def test_voice_cancel_during_disambiguation_works(self):
        # C7 sub 5.
        mgr, _tts, _send, fsm = _make_manager()
        mgr.on_navigation_command("living room")
        mgr.on_user_response("the seat")
        for _ in range(2):
            mgr.on_detections([
                _detect("chair", region="left", distance="medium"),
                _detect("couch", region="right", distance="medium"),
            ])
        mgr.on_user_response("stop")
        assert mgr.is_active() is False
        assert any(ev[0] == "task_cancelled" for ev in fsm.events)


# ======================================================================
# G1 + G2 + G3: detection-timeout + recovery + skip
# ======================================================================

class TestDetectionTimeout:
    def test_60s_no_detection_speaks_recovery_prompt(self):
        mgr, tts, _send, _fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        mgr.on_user_response("the chair")
        # Time passes with no detections.
        future = (
            mgr.context.current_waypoint_started_at
            + LANDMARK_DETECTION_TIMEOUT_SECONDS + 1
        )
        mgr.check_timeout(now=future)
        # Recovery prompt should have been spoken, with the waypoint name.
        expected = DETECTION_TIMEOUT_TEXT.format(waypoint="chair")
        assert expected in tts.calls
        assert mgr._awaiting_recovery is True

    def test_skip_command_after_timeout_drops_current_and_asks_for_next(self):
        # G3 = A: with no queued next waypoint, prompt for one.
        mgr, tts, _send, _fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        mgr.on_user_response("the chair")
        future = (
            mgr.context.current_waypoint_started_at
            + LANDMARK_DETECTION_TIMEOUT_SECONDS + 1
        )
        mgr.check_timeout(now=future)
        # User says "skip".
        mgr.on_user_response("skip")
        # The chair waypoint should be SKIPPED, and we should be back in
        # awaiting_waypoint with the next-landmark prompt spoken.
        assert mgr.context.waypoints[0].status == WaypointStatus.SKIPPED
        assert mgr.context.awaiting_waypoint is True
        assert NEXT_WAYPOINT_PROMPT_TEXT in tts.calls

    def test_skip_advances_to_queued_waypoint(self):
        mgr, tts, _send, _fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        mgr.on_user_response("the chair and the table")
        future = (
            mgr.context.current_waypoint_started_at
            + LANDMARK_DETECTION_TIMEOUT_SECONDS + 1
        )
        mgr.check_timeout(now=future)
        mgr.on_user_response("skip")
        # Chair skipped, table becomes active.
        assert mgr.context.waypoints[0].status == WaypointStatus.SKIPPED
        assert mgr.context.current_waypoint().normalized_text == "table"
        assert WAYPOINT_CONFIRMATION_TEMPLATE.format(waypoint="table") in tts.calls

    def test_describing_new_waypoint_after_timeout_works(self):
        # G2 alternate path: user gives a different landmark instead of "skip".
        mgr, _tts, _send, _fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        mgr.on_user_response("the chair")
        future = (
            mgr.context.current_waypoint_started_at
            + LANDMARK_DETECTION_TIMEOUT_SECONDS + 1
        )
        mgr.check_timeout(now=future)
        mgr.on_user_response("the table")
        # Original chair waypoint stays around (active); table is added.
        norms = [w.normalized_text for w in mgr.context.waypoints]
        assert "table" in norms
