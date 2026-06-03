"""End-to-end smoke test for Part 5 — a complete navigation session.

E1 = A: One scripted scenario that drives nav command -> 2 waypoints ->
reached each -> completion event. Proves the parts integrate.

E2 = A: Asserts the FSM received task_completed with the expected
summary payload at the end.
"""

from __future__ import annotations

from typing import Optional

from navigation.context import WaypointStatus
from navigation.manager import (
    NavigationTaskManager,
    NAVIGATION_COMPLETE_TEXT,
    WAYPOINT_REACHED_TEMPLATE,
    WAYPOINT_CONFIRMATION_TEMPLATE,
    WHATS_NEXT_PROMPT_TEXT,
    OBSTACLE_PERSON_LEVEL0_TEMPLATE,
)


# --- Fakes ---

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


def _detect(cls: str, region: str = "center", distance: str = "near") -> dict:
    return {
        "class_name": cls,
        "confidence": 0.9,
        "bbox": [100, 100, 400, 400],
        "region": region,
        "distance_category": distance,
    }


# ======================================================================
# Full session, happy path
# ======================================================================

def test_full_session_two_waypoints_complete():
    """A complete, realistic session:
        1. User says "navigate to kitchen".
        2. User says "the chair and the table".
        3. Chair detected (right far -> right medium -> center near) -> reached.
        4. Table announced; table detected (center far -> center near) -> reached.
        5. "What's next?" prompts; user says "done".
        6. FSM receives task_completed with the right summary.
    """
    tts = FakeTTS()
    send = FakeSendTTS()
    fsm = FakeFSM()
    nav = NavigationTaskManager(tts, send, fsm)

    # 1. Navigate command.
    nav.on_navigation_command("kitchen")
    assert "Where should I take you first?" in tts.calls

    # 2. Two waypoints in one utterance.
    nav.on_user_response("the chair and the table")
    assert WAYPOINT_CONFIRMATION_TEMPLATE.format(waypoint="chair") in tts.calls
    assert nav.context.current_waypoint().normalized_text == "chair"
    assert nav.context.has_more_waypoints() is True

    # 3. Walk to chair through guidance states.
    nav.on_detections([_detect("chair", region="right", distance="far")])
    nav.on_detections([_detect("chair", region="right", distance="far")])
    assert "Chair on your right, far away." in tts.calls

    nav.on_detections([_detect("chair", region="right", distance="medium")])
    assert "Chair on your right, getting closer." in tts.calls

    # Reach the chair (center near).
    nav.on_detections([_detect("chair", region="center", distance="near")])
    assert WAYPOINT_REACHED_TEMPLATE.format(waypoint="chair") in tts.calls

    # 4. Table announced automatically.
    assert WAYPOINT_CONFIRMATION_TEMPLATE.format(waypoint="table") in tts.calls
    assert nav.context.current_waypoint().normalized_text == "table"

    # Walk to table (using dining table since "table" maps to it).
    nav.on_detections([_detect("dining table", region="center", distance="far")])
    nav.on_detections([_detect("dining table", region="center", distance="far")])
    # Center+far should produce a "Table far ahead." cue; we use the
    # locked/display name actually emitted by manager. Just verify reached
    # below; the exact first-cue phrasing for "dining table" is tested
    # elsewhere.

    # Reach the table.
    nav.on_detections([_detect("dining table", region="center", distance="near")])
    assert "You've reached the table." in tts.calls

    # 5. "What's next?" prompt should now be active.
    assert WHATS_NEXT_PROMPT_TEXT in tts.calls
    assert nav.context.awaiting_next_or_done is True

    # User says done.
    nav.on_user_response("done")
    assert NAVIGATION_COMPLETE_TEXT in tts.calls

    # 6. FSM event check (E2 = A).
    completed = [ev for ev in fsm.events if ev[0] == "task_completed"]
    assert len(completed) == 1
    summary = completed[0][1]["summary"]
    assert summary["destination"] == "kitchen"
    assert summary["waypoints_reached"] == 2
    assert summary["waypoints_skipped"] == 0
    assert summary["duration_seconds"] >= 0  # depends on real time

    # Task is fully torn down.
    assert nav.is_active() is False


# ======================================================================
# Full session with an obstacle interruption mid-walk
# ======================================================================

def test_full_session_with_obstacle_then_recovers():
    """Same flow but a person blocks the path before chair is reached.
    Verifies that obstacle preemption doesn't break the reached path —
    once the person clears, reached still fires.
    """
    tts = FakeTTS()
    send = FakeSendTTS()
    fsm = FakeFSM()
    nav = NavigationTaskManager(tts, send, fsm)

    nav.on_navigation_command("kitchen")
    nav.on_user_response("the chair")

    # Chair visible + person in path.
    nav.on_detections([
        _detect("chair", region="center", distance="near"),
        _detect("person", region="center", distance="near"),
    ])
    nav.on_detections([
        _detect("chair", region="center", distance="near"),
        _detect("person", region="center", distance="near"),
    ])
    # Person warned; chair NOT yet announced reached.
    assert OBSTACLE_PERSON_LEVEL0_TEMPLATE in tts.calls
    assert WAYPOINT_REACHED_TEMPLATE.format(waypoint="chair") not in tts.calls

    # Person clears; chair still center+near.
    nav.on_detections([_detect("chair", region="center", distance="near")])
    assert WAYPOINT_REACHED_TEMPLATE.format(waypoint="chair") in tts.calls

    # Complete.
    nav.on_user_response("done")
    completed = [ev for ev in fsm.events if ev[0] == "task_completed"]
    assert len(completed) == 1


# ======================================================================
# Full session with "I'm here" override
# ======================================================================

def test_full_session_with_im_here_override():
    """User declares arrival manually; detector hasn't seen anything.
    Confirmation dialog asks back; user says yes; task completes."""
    tts = FakeTTS()
    send = FakeSendTTS()
    fsm = FakeFSM()
    nav = NavigationTaskManager(tts, send, fsm)

    nav.on_navigation_command("kitchen")
    nav.on_user_response("the chair")
    # User declares arrival with no detections.
    nav.on_user_response("I'm here")
    # Confirmation should be pending.
    assert nav.context.awaiting_arrival_confirmation is True
    # User confirms.
    nav.on_user_response("yes")
    # Reached fires + "What's next?".
    assert WAYPOINT_REACHED_TEMPLATE.format(waypoint="chair") in tts.calls
    assert WHATS_NEXT_PROMPT_TEXT in tts.calls
    assert nav.context.waypoints[0].status == WaypointStatus.REACHED

    nav.on_user_response("done")
    completed = [ev for ev in fsm.events if ev[0] == "task_completed"]
    assert len(completed) == 1
    assert completed[0][1]["summary"]["waypoints_reached"] == 1


# ======================================================================
# Full session with cancellation
# ======================================================================

def test_full_session_cancelled_emits_task_cancelled():
    tts = FakeTTS()
    send = FakeSendTTS()
    fsm = FakeFSM()
    nav = NavigationTaskManager(tts, send, fsm)

    nav.on_navigation_command("kitchen")
    nav.on_user_response("the chair")
    nav.on_user_response("stop")

    cancelled = [ev for ev in fsm.events if ev[0] == "task_cancelled"]
    assert len(cancelled) == 1
    completed = [ev for ev in fsm.events if ev[0] == "task_completed"]
    assert completed == []
    assert nav.is_active() is False
