"""Manager-level tests for Part 3 — obstacle warnings & phrasings."""

from __future__ import annotations

from typing import Optional

from navigation.manager import (
    NavigationTaskManager,
    OBSTACLE_WARNING_LEVEL0_TEMPLATE,
    OBSTACLE_WARNING_LEVEL1_TEMPLATE,
    OBSTACLE_PERSON_LEVEL0_TEMPLATE,
    OBSTACLE_PERSON_LEVEL1_TEMPLATE,
    OBSTACLE_PERSON_LEVEL2_TEMPLATE,
    WAYPOINT_REACHED_TEMPLATE,
)
from navigation.detector import OBSTACLE_WARNING_COOLDOWN_SECONDS


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


def _make_manager():
    tts, send, fsm = FakeTTS(), FakeSendTTS(), FakeFSM()
    return NavigationTaskManager(tts, send, fsm), tts, send, fsm


def _detect(cls: str, region: str = "center", distance: str = "near",
            conf: float = 0.85, bbox=(100, 100, 400, 400)) -> dict:
    return {
        "class_name": cls, "confidence": conf,
        "region": region, "distance_category": distance,
        "bbox": list(bbox),
    }


# ======================================================================
# Spoken phrasing per category and level
# ======================================================================

class TestSpokenPhrasing:
    def test_person_level0_speaks_person_ahead(self):
        mgr, tts, _send, _fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        mgr.on_user_response("the chair")
        # Person in path.
        mgr.on_detections([_detect("person")])
        assert OBSTACLE_PERSON_LEVEL0_TEMPLATE in tts.calls

    def test_furniture_level0_speaks_class_capitalized(self):
        mgr, tts, _send, _fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        mgr.on_user_response("the doorway")
        mgr.on_detections([_detect("couch")])
        expected = OBSTACLE_WARNING_LEVEL0_TEMPLATE.format(noun="Couch")
        assert expected in tts.calls

    def test_tv_display_name_overridden(self):
        mgr, tts, _send, _fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        mgr.on_user_response("the doorway")
        mgr.on_detections([_detect("tv")])
        expected = OBSTACLE_WARNING_LEVEL0_TEMPLATE.format(noun="TV")
        assert expected in tts.calls

    def test_escalation_level1_phrasing(self):
        mgr, tts, _send, _fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        mgr.on_user_response("the doorway")
        mgr.on_detections([_detect("person")])
        # Manually advance cooldown so the next call counts as a re-warning.
        mgr.context.last_obstacle_warning_at["person"] -= (
            OBSTACLE_WARNING_COOLDOWN_SECONDS + 1
        )
        mgr.on_detections([_detect("person")])
        # Level-1 person phrasing should be present.
        assert OBSTACLE_PERSON_LEVEL1_TEMPLATE in tts.calls

    def test_escalation_caps_at_level2(self):
        mgr, tts, _send, _fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        mgr.on_user_response("the doorway")
        for _ in range(5):
            mgr.on_detections([_detect("person")])
            mgr.context.last_obstacle_warning_at["person"] -= (
                OBSTACLE_WARNING_COOLDOWN_SECONDS + 1
            )
        # Eventually level-2 phrasing fires; level-2 is the cap.
        assert OBSTACLE_PERSON_LEVEL2_TEMPLATE in tts.calls


# ======================================================================
# Preemption end-to-end
# ======================================================================

class TestPreemptionThroughManager:
    def test_obstacle_suppresses_reached_speech(self):
        mgr, tts, _send, _fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        mgr.on_user_response("the chair")
        # Two frames where chair is centered+near (would fire reached) AND
        # a person is in the way (should fire warning and suppress reached).
        for _ in range(2):
            mgr.on_detections([_detect("chair"), _detect("person")])
        # Reached announcement should NOT have been spoken.
        reached_line = WAYPOINT_REACHED_TEMPLATE.format(waypoint="chair")
        assert reached_line not in tts.calls
        # The obstacle warning WAS spoken.
        assert OBSTACLE_PERSON_LEVEL0_TEMPLATE in tts.calls


# ======================================================================
# Active landmark is never an obstacle
# ======================================================================

class TestLandmarkExempt:
    def test_chair_as_target_is_not_warned_about(self):
        mgr, tts, _send, _fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        mgr.on_user_response("the chair")
        mgr.on_detections([_detect("chair")])  # one frame, chair only
        # No obstacle warning was spoken.
        chair_warnings = [
            c for c in tts.calls
            if c.startswith("Chair")
        ]
        # The "Looking for chair." line was spoken, but it doesn't start
        # with capital Chair — it starts with "Looking". The only line
        # starting with capital Chair would be an obstacle warning.
        assert chair_warnings == []
