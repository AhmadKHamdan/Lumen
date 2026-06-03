"""Manager-level tests for Part 4 — guidance spoken phrasings."""

from __future__ import annotations

from typing import Optional

from navigation.manager import (
    NavigationTaskManager,
    GUIDANCE_LOST_SIGHT_TEMPLATE,
)


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
# Phrasing for first_cue across regions and distances
# ======================================================================

class TestFirstCuePhrasing:
    def test_right_far(self):
        mgr, tts, _send, _fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        mgr.on_user_response("the chair")
        # Detection on right, far. Two frames -> temporal consistency.
        mgr.on_detections([_detect("chair", region="right", distance="far")])
        mgr.on_detections([_detect("chair", region="right", distance="far")])
        assert "Chair on your right, far away." in tts.calls

    def test_left_medium(self):
        mgr, tts, _send, _fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        mgr.on_user_response("the chair")
        mgr.on_detections([_detect("chair", region="left", distance="medium")])
        mgr.on_detections([_detect("chair", region="left", distance="medium")])
        assert "Chair on your left, approaching." in tts.calls

    def test_center_far(self):
        mgr, tts, _send, _fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        mgr.on_user_response("the chair")
        mgr.on_detections([_detect("chair", region="center", distance="far")])
        mgr.on_detections([_detect("chair", region="center", distance="far")])
        # Center+far -> "Chair far ahead." (no double "ahead")
        assert "Chair far ahead." in tts.calls

    def test_center_medium(self):
        mgr, tts, _send, _fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        mgr.on_user_response("the chair")
        mgr.on_detections([_detect("chair", region="center", distance="medium")])
        mgr.on_detections([_detect("chair", region="center", distance="medium")])
        # Center + medium for first_cue uses the same "getting closer"
        # phrasing as the distance-changed case (handled in manager).
        assert "Chair ahead, getting closer." in tts.calls

    def test_right_near_course_correction(self):
        # D4 = A: near but off-center is a course-correction moment.
        mgr, tts, _send, _fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        mgr.on_user_response("the chair")
        mgr.on_detections([_detect("chair", region="right", distance="near")])
        mgr.on_detections([_detect("chair", region="right", distance="near")])
        assert "Chair on your right, right in front of you." in tts.calls

    def test_tv_uses_display_override(self):
        mgr, tts, _send, _fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        mgr.on_user_response("the tv")
        mgr.on_detections([_detect("tv", region="right", distance="medium")])
        mgr.on_detections([_detect("tv", region="right", distance="medium")])
        assert "TV on your right, approaching." in tts.calls


# ======================================================================
# Distance-only change uses "getting closer"
# ======================================================================

class TestGettingCloser:
    def test_medium_to_medium_same_region_no_speech(self):
        mgr, tts, _send, _fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        mgr.on_user_response("the chair")
        mgr.on_detections([_detect("chair", region="right", distance="far")])
        mgr.on_detections([_detect("chair", region="right", distance="far")])
        before = len(tts.calls)
        # Identical state -> no new cue.
        mgr.on_detections([_detect("chair", region="right", distance="far")])
        assert len(tts.calls) == before

    def test_distance_to_medium_uses_getting_closer(self):
        mgr, tts, _send, _fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        mgr.on_user_response("the chair")
        # Start at right, far.
        mgr.on_detections([_detect("chair", region="right", distance="far")])
        mgr.on_detections([_detect("chair", region="right", distance="far")])
        # Now closer to medium.
        mgr.on_detections([_detect("chair", region="right", distance="medium")])
        assert "Chair on your right, getting closer." in tts.calls

    def test_medium_to_near_announces_right_in_front(self):
        mgr, tts, _send, _fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        mgr.on_user_response("the chair")
        mgr.on_detections([_detect("chair", region="right", distance="medium")])
        mgr.on_detections([_detect("chair", region="right", distance="medium")])
        # Closer to near.
        mgr.on_detections([_detect("chair", region="right", distance="near")])
        assert "Chair on your right, right in front of you." in tts.calls


# ======================================================================
# Region change re-fires full cue
# ======================================================================

class TestRegionChange:
    def test_right_to_left_full_cue(self):
        mgr, tts, _send, _fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        mgr.on_user_response("the chair")
        mgr.on_detections([_detect("chair", region="right", distance="medium")])
        mgr.on_detections([_detect("chair", region="right", distance="medium")])
        # Suddenly on the left, same distance.
        mgr.on_detections([_detect("chair", region="left", distance="medium")])
        assert "Chair on your left, approaching." in tts.calls


# ======================================================================
# Lost sight + return
# ======================================================================

class TestLostSightSpoken:
    def test_lost_sight_announces(self):
        mgr, tts, _send, _fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        mgr.on_user_response("the chair")
        # See chair, speak first cue.
        mgr.on_detections([_detect("chair", region="right", distance="far")])
        mgr.on_detections([_detect("chair", region="right", distance="far")])
        # Now empty frame.
        mgr.on_detections([])
        expected = GUIDANCE_LOST_SIGHT_TEMPLATE.format(waypoint="chair")
        assert expected in tts.calls


# ======================================================================
# Obstacle preempts guidance
# ======================================================================

class TestObstacleSuppression:
    def test_obstacle_suppresses_guidance_speech(self):
        mgr, tts, _send, _fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        mgr.on_user_response("the chair")
        # Build temporal consistency on chair (right far).
        mgr.on_detections([_detect("chair", region="right", distance="far")])
        mgr.on_detections([_detect("chair", region="right", distance="far")])
        # Now: chair shifts to center far (would trigger region_changed
        # cue) but person obstacle present -> obstacle preempts.
        before = len(tts.calls)
        mgr.on_detections([
            _detect("chair", region="center", distance="far"),
            _detect("person", region="center", distance="near"),
        ])
        # Person obstacle should be spoken, but not the "Chair ahead..." cue.
        new_lines = tts.calls[before:]
        assert "Person ahead." in new_lines
        # No chair guidance line.
        assert not any("Chair" in line and "Person" not in line
                       for line in new_lines)
