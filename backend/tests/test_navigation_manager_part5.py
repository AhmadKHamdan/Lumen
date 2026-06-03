"""Manager-level tests for Part 5 — task completion, arrival override,
'What's next?', FSM events, status query, dict adapter."""

from __future__ import annotations

from typing import Optional

from navigation.context import WaypointStatus
from navigation.detector import LANDMARK_DETECTION_TIMEOUT_SECONDS
from navigation.manager import (
    NavigationTaskManager,
    create_navigation_session,
    NAVIGATION_COMPLETE_TEXT,
    WHATS_NEXT_PROMPT_TEXT,
    COMPLETION_TIMEOUT_TEXT,
    ARRIVAL_CONFIRMATION_TEMPLATE,
    ARRIVAL_CONFIRMATION_NO_OBSERVATION_TEMPLATE,
    ARRIVAL_CONFIRMED_DECLINED_TEXT,
    WAYPOINT_PROMPT_TIMEOUT_SECONDS,
    WAYPOINT_REACHED_TEMPLATE,
    WAYPOINT_PROMPT_TEXT,
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
# Task completion (A1 + A2 + C1 + C2)
# ======================================================================

class TestTaskCompletion:
    def test_reached_with_empty_queue_prompts_whats_next(self):
        # A2 = A: queue empty after reached -> "What's next?"
        mgr, tts, _send, _fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        mgr.on_user_response("the chair")
        mgr.on_detections([_detect("chair")])
        mgr.on_detections([_detect("chair")])
        # Reached, then "What's next?" prompt.
        assert WAYPOINT_REACHED_TEMPLATE.format(waypoint="chair") in tts.calls
        assert WHATS_NEXT_PROMPT_TEXT in tts.calls
        assert mgr.context.awaiting_next_or_done is True
        assert mgr.context.frame_processing_enabled is False  # paused

    def test_user_says_done_completes_task(self):
        mgr, tts, _send, fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        mgr.on_user_response("the chair")
        mgr.on_detections([_detect("chair")])
        mgr.on_detections([_detect("chair")])
        # Now user says "done".
        mgr.on_user_response("done")
        # Navigation complete spoken + FSM event fired.
        assert NAVIGATION_COMPLETE_TEXT in tts.calls
        completed = [ev for ev in fsm.events if ev[0] == "task_completed"]
        assert len(completed) == 1
        payload = completed[0][1]
        assert payload["task"] == "navigation"
        summary = payload["summary"]
        assert summary["destination"] == "kitchen"
        assert summary["waypoints_reached"] == 1
        assert summary["waypoints_skipped"] == 0
        assert mgr.is_active() is False

    def test_user_adds_new_waypoint_after_whats_next(self):
        # A2 = A: user can also say a landmark instead of "done".
        mgr, tts, _send, _fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        mgr.on_user_response("the chair")
        mgr.on_detections([_detect("chair")])
        mgr.on_detections([_detect("chair")])
        # Task continues with a new waypoint.
        mgr.on_user_response("the table")
        # Looking-for-table spoken; new waypoint stored; task still active.
        assert "Looking for table." in tts.calls
        assert mgr.is_active() is True
        assert mgr.context.current_waypoint().normalized_text == "table"
        assert mgr.context.awaiting_next_or_done is False

    def test_whats_next_timeout_auto_completes(self):
        # A4 = A: 30s silence after "What's next?" -> auto-complete.
        mgr, tts, _send, fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        mgr.on_user_response("the chair")
        mgr.on_detections([_detect("chair")])
        mgr.on_detections([_detect("chair")])
        # Fire the timeout.
        future = mgr.context.last_prompt_at + WAYPOINT_PROMPT_TIMEOUT_SECONDS + 1
        mgr.check_timeout(now=future)
        assert COMPLETION_TIMEOUT_TEXT in tts.calls
        assert any(ev[0] == "task_completed" for ev in fsm.events)
        assert mgr.is_active() is False

    def test_stop_during_whats_next_cancels(self):
        mgr, _tts, _send, fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        mgr.on_user_response("the chair")
        mgr.on_detections([_detect("chair")])
        mgr.on_detections([_detect("chair")])
        mgr.on_user_response("stop")
        assert any(ev[0] == "task_cancelled" for ev in fsm.events)
        assert mgr.is_active() is False

    def test_completion_summary_includes_skipped(self):
        # Skip one, reach one. Make sure summary counts both correctly.
        mgr, _tts, _send, fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        mgr.on_user_response("the chair and the table")
        # Skip the chair via the recovery flow.
        future = (
            mgr.context.current_waypoint_started_at
            + LANDMARK_DETECTION_TIMEOUT_SECONDS + 1
        )
        mgr.check_timeout(now=future)
        mgr.on_user_response("skip")
        # Now reach the table.
        mgr.on_detections([_detect("dining table")])
        mgr.on_detections([_detect("dining table")])
        # "What's next?" should have fired. Say done.
        mgr.on_user_response("done")
        # Summary.
        completed = [ev for ev in fsm.events if ev[0] == "task_completed"]
        assert len(completed) == 1
        summary = completed[0][1]["summary"]
        assert summary["waypoints_reached"] == 1
        assert summary["waypoints_skipped"] == 1


# ======================================================================
# I'm here override (B1 keep confirmation + B2 + B3 + B4 + B5)
# ======================================================================

class TestArrivalOverride:
    def test_im_here_when_detector_disagrees_asks_for_confirmation(self):
        # No detections seen -> ARRIVAL_CONFIRMATION_NO_OBSERVATION template.
        mgr, tts, _send, _fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        mgr.on_user_response("the chair")
        # No detections at all.
        mgr.on_user_response("I'm here")
        # Confirmation prompt should have fired with the no-observation form.
        expected = ARRIVAL_CONFIRMATION_NO_OBSERVATION_TEMPLATE.format(waypoint="chair")
        assert expected in tts.calls
        assert mgr.context.awaiting_arrival_confirmation is True

    def test_im_here_then_yes_marks_reached(self):
        mgr, tts, _send, _fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        mgr.on_user_response("the chair")
        mgr.on_user_response("I'm here")
        mgr.on_user_response("yes")
        # Reached announcement + "What's next?" should both follow.
        assert WAYPOINT_REACHED_TEMPLATE.format(waypoint="chair") in tts.calls
        assert WHATS_NEXT_PROMPT_TEXT in tts.calls
        # Waypoint marked REACHED.
        assert mgr.context is not None  # awaiting next or done
        wp_status = mgr.context.waypoints[0].status
        assert wp_status == WaypointStatus.REACHED

    def test_im_here_then_no_keeps_going(self):
        mgr, tts, _send, _fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        mgr.on_user_response("the chair")
        mgr.on_user_response("I'm here")
        mgr.on_user_response("no")
        # Should NOT have marked reached; should keep the task active.
        assert mgr.context.awaiting_arrival_confirmation is False
        assert mgr.context.waypoints[0].status != WaypointStatus.REACHED
        assert ARRIVAL_CONFIRMED_DECLINED_TEXT in tts.calls
        assert mgr.is_active() is True

    def test_im_here_during_collection_re_prompts(self):
        # B3: "I'm here" before any waypoint exists -> nothing to arrive at.
        mgr, tts, _send, _fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        mgr.on_user_response("I'm here")
        # Should NOT mark anything as reached. Should re-prompt.
        any_arrival_msg = [c for c in tts.calls if "no destination yet" in c.lower()]
        assert any_arrival_msg, f"expected no-destination reprompt; tts={tts.calls}"
        assert mgr.context.awaiting_waypoint is True

    def test_im_here_with_visible_detection_uses_observation_template(self):
        # If we've seen the chair on the right far, the prompt should
        # include that observation.
        mgr, tts, _send, _fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        mgr.on_user_response("the chair")
        # Build up some history with a confirmed visible chair.
        mgr.on_detections([_detect("chair", region="right", distance="far")])
        mgr.on_detections([_detect("chair", region="right", distance="far")])
        # Now "I'm here".
        mgr.on_user_response("I'm here")
        # Confirmation prompt should mention seeing the chair.
        relevant = [c for c in tts.calls if c.endswith(
            "Are you at the chair? Say 'yes' to confirm or 'no' to keep going."
        )]
        assert relevant, f"expected observation-based prompt; tts={tts.calls}"

    def test_voice_cancel_during_arrival_confirmation(self):
        mgr, _tts, _send, fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        mgr.on_user_response("the chair")
        mgr.on_user_response("I'm here")
        mgr.on_user_response("stop")
        assert any(ev[0] == "task_cancelled" for ev in fsm.events)


# ======================================================================
# F2: re-issued navigate command cancels current, restarts
# ======================================================================

class TestReissuedCommand:
    def test_new_command_cancels_active_task(self):
        mgr, tts, _send, _fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        mgr.on_user_response("the chair")
        # Now user says navigate to bathroom (FSM would call on_navigation_command).
        mgr.on_navigation_command("bathroom")
        # New task active, destination changed, asked for a fresh first landmark.
        assert mgr.is_active() is True
        assert mgr.context.destination == "bathroom"
        assert WAYPOINT_PROMPT_TEXT in tts.calls


# ======================================================================
# F4: get_status query API
# ======================================================================

class TestGetStatus:
    def test_status_inactive(self):
        mgr, _tts, _send, _fsm = _make_manager()
        s = mgr.get_status()
        assert s == {"active": False}

    def test_status_active(self):
        mgr, _tts, _send, _fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        mgr.on_user_response("the chair")
        s = mgr.get_status()
        assert s["active"] is True
        assert s["destination"] == "kitchen"
        assert s["current_waypoint"] == "chair"
        assert s["awaiting_waypoint"] is False
        assert s["frame_processing_enabled"] is True

    def test_status_during_disambiguation(self):
        mgr, _tts, _send, _fsm = _make_manager()
        mgr.on_navigation_command("living room")
        mgr.on_user_response("the seat")
        # Trigger disambig.
        for _ in range(2):
            mgr.on_detections([
                _detect("chair", region="left", distance="medium"),
                _detect("couch", region="right", distance="medium"),
            ])
        s = mgr.get_status()
        assert s["awaiting_disambiguation"] is True


# ======================================================================
# D3: detection-dict adapter
# ======================================================================

class TestDictAdapter:
    def test_class_alias_accepted(self):
        # Ahmad might send {"class": "chair"} instead of {"class_name": ...}.
        mgr, _tts, _send, _fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        mgr.on_user_response("the chair")
        # Use the "class" key form.
        mgr.on_detections([{
            "class": "chair",
            "confidence": 0.9,
            "bbox": [100, 100, 400, 400],
            "region": "center",
            "distance_category": "near",
        }])
        # History should contain the chair detection.
        assert mgr.context.detection_history[-1] == {"chair"}

    def test_conf_alias_accepted(self):
        mgr, _tts, _send, _fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        mgr.on_user_response("the chair")
        mgr.on_detections([{
            "class_name": "chair",
            "conf": 0.9,                # not "confidence"
            "bbox": [100, 100, 400, 400],
            "region": "center",
            "distance_category": "near",
        }])
        assert mgr.context.detection_history[-1] == {"chair"}

    def test_malformed_dict_dropped(self):
        # No class identifier at all -> dropped silently.
        mgr, _tts, _send, _fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        mgr.on_user_response("the chair")
        mgr.on_detections([{"foo": "bar"}])
        # No exception; nothing added.
        # detection_history could be [] or have an empty set; either is fine.
        assert mgr.is_active() is True


# ======================================================================
# D4: create_navigation_session helper
# ======================================================================

class TestWiringHelper:
    def test_helper_constructs_manager(self):
        tts, send, fsm = FakeTTS(), FakeSendTTS(), FakeFSM()
        nav = create_navigation_session(
            tts_service=tts, fsm=fsm, send_tts=send,
        )
        assert isinstance(nav, NavigationTaskManager)
        nav.on_navigation_command("kitchen")
        assert nav.is_active() is True


# ======================================================================
# F3: FSM event failure doesn't crash session
# ======================================================================

class TestFSMResilience:
    def test_fsm_raising_does_not_crash_cancel(self):
        class BrokenFSM:
            def handle_event(self, event, payload=None):
                raise RuntimeError("boom")
        tts, send = FakeTTS(), FakeSendTTS()
        nav = NavigationTaskManager(tts, send, BrokenFSM())
        nav.on_navigation_command("kitchen")
        # Cancel should log but not crash.
        nav.cancel()
        # Manager state cleaned up despite FSM error.
        assert nav.is_active() is False

    def test_fsm_raising_does_not_crash_complete(self):
        class BrokenFSM:
            def handle_event(self, event, payload=None):
                raise RuntimeError("boom")
        tts, send = FakeTTS(), FakeSendTTS()
        nav = NavigationTaskManager(tts, send, BrokenFSM())
        nav.on_navigation_command("kitchen")
        nav.on_user_response("the chair")
        nav.on_detections([_detect("chair")])
        nav.on_detections([_detect("chair")])
        nav.on_user_response("done")
        # Even though FSM raised, completion announcement happened + state cleaned.
        assert NAVIGATION_COMPLETE_TEXT in tts.calls
        assert nav.is_active() is False
