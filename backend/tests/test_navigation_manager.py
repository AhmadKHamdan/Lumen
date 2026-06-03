"""Unit tests for navigation.manager.

We inject simple stubs for TTS, send_tts callable, and the FSM so the
manager can be exercised end-to-end without partners' Sprint 1–3 code.
"""

from __future__ import annotations

import time
from typing import Optional

from navigation.manager import (
    NavigationTaskManager,
    WAYPOINT_PROMPT_TEXT,
    WAYPOINT_REPROMPT_TEXT,
    WAYPOINT_CONFIRMATION_TEMPLATE,
    WAYPOINT_CANCELLED_TEXT,
    WAYPOINT_TIMEOUT_TEXT,
    WAYPOINT_GIVEUP_TEXT,
    WAYPOINT_PROMPT_TIMEOUT_SECONDS,
    MAX_WAYPOINT_REPROMPTS,
)


# ----------------------------------------------------------------------
# Stubs
# ----------------------------------------------------------------------

class FakeTTS:
    """Records every synthesize() call. Returns a deterministic byte tag."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def synthesize(self, text: str) -> bytes:
        self.calls.append(text)
        return f"<AUDIO:{text}>".encode("utf-8")


class FakeSendTTS:
    """Records every audio blob pushed to the WebSocket."""

    def __init__(self) -> None:
        self.sent: list[bytes] = []

    def __call__(self, audio: bytes) -> None:
        self.sent.append(audio)


class FakeFSM:
    """Records FSM events. Never raises."""

    def __init__(self) -> None:
        self.events: list[tuple[str, Optional[dict]]] = []

    def handle_event(self, event: str, payload: Optional[dict] = None) -> None:
        self.events.append((event, payload))


def _make_manager() -> tuple[NavigationTaskManager, FakeTTS, FakeSendTTS, FakeFSM]:
    tts = FakeTTS()
    send = FakeSendTTS()
    fsm = FakeFSM()
    mgr = NavigationTaskManager(tts_service=tts, send_tts=send, fsm=fsm)
    return mgr, tts, send, fsm


# ----------------------------------------------------------------------
# on_navigation_command — entry from FSM
# ----------------------------------------------------------------------

class TestOnNavigationCommand:
    def test_creates_context_and_prompts(self):
        mgr, tts, send, _fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        assert mgr.is_active() is True
        assert mgr.context.destination == "kitchen"
        assert mgr.context.awaiting_waypoint is True
        # Decision 7 override: "Where should I take you first?"
        assert tts.calls == [WAYPOINT_PROMPT_TEXT]
        assert len(send.sent) == 1

    def test_frame_processing_off_at_entry(self):
        # Decision 10.
        mgr, *_ = _make_manager()
        mgr.on_navigation_command("kitchen")
        assert mgr.is_frame_processing_enabled() is False

    def test_destination_is_trimmed(self):
        mgr, *_ = _make_manager()
        mgr.on_navigation_command("  kitchen  ")
        assert mgr.context.destination == "kitchen"


# ----------------------------------------------------------------------
# on_user_response — waypoint collection
# ----------------------------------------------------------------------

class TestWaypointCollection:
    def test_valid_waypoint_stored_and_confirmed(self):
        mgr, tts, _send, _fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        mgr.on_user_response("go through the doorway")
        # Waypoint stored.
        assert len(mgr.context.waypoints) == 1
        wp = mgr.context.waypoints[0]
        assert wp.raw_text == "go through the doorway"
        assert wp.normalized_text == "doorway"
        # Sub-state exited.
        assert mgr.context.awaiting_waypoint is False
        # Frame processing on (decision 10).
        assert mgr.is_frame_processing_enabled() is True
        # Decision 14 + 15: confirmation spoken.
        assert tts.calls[-1] == WAYPOINT_CONFIRMATION_TEMPLATE.format(waypoint="doorway")

    def test_multi_waypoint_utterance(self):
        # Decision 8 + B4 = C. "doorway" maps to ["door"]; "turn left" does
        # not map (it's a direction, not a landmark). Per B4=C we accept the
        # mappable one and announce which was dropped.
        mgr, tts, _send, _fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        mgr.on_user_response("doorway then turn left")
        assert [w.normalized_text for w in mgr.context.waypoints] == ["doorway"]
        # First (and only stored) waypoint is the active target.
        assert mgr.context.current_waypoint().normalized_text == "doorway"
        # "I didn't recognize: turn left." should have been spoken before the
        # "Looking for ..." confirmation.
        rejected = [c for c in tts.calls if "didn't recognize" in c]
        assert rejected, f"expected partial-recognition announcement; tts={tts.calls}"

    def test_empty_response_triggers_reprompt(self):
        # Decision 12.
        mgr, tts, _send, _fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        mgr.on_user_response("")
        assert mgr.context.retry_count == 1
        assert mgr.context.awaiting_waypoint is True
        assert tts.calls[-1] == WAYPOINT_REPROMPT_TEXT

    def test_filler_only_response_triggers_reprompt(self):
        mgr, tts, _send, _fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        mgr.on_user_response("go to the")
        assert mgr.context.retry_count == 1
        assert tts.calls[-1] == WAYPOINT_REPROMPT_TEXT

    def test_three_bad_responses_cancel(self):
        # Decision 12: up to 2 re-prompts (so 3rd failure → cancel).
        mgr, tts, _send, fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        for _ in range(MAX_WAYPOINT_REPROMPTS + 1):
            mgr.on_user_response("")
        # Last spoken line is the give-up text.
        assert WAYPOINT_GIVEUP_TEXT in tts.calls
        # FSM was told.
        assert any(ev[0] == "task_cancelled" for ev in fsm.events)
        assert mgr.is_active() is False

    def test_retry_count_resets_after_success(self):
        mgr, _tts, _send, _fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        mgr.on_user_response("")        # retry 1
        mgr.on_user_response("doorway")  # success
        assert mgr.context.retry_count == 0


# ----------------------------------------------------------------------
# Voice cancellation (decision 11)
# ----------------------------------------------------------------------

class TestVoiceCancel:
    def test_stop_during_waypoint_collection_cancels(self):
        mgr, tts, _send, fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        mgr.on_user_response("stop")
        assert mgr.is_active() is False
        assert any(ev[0] == "task_cancelled" for ev in fsm.events)
        assert WAYPOINT_CANCELLED_TEXT in tts.calls

    def test_cancel_after_waypoints_added(self):
        mgr, _tts, _send, fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        mgr.on_user_response("doorway")
        # We are now past awaiting_waypoint; voice cancel should still work.
        mgr.on_user_response("never mind")
        assert mgr.is_active() is False
        assert any(ev[0] == "task_cancelled" for ev in fsm.events)


# ----------------------------------------------------------------------
# Timeout (decision 13)
# ----------------------------------------------------------------------

class TestTimeout:
    def test_within_window_no_timeout(self):
        mgr, *_ = _make_manager()
        mgr.on_navigation_command("kitchen")
        # 10 seconds after the prompt — under the 30s threshold.
        future = mgr.context.last_prompt_at + 10
        assert mgr.check_timeout(now=future) is False
        assert mgr.is_active() is True

    def test_at_threshold_fires(self):
        mgr, tts, _send, fsm = _make_manager()
        mgr.on_navigation_command("kitchen")
        future = mgr.context.last_prompt_at + WAYPOINT_PROMPT_TIMEOUT_SECONDS
        assert mgr.check_timeout(now=future) is True
        assert mgr.is_active() is False
        assert WAYPOINT_TIMEOUT_TEXT in tts.calls
        assert any(ev[0] == "task_cancelled" for ev in fsm.events)

    def test_prompt_timeout_does_not_fire_after_waypoint_stored(self):
        # The 30-second WAYPOINT_PROMPT timer should not fire once we have
        # left the awaiting_waypoint phase. (The 60-second DETECTION timer is
        # a separate concern tested in test_navigation_detector.py.)
        from navigation.detector import LANDMARK_DETECTION_TIMEOUT_SECONDS
        mgr, *_ = _make_manager()
        mgr.on_navigation_command("kitchen")
        mgr.on_user_response("doorway")  # exits awaiting_waypoint, spins up detector
        # Pick a "future" within the detection-timeout window so we only
        # exercise the prompt-timeout branch.
        safe_future = mgr.context.last_prompt_at + LANDMARK_DETECTION_TIMEOUT_SECONDS - 5
        assert mgr.check_timeout(now=safe_future) is False
        assert mgr.is_active() is True

    def test_check_timeout_with_no_context_safe(self):
        mgr, *_ = _make_manager()
        # No task started — check_timeout must be a no-op, not crash.
        assert mgr.check_timeout() is False


# ----------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------

class TestEdgeCases:
    def test_on_user_response_without_context_is_noop(self):
        mgr, *_ = _make_manager()
        # Should not raise.
        mgr.on_user_response("doorway")
        assert mgr.is_active() is False

    def test_cancel_without_context_is_noop(self):
        mgr, _tts, _send, fsm = _make_manager()
        mgr.cancel()
        assert fsm.events == []

    def test_tts_failure_does_not_break_flow(self):
        class BrokenTTS:
            def synthesize(self, text: str) -> bytes:
                raise RuntimeError("gTTS down")

        send = FakeSendTTS()
        fsm = FakeFSM()
        mgr = NavigationTaskManager(tts_service=BrokenTTS(), send_tts=send, fsm=fsm)
        # Even with TTS failing, context still gets created and state advances.
        mgr.on_navigation_command("kitchen")
        assert mgr.is_active() is True
        mgr.on_user_response("doorway")
        assert mgr.context.awaiting_waypoint is False
        assert mgr.is_frame_processing_enabled() is True
        # No audio was pushed (synthesize failed), but no crash either.
        assert send.sent == []
