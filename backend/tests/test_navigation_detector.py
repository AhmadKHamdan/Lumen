"""Unit tests for navigation.detector.LandmarkDetector.

G8 = A: tests use hand-built detection dicts (the "FakeDetector" approach).
No YOLO, no Ahmad's code, no GPU — just deterministic dict-shaped input.
"""

from __future__ import annotations

import logging
from typing import Optional

import pytest

from navigation.context import (
    NavigationContext,
    Waypoint,
    WaypointStatus,
    DETECTION_MIN_HITS,
    DETECTION_WINDOW,
)
from navigation.detector import (
    LandmarkDetector,
    CONFIDENCE_THRESHOLD,
    LANDMARK_DETECTION_TIMEOUT_SECONDS,
    DISAMBIGUATION_TIMEOUT_SECONDS,
)


# ----------------------------------------------------------------------
# Test helpers
# ----------------------------------------------------------------------

def make_detection(
    cls: str,
    conf: float = 0.85,
    region: Optional[str] = "center",
    distance: Optional[str] = "near",
    bbox=(100, 100, 400, 400),
) -> dict:
    """Build a synthetic detection dict matching the C1=A contract."""
    d = {
        "class_name": cls,
        "confidence": conf,
        "bbox": list(bbox),
    }
    if region is not None:
        d["region"] = region
    if distance is not None:
        d["distance_category"] = distance
    return d


class Callbacks:
    """Record every callback fire for assertions."""
    def __init__(self) -> None:
        self.reached: list[Waypoint] = []
        self.disambig: list[list[str]] = []
        self.timeout_fired: int = 0

    def on_reached(self, wp: Waypoint) -> None:
        self.reached.append(wp)

    def on_disambig(self, choices: list[str]) -> None:
        self.disambig.append(list(choices))

    def on_timeout(self) -> None:
        self.timeout_fired += 1


def make_detector_with_waypoint(
    target_classes: list[str],
    *,
    awaiting_disambiguation: bool = False,
) -> tuple[LandmarkDetector, NavigationContext, Callbacks]:
    """Build a NavigationContext + LandmarkDetector with one active waypoint."""
    ctx = NavigationContext(destination="kitchen")
    ctx.frame_processing_enabled = True
    ctx.add_waypoint("looking for it", "it", target_classes)
    if awaiting_disambiguation:
        ctx.awaiting_disambiguation = True
        ctx.disambiguation_choices = list(target_classes)

    cb = Callbacks()
    det = LandmarkDetector(
        context=ctx,
        on_reached=cb.on_reached,
        on_disambiguation_needed=cb.on_disambig,
        on_detection_timeout=cb.on_timeout,
        logger=logging.getLogger("test"),
    )
    return det, ctx, cb


# ----------------------------------------------------------------------
# Filtering (C4 + C5)
# ----------------------------------------------------------------------

class TestFiltering:
    def test_drops_below_confidence_threshold(self):
        det, ctx, cb = make_detector_with_waypoint(["chair"])
        det.on_frame_detections([make_detection("chair", conf=CONFIDENCE_THRESHOLD - 0.05)])
        # Confidence too low -> nothing added to history.
        assert ctx.detection_history[-1] == set()
        assert cb.reached == []

    def test_drops_unrelated_classes(self):
        det, ctx, cb = make_detector_with_waypoint(["chair"])
        det.on_frame_detections([make_detection("dog")])
        assert ctx.detection_history[-1] == set()

    def test_keeps_relevant_detections(self):
        det, ctx, _ = make_detector_with_waypoint(["chair"])
        det.on_frame_detections([make_detection("chair", conf=0.8)])
        assert ctx.detection_history[-1] == {"chair"}

    def test_no_op_when_gate_closed(self):
        det, ctx, cb = make_detector_with_waypoint(["chair"])
        ctx.frame_processing_enabled = False
        det.on_frame_detections([make_detection("chair", region="center", distance="near")])
        det.on_frame_detections([make_detection("chair", region="center", distance="near")])
        # No history pushed.
        assert len(ctx.detection_history) == 0
        assert cb.reached == []


# ----------------------------------------------------------------------
# Temporal consistency (D1 + D2 + D3)
# ----------------------------------------------------------------------

class TestTemporalConsistency:
    def test_single_centered_near_detection_does_not_fire_reached(self):
        # D3 = B: majority (2-of-3) required. One frame is not enough.
        det, _ctx, cb = make_detector_with_waypoint(["chair"])
        det.on_frame_detections([make_detection("chair", region="center", distance="near")])
        assert cb.reached == []

    def test_two_of_three_fires_reached(self):
        det, _ctx, cb = make_detector_with_waypoint(["chair"])
        # Frame 1: hit. Frame 2: miss. Frame 3: hit. -> 2-of-3 confirmation.
        det.on_frame_detections([make_detection("chair", region="center", distance="near")])
        det.on_frame_detections([])
        det.on_frame_detections([make_detection("chair", region="center", distance="near")])
        assert len(cb.reached) == 1
        assert cb.reached[0].status == WaypointStatus.REACHED

    def test_three_misses_do_not_fire_reached(self):
        det, _ctx, cb = make_detector_with_waypoint(["chair"])
        for _ in range(3):
            det.on_frame_detections([])
        assert cb.reached == []

    def test_window_size_is_three(self):
        # Verify the bounded buffer.
        det, ctx, _ = make_detector_with_waypoint(["chair"])
        for _ in range(10):
            det.on_frame_detections([])
        assert len(ctx.detection_history) == DETECTION_WINDOW

    def test_far_detection_does_not_satisfy_reached(self):
        # E1: must be "near".
        det, _ctx, cb = make_detector_with_waypoint(["chair"])
        for _ in range(3):
            det.on_frame_detections([make_detection("chair", region="center", distance="far")])
        # Temporal consistency is satisfied, but reached fails on distance.
        assert cb.reached == []

    def test_off_center_detection_does_not_satisfy_reached(self):
        # E1: must be "center".
        det, _ctx, cb = make_detector_with_waypoint(["chair"])
        for _ in range(3):
            det.on_frame_detections([make_detection("chair", region="right", distance="near")])
        assert cb.reached == []

    def test_centered_near_after_temporal_window(self):
        # Buffer needs 2-of-3 hits anywhere; the *current* frame must be
        # center + near for reached to fire.
        det, _ctx, cb = make_detector_with_waypoint(["chair"])
        det.on_frame_detections([make_detection("chair", region="right", distance="far")])
        det.on_frame_detections([make_detection("chair", region="center", distance="near")])
        # Two hits in two frames = passes consistency. Current frame is
        # center+near, so reached fires.
        assert len(cb.reached) == 1


# ----------------------------------------------------------------------
# Reached re-arming (E6)
# ----------------------------------------------------------------------

class TestReachedReArming:
    def test_reached_fires_only_once_per_waypoint(self):
        det, _ctx, cb = make_detector_with_waypoint(["chair"])
        det.on_frame_detections([make_detection("chair", region="center", distance="near")])
        det.on_frame_detections([make_detection("chair", region="center", distance="near")])
        assert len(cb.reached) == 1
        # More frames keep arriving — should not re-fire.
        det.on_frame_detections([make_detection("chair", region="center", distance="near")])
        det.on_frame_detections([make_detection("chair", region="center", distance="near")])
        assert len(cb.reached) == 1

    def test_reset_for_new_waypoint_rearms(self):
        det, _ctx, cb = make_detector_with_waypoint(["chair"])
        det.on_frame_detections([make_detection("chair", region="center", distance="near")])
        det.on_frame_detections([make_detection("chair", region="center", distance="near")])
        assert len(cb.reached) == 1
        det.reset_for_new_waypoint()
        det.on_frame_detections([make_detection("chair", region="center", distance="near")])
        det.on_frame_detections([make_detection("chair", region="center", distance="near")])
        assert len(cb.reached) == 2


# ----------------------------------------------------------------------
# Disambiguation (C7)
# ----------------------------------------------------------------------

class TestDisambiguation:
    def test_triggers_when_two_classes_temporally_confirmed(self):
        det, ctx, cb = make_detector_with_waypoint(["chair", "couch"])
        # Both visible in 2 of last 3 frames -> trigger.
        det.on_frame_detections([
            make_detection("chair", region="left", distance="medium"),
            make_detection("couch", region="right", distance="medium"),
        ])
        det.on_frame_detections([
            make_detection("chair", region="center", distance="near"),
            make_detection("couch", region="right", distance="medium"),
        ])
        assert len(cb.disambig) == 1
        assert set(cb.disambig[0]) == {"chair", "couch"}
        assert ctx.awaiting_disambiguation is True

    def test_pauses_reached_during_disambiguation(self):
        # Even if everything looks ready, no reached event fires until
        # the manager calls disambiguate(class).
        det, ctx, cb = make_detector_with_waypoint(
            ["chair", "couch"], awaiting_disambiguation=True
        )
        for _ in range(3):
            det.on_frame_detections([
                make_detection("chair", region="center", distance="near"),
            ])
        assert cb.reached == []

    def test_disambiguate_locks_to_chosen_class(self):
        det, ctx, cb = make_detector_with_waypoint(["chair", "couch"])
        det.disambiguate("chair")
        assert ctx.current_waypoint().locked_class == "chair"
        assert ctx.awaiting_disambiguation is False
        # Now reached can fire for chair alone.
        det.on_frame_detections([make_detection("chair", region="center", distance="near")])
        det.on_frame_detections([make_detection("chair", region="center", distance="near")])
        assert len(cb.reached) == 1

    def test_disambiguate_fallback_picks_most_observed_class(self):
        det, ctx, _cb = make_detector_with_waypoint(["chair", "couch"])
        # Pre-populate history: chair seen in 3 frames, couch in 1.
        ctx.detection_history.append({"chair"})
        ctx.detection_history.append({"chair", "couch"})
        ctx.detection_history.append({"chair"})
        ctx.awaiting_disambiguation = True
        ctx.disambiguation_choices = ["chair", "couch"]
        det.disambiguate_fallback()
        assert ctx.current_waypoint().locked_class == "chair"


# ----------------------------------------------------------------------
# Timeouts (G1 detection-timeout + C7 sub 4 disambiguation-timeout)
# ----------------------------------------------------------------------

class TestTimeouts:
    def test_detection_timeout_fires_after_60s_with_no_hits(self):
        det, ctx, cb = make_detector_with_waypoint(["chair"])
        for _ in range(3):
            det.on_frame_detections([])  # nothing seen
        future = ctx.current_waypoint_started_at + LANDMARK_DETECTION_TIMEOUT_SECONDS + 1
        det.check_timeout(now=future)
        assert cb.timeout_fired == 1

    def test_detection_timeout_resets_when_anything_detected(self):
        det, ctx, cb = make_detector_with_waypoint(["chair"])
        # See it once, doesn't trigger reached but updates last_detection_at.
        det.on_frame_detections([make_detection("chair", region="left", distance="medium")])
        future = ctx.current_waypoint_started_at + LANDMARK_DETECTION_TIMEOUT_SECONDS + 1
        det.check_timeout(now=future)
        # Because last_detection_at is set, the timeout shouldn't fire.
        assert cb.timeout_fired == 0

    def test_detection_timeout_fires_at_most_once(self):
        det, ctx, cb = make_detector_with_waypoint(["chair"])
        for _ in range(3):
            det.on_frame_detections([])
        future = ctx.current_waypoint_started_at + LANDMARK_DETECTION_TIMEOUT_SECONDS + 5
        det.check_timeout(now=future)
        det.check_timeout(now=future + 10)
        det.check_timeout(now=future + 20)
        assert cb.timeout_fired == 1

    def test_disambiguation_timeout_calls_fallback(self):
        det, ctx, _cb = make_detector_with_waypoint(["chair", "couch"])
        ctx.detection_history.append({"chair"})
        ctx.detection_history.append({"chair"})
        ctx.detection_history.append({"chair", "couch"})
        ctx.awaiting_disambiguation = True
        ctx.disambiguation_choices = ["chair", "couch"]
        import time as _time
        ctx.disambiguation_started_at = _time.time()
        future = ctx.disambiguation_started_at + DISAMBIGUATION_TIMEOUT_SECONDS + 1
        det.check_timeout(now=future)
        assert ctx.current_waypoint().locked_class == "chair"


# ----------------------------------------------------------------------
# Missing-field tolerance (G5 = B)
# ----------------------------------------------------------------------

class TestMissingFields:
    def test_missing_region_does_not_crash(self):
        det, _ctx, cb = make_detector_with_waypoint(["chair"])
        # No region field — should still buffer the detection (warning logged
        # once) but reached check should fail because region != "center".
        for _ in range(3):
            det.on_frame_detections([make_detection("chair", region=None, distance="near")])
        assert cb.reached == []

    def test_missing_distance_does_not_crash(self):
        det, _ctx, cb = make_detector_with_waypoint(["chair"])
        for _ in range(3):
            det.on_frame_detections([make_detection("chair", region="center", distance=None)])
        assert cb.reached == []

    def test_warns_only_once(self, caplog):
        det, _ctx, _cb = make_detector_with_waypoint(["chair"])
        with caplog.at_level(logging.WARNING):
            for _ in range(5):
                det.on_frame_detections([make_detection("chair", region=None, distance=None)])
        # Each missing field should log at most once.
        region_warnings = [
            r for r in caplog.records if "detection_missing_region" in r.getMessage()
        ]
        distance_warnings = [
            r for r in caplog.records if "detection_missing_distance" in r.getMessage()
        ]
        assert len(region_warnings) == 1
        assert len(distance_warnings) == 1


# ----------------------------------------------------------------------
# C6/C7 tie-breaks (highest confidence)
# ----------------------------------------------------------------------

class TestTieBreaks:
    def test_multiple_instances_picks_highest_confidence(self):
        # Two chairs in frame, both centered + near.
        det, _ctx, cb = make_detector_with_waypoint(["chair"])
        for _ in range(2):
            det.on_frame_detections([
                make_detection("chair", conf=0.7, region="center", distance="near"),
                make_detection("chair", conf=0.95, region="center", distance="near"),
            ])
        assert len(cb.reached) == 1
