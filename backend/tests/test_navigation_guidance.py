"""Detector-level tests for Part 4 — guidance generation & throttling."""

from __future__ import annotations

import logging
from typing import Optional

import pytest

from navigation.context import NavigationContext, Waypoint
from navigation.detector import LandmarkDetector


# ----------------------------------------------------------------------
# Helpers (consistent with Parts 2 + 3 fakes)
# ----------------------------------------------------------------------

def make_detection(
    cls: str,
    conf: float = 0.85,
    region: Optional[str] = "center",
    distance: Optional[str] = "near",
    bbox=(100, 100, 400, 400),
) -> dict:
    d = {"class_name": cls, "confidence": conf, "bbox": list(bbox)}
    if region is not None:
        d["region"] = region
    if distance is not None:
        d["distance_category"] = distance
    return d


class Callbacks:
    def __init__(self) -> None:
        self.reached: list = []
        self.disambig: list = []
        self.timeout_fired: int = 0
        self.obstacles: list = []
        self.guidance: list = []

    def on_reached(self, wp): self.reached.append(wp)
    def on_disambig(self, choices): self.disambig.append(list(choices))
    def on_timeout(self): self.timeout_fired += 1
    def on_obstacle(self, info): self.obstacles.append(dict(info))
    def on_guidance(self, info): self.guidance.append(dict(info))


def make_detector(target_classes: list[str]):
    ctx = NavigationContext(destination="kitchen")
    ctx.frame_processing_enabled = True
    ctx.add_waypoint("looking", "x", target_classes)
    cb = Callbacks()
    det = LandmarkDetector(
        context=ctx,
        on_reached=cb.on_reached,
        on_disambiguation_needed=cb.on_disambig,
        on_detection_timeout=cb.on_timeout,
        on_obstacle=cb.on_obstacle,
        on_guidance=cb.on_guidance,
        logger=logging.getLogger("test"),
    )
    return det, ctx, cb


# ======================================================================
# A3 = A: temporal consistency required before first cue
# ======================================================================

class TestTemporalRequirement:
    def test_single_frame_no_guidance(self):
        # 1-of-3 does not satisfy 2-of-3 consistency.
        det, _ctx, cb = make_detector(["chair"])
        det.on_frame_detections([make_detection("chair", region="right", distance="far")])
        assert cb.guidance == []

    def test_two_of_three_fires_first_cue(self):
        det, _ctx, cb = make_detector(["chair"])
        det.on_frame_detections([make_detection("chair", region="right", distance="far")])
        det.on_frame_detections([make_detection("chair", region="right", distance="far")])
        assert len(cb.guidance) == 1
        assert cb.guidance[0]["change_type"] == "first_cue"
        assert cb.guidance[0]["class_name"] == "chair"
        assert cb.guidance[0]["region"] == "right"
        assert cb.guidance[0]["distance_category"] == "far"

    def test_three_misses_no_guidance(self):
        det, _ctx, cb = make_detector(["chair"])
        for _ in range(3):
            det.on_frame_detections([])
        assert cb.guidance == []


# ======================================================================
# A1 = B: change-triggered only
# ======================================================================

class TestChangeTrigger:
    def test_same_state_does_not_re_speak(self):
        # Once we've spoken, repeating the same region+distance is silent.
        det, _ctx, cb = make_detector(["chair"])
        # Build temporal confirmation.
        det.on_frame_detections([make_detection("chair", region="right", distance="far")])
        det.on_frame_detections([make_detection("chair", region="right", distance="far")])
        assert len(cb.guidance) == 1  # first cue
        # More frames with identical state -> no new cues.
        for _ in range(5):
            det.on_frame_detections([make_detection("chair", region="right", distance="far")])
        assert len(cb.guidance) == 1

    def test_region_change_speaks(self):
        det, _ctx, cb = make_detector(["chair"])
        det.on_frame_detections([make_detection("chair", region="right", distance="far")])
        det.on_frame_detections([make_detection("chair", region="right", distance="far")])
        # Now region changes.
        det.on_frame_detections([make_detection("chair", region="center", distance="far")])
        assert len(cb.guidance) == 2
        assert cb.guidance[1]["change_type"] == "region_changed"

    def test_distance_change_speaks(self):
        det, _ctx, cb = make_detector(["chair"])
        det.on_frame_detections([make_detection("chair", region="right", distance="far")])
        det.on_frame_detections([make_detection("chair", region="right", distance="far")])
        # Distance closes.
        det.on_frame_detections([make_detection("chair", region="right", distance="medium")])
        assert len(cb.guidance) == 2
        assert cb.guidance[1]["change_type"] == "distance_changed"

    def test_both_change_marked_correctly(self):
        det, _ctx, cb = make_detector(["chair"])
        det.on_frame_detections([make_detection("chair", region="right", distance="far")])
        det.on_frame_detections([make_detection("chair", region="right", distance="far")])
        # Both change at once.
        det.on_frame_detections([make_detection("chair", region="left", distance="medium")])
        assert cb.guidance[-1]["change_type"] == "both_changed"


# ======================================================================
# D4 = A: guidance fires when near but not centered
# ======================================================================

class TestNearOffCenter:
    def test_near_off_center_gets_guidance(self):
        # Chair temporally confirmed, near, but on the right -> reached
        # criterion misses (center required) but guidance should fire.
        det, _ctx, cb = make_detector(["chair"])
        det.on_frame_detections([make_detection("chair", region="right", distance="near")])
        det.on_frame_detections([make_detection("chair", region="right", distance="near")])
        # Guidance for first cue should be there.
        assert len(cb.guidance) >= 1
        assert cb.guidance[-1]["distance_category"] == "near"
        assert cb.guidance[-1]["region"] == "right"
        # Reached should NOT have fired (center required).
        assert cb.reached == []


# ======================================================================
# C5 = A: state cleared on advance, next waypoint gets fresh first cue
# ======================================================================

class TestAdvanceResetsGuidance:
    def test_advance_resets_last_spoken_state(self):
        ctx = NavigationContext(destination="kitchen")
        ctx.frame_processing_enabled = True
        ctx.add_waypoint("a", "a", ["chair"])
        ctx.add_waypoint("b", "b", ["dining table"])
        cb = Callbacks()
        det = LandmarkDetector(
            context=ctx,
            on_reached=cb.on_reached,
            on_disambiguation_needed=cb.on_disambig,
            on_detection_timeout=cb.on_timeout,
            on_obstacle=cb.on_obstacle,
            on_guidance=cb.on_guidance,
            logger=logging.getLogger("test"),
        )
        # Hit chair on the right -> first cue.
        det.on_frame_detections([make_detection("chair", region="right", distance="far")])
        det.on_frame_detections([make_detection("chair", region="right", distance="far")])
        assert len(cb.guidance) == 1
        # Advance to second waypoint.
        ctx.advance()
        det.reset_for_new_waypoint()
        # Even though previous landmark was on the right far, the next
        # waypoint must start with its own first_cue when detected.
        det.on_frame_detections([make_detection("dining table", region="right", distance="far")])
        det.on_frame_detections([make_detection("dining table", region="right", distance="far")])
        # New guidance entry; change_type should be "first_cue", not "no change".
        assert len(cb.guidance) == 2
        assert cb.guidance[-1]["change_type"] == "first_cue"
        assert cb.guidance[-1]["class_name"] == "dining table"


# ======================================================================
# B6 + Interpretation 1: class-name change forces a fresh full cue
# ======================================================================

class TestClassNameChange:
    def test_class_change_forces_fresh_cue_within_same_position(self):
        # Manually simulate the scenario: spoken class was "chair", then
        # detection comes in for "couch" at the SAME region+distance.
        # Per B6 Interpretation 1, this should trigger a fresh first_cue.
        det, ctx, cb = make_detector(["chair", "couch"])
        # Set up last-spoken state as if we'd already spoken about "chair"
        # on the right far.
        ctx.last_spoken_region = "right"
        ctx.last_spoken_distance = "far"
        ctx.last_spoken_class = "chair"
        ctx.landmark_visible_last_frame = True
        # Pre-fill history so temporal consistency is satisfied for couch.
        ctx.detection_history.append({"couch"})
        ctx.detection_history.append({"couch"})
        # Couch appears at same right+far. Same position, DIFFERENT class.
        det.on_frame_detections([make_detection("couch", region="right", distance="far")])
        # We should have a fresh first_cue (because class changed).
        assert len(cb.guidance) == 1
        assert cb.guidance[0]["change_type"] == "first_cue"
        assert cb.guidance[0]["class_name"] == "couch"
        assert cb.guidance[0]["is_class_change"] is True


# ======================================================================
# F2 = B: lost-sight cue when landmark disappears after being visible
# ======================================================================

class TestLostSight:
    def test_lost_sight_fires_after_visible_then_gone(self):
        det, _ctx, cb = make_detector(["chair"])
        # Two frames visible -> first cue.
        det.on_frame_detections([make_detection("chair", region="right", distance="far")])
        det.on_frame_detections([make_detection("chair", region="right", distance="far")])
        guidance_after_visible = len(cb.guidance)
        # Now no chair in frame.
        det.on_frame_detections([])
        assert len(cb.guidance) == guidance_after_visible + 1
        assert cb.guidance[-1]["change_type"] == "lost_sight"

    def test_lost_sight_does_not_fire_without_prior_visibility(self):
        # If we never had a confirmed-visible state, "going away" of
        # nothing shouldn't fire.
        det, _ctx, cb = make_detector(["chair"])
        det.on_frame_detections([])
        det.on_frame_detections([])
        det.on_frame_detections([])
        assert cb.guidance == []

    def test_returning_after_lost_sight_gives_fresh_first_cue(self):
        det, _ctx, cb = make_detector(["chair"])
        det.on_frame_detections([make_detection("chair", region="right", distance="far")])
        det.on_frame_detections([make_detection("chair", region="right", distance="far")])
        # Lost sight.
        det.on_frame_detections([])
        # Chair returns at same region+distance. Because last_spoken_*
        # were cleared by lost_sight, this is a fresh first_cue.
        det.on_frame_detections([make_detection("chair", region="right", distance="far")])
        det.on_frame_detections([make_detection("chair", region="right", distance="far")])
        # Last entry is a new first_cue.
        assert cb.guidance[-1]["change_type"] == "first_cue"


# ======================================================================
# A4 = A: obstacle preempts guidance (in same frame)
# ======================================================================

class TestObstaclePreemption:
    def test_obstacle_suppresses_guidance_for_same_frame(self):
        det, _ctx, cb = make_detector(["chair"])
        # Build chair temporal consistency.
        det.on_frame_detections([make_detection("chair", region="right", distance="far")])
        det.on_frame_detections([make_detection("chair", region="right", distance="far")])
        # First guidance fired.
        assert len(cb.guidance) == 1
        # Now: chair moves to center (would trigger region_changed cue)
        # AND a person obstacle enters.
        det.on_frame_detections([
            make_detection("chair", region="center", distance="far"),
            make_detection("person", region="center", distance="near"),
        ])
        # Obstacle warning fired; guidance suppressed for this frame.
        assert len(cb.obstacles) == 1
        # Guidance count should NOT have grown.
        assert len(cb.guidance) == 1


# ======================================================================
# A5 = A: guidance paused during disambiguation
# ======================================================================

class TestDisambiguationPause:
    def test_guidance_paused_during_disambiguation(self):
        det, ctx, cb = make_detector(["chair", "couch"])
        # Trigger disambiguation by showing both classes for 2 frames.
        det.on_frame_detections([
            make_detection("chair", region="left", distance="medium"),
            make_detection("couch", region="right", distance="medium"),
        ])
        det.on_frame_detections([
            make_detection("chair", region="center", distance="near"),
            make_detection("couch", region="right", distance="medium"),
        ])
        assert ctx.awaiting_disambiguation is True
        # Disambig was triggered, but before this, guidance might or might
        # not have fired during the first uncertain frames. Reset and check
        # forward.
        cb.guidance.clear()
        # Now MORE frames arrive — guidance should NOT fire while pending.
        det.on_frame_detections([
            make_detection("chair", region="left", distance="far"),
        ])
        det.on_frame_detections([
            make_detection("chair", region="left", distance="far"),
        ])
        assert cb.guidance == []


# ======================================================================
# E5 = A: missing region/distance -> no guidance
# ======================================================================

class TestMissingFields:
    def test_missing_region_no_guidance(self):
        det, _ctx, cb = make_detector(["chair"])
        det.on_frame_detections([make_detection("chair", region=None, distance="far")])
        det.on_frame_detections([make_detection("chair", region=None, distance="far")])
        assert cb.guidance == []

    def test_missing_distance_no_guidance(self):
        det, _ctx, cb = make_detector(["chair"])
        det.on_frame_detections([make_detection("chair", region="right", distance=None)])
        det.on_frame_detections([make_detection("chair", region="right", distance=None)])
        assert cb.guidance == []
