"""Detector-level tests for Part 3 obstacle handling."""

from __future__ import annotations

import logging
import time
from typing import Optional

import pytest

from navigation.context import NavigationContext, Waypoint, WaypointStatus
from navigation.detector import (
    LandmarkDetector,
    OBSTACLE_WARNING_COOLDOWN_SECONDS,
    MAX_OBSTACLE_ESCALATION_LEVEL,
)


# ----------------------------------------------------------------------
# Helpers (same FakeDetector pattern as Part 2)
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
        self.reached: list[Waypoint] = []
        self.disambig: list[list[str]] = []
        self.timeout_fired: int = 0
        self.obstacles: list[dict] = []

    def on_reached(self, wp): self.reached.append(wp)
    def on_disambig(self, choices): self.disambig.append(list(choices))
    def on_timeout(self): self.timeout_fired += 1
    def on_obstacle(self, info): self.obstacles.append(dict(info))


def make_detector(target_classes: list[str]):
    ctx = NavigationContext(destination="kitchen")
    ctx.frame_processing_enabled = True
    ctx.add_waypoint("looking for it", "it", target_classes)
    cb = Callbacks()
    det = LandmarkDetector(
        context=ctx,
        on_reached=cb.on_reached,
        on_disambiguation_needed=cb.on_disambig,
        on_detection_timeout=cb.on_timeout,
        on_obstacle=cb.on_obstacle,
        logger=logging.getLogger("test"),
    )
    return det, ctx, cb


# ======================================================================
# Basic obstacle warning firing
# ======================================================================

class TestObstacleWarning:
    def test_person_in_path_fires_warning(self):
        det, _ctx, cb = make_detector(["chair"])
        det.on_frame_detections([make_detection("person", region="center", distance="near")])
        assert len(cb.obstacles) == 1
        assert cb.obstacles[0]["class_name"] == "person"
        assert cb.obstacles[0]["is_person"] is True

    def test_furniture_in_path_fires_warning(self):
        det, _ctx, cb = make_detector(["doorway"])
        det.on_frame_detections([make_detection("couch", region="center", distance="near")])
        assert len(cb.obstacles) == 1
        assert cb.obstacles[0]["class_name"] == "couch"
        assert cb.obstacles[0]["is_person"] is False

    def test_off_center_obstacle_does_not_warn(self):
        # B1+B3: only center-region obstacles count.
        det, _ctx, cb = make_detector(["chair"])
        det.on_frame_detections([make_detection("person", region="right", distance="near")])
        assert cb.obstacles == []

    def test_far_obstacle_does_not_warn(self):
        # B2: only "near" obstacles warn.
        det, _ctx, cb = make_detector(["chair"])
        det.on_frame_detections([make_detection("person", region="center", distance="far")])
        assert cb.obstacles == []

    def test_low_confidence_obstacle_dropped(self):
        det, _ctx, cb = make_detector(["chair"])
        det.on_frame_detections([make_detection("person", conf=0.3, region="center", distance="near")])
        assert cb.obstacles == []


# ======================================================================
# A2: landmark itself is NEVER an obstacle
# ======================================================================

class TestLandmarkExempt:
    def test_active_landmark_class_does_not_warn(self):
        # We're looking for the "chair" — a chair in our path is the goal,
        # not an obstacle.
        det, _ctx, cb = make_detector(["chair"])
        det.on_frame_detections([make_detection("chair", region="center", distance="near")])
        assert cb.obstacles == []

    def test_other_target_class_in_multi_class_alias_also_exempt(self):
        # "seat" maps to chair+couch+bench. None of them should warn.
        det, _ctx, cb = make_detector(["chair", "couch", "bench"])
        det.on_frame_detections([
            make_detection("chair", region="center", distance="near"),
            make_detection("couch", region="center", distance="near"),
        ])
        assert cb.obstacles == []

    def test_non_landmark_obstacle_still_warns_when_landmark_visible(self):
        # User heading to chair; a person is also in the way.
        det, _ctx, cb = make_detector(["chair"])
        det.on_frame_detections([
            make_detection("chair", region="left", distance="medium"),
            make_detection("person", region="center", distance="near"),
        ])
        assert len(cb.obstacles) == 1
        assert cb.obstacles[0]["class_name"] == "person"


# ======================================================================
# A5 size gate for small objects
# ======================================================================

class TestSizeGate:
    def test_small_object_with_small_bbox_no_warning(self):
        # A cup with a tiny bbox is too far to be a real obstacle.
        det, _ctx, cb = make_detector(["chair"])
        det.on_frame_detections([
            make_detection("cup", region="center", distance="near", bbox=[0, 0, 50, 50])
        ])
        assert cb.obstacles == []

    def test_small_object_with_large_bbox_does_warn(self):
        # A cup right under your feet — toe-strike risk.
        det, _ctx, cb = make_detector(["chair"])
        det.on_frame_detections([
            make_detection("cup", region="center", distance="near", bbox=[0, 0, 250, 250])
        ])
        assert len(cb.obstacles) == 1
        assert cb.obstacles[0]["class_name"] == "cup"

    def test_size_gate_not_applied_to_large_obstacles(self):
        # Even a small bbox couch warns (the detector trusts the class).
        det, _ctx, cb = make_detector(["chair"])
        det.on_frame_detections([
            make_detection("couch", region="center", distance="near", bbox=[0, 0, 30, 30])
        ])
        assert len(cb.obstacles) == 1


# ======================================================================
# E2/E3 cooldown throttling
# ======================================================================

class TestCooldown:
    def test_same_class_within_cooldown_suppressed(self):
        det, _ctx, cb = make_detector(["chair"])
        det.on_frame_detections([make_detection("person", region="center", distance="near")])
        det.on_frame_detections([make_detection("person", region="center", distance="near")])
        det.on_frame_detections([make_detection("person", region="center", distance="near")])
        # Three frames in <3s -> only first one warns.
        assert len(cb.obstacles) == 1

    def test_same_class_after_cooldown_re_warns(self):
        det, ctx, cb = make_detector(["chair"])
        det.on_frame_detections([make_detection("person", region="center", distance="near")])
        # Simulate cooldown passing by rewinding the recorded timestamp.
        ctx.last_obstacle_warning_at["person"] = (
            time.time() - OBSTACLE_WARNING_COOLDOWN_SECONDS - 1
        )
        det.on_frame_detections([make_detection("person", region="center", distance="near")])
        assert len(cb.obstacles) == 2


# ======================================================================
# E5 different class bypasses cooldown
# ======================================================================

class TestDifferentClassBypass:
    def test_different_class_warns_immediately(self):
        det, _ctx, cb = make_detector(["doorway"])
        det.on_frame_detections([make_detection("chair", region="center", distance="near")])
        det.on_frame_detections([make_detection("person", region="center", distance="near")])
        # Person and chair are different classes; person fires immediately
        # despite chair being under cooldown.
        cls_names = [o["class_name"] for o in cb.obstacles]
        assert cls_names == ["chair", "person"]


# ======================================================================
# E6 priority ordering — person > furniture
# ======================================================================

class TestPriority:
    def test_person_chosen_over_furniture_when_both_present(self):
        det, _ctx, cb = make_detector(["doorway"])
        det.on_frame_detections([
            make_detection("chair", region="center", distance="near",
                           bbox=[0, 0, 500, 500]),  # very big bbox
            make_detection("person", region="center", distance="near",
                           bbox=[200, 200, 300, 300]),  # smaller bbox
        ])
        # Even though chair has bigger bbox, person wins on priority.
        assert len(cb.obstacles) == 1
        assert cb.obstacles[0]["class_name"] == "person"

    def test_nearest_furniture_when_multiple_furniture(self):
        det, _ctx, cb = make_detector(["doorway"])
        det.on_frame_detections([
            make_detection("chair", region="center", distance="near",
                           bbox=[0, 0, 100, 100]),
            make_detection("couch", region="center", distance="near",
                           bbox=[0, 0, 400, 400]),  # larger -> nearer
        ])
        assert cb.obstacles[0]["class_name"] == "couch"


# ======================================================================
# Issue 3 — escalation ladder
# ======================================================================

class TestEscalation:
    def test_first_warning_is_level_0(self):
        det, _ctx, cb = make_detector(["doorway"])
        det.on_frame_detections([make_detection("chair", region="center", distance="near")])
        assert cb.obstacles[0]["escalation_level"] == 0

    def test_second_warning_after_cooldown_is_level_1(self):
        det, ctx, cb = make_detector(["doorway"])
        det.on_frame_detections([make_detection("chair", region="center", distance="near")])
        # Force cooldown elapsed without the obstacle "clearing" (it stayed
        # in path), so escalation level increments.
        ctx.last_obstacle_warning_at["chair"] = (
            time.time() - OBSTACLE_WARNING_COOLDOWN_SECONDS - 1
        )
        det.on_frame_detections([make_detection("chair", region="center", distance="near")])
        assert cb.obstacles[1]["escalation_level"] == 1

    def test_escalation_caps_at_max(self):
        det, ctx, cb = make_detector(["doorway"])
        for _ in range(10):
            det.on_frame_detections([make_detection("chair", region="center", distance="near")])
            ctx.last_obstacle_warning_at["chair"] = (
                time.time() - OBSTACLE_WARNING_COOLDOWN_SECONDS - 1
            )
        # Every recorded level should be at most MAX.
        levels = [o["escalation_level"] for o in cb.obstacles]
        assert max(levels) == MAX_OBSTACLE_ESCALATION_LEVEL

    def test_escalation_resets_when_obstacle_clears(self):
        det, ctx, cb = make_detector(["doorway"])
        det.on_frame_detections([make_detection("chair", region="center", distance="near")])
        ctx.last_obstacle_warning_at["chair"] = (
            time.time() - OBSTACLE_WARNING_COOLDOWN_SECONDS - 1
        )
        det.on_frame_detections([make_detection("chair", region="center", distance="near")])
        # Level was 1. Now a frame with no chair -> chair "cleared".
        det.on_frame_detections([])
        # And it reappears later (cooldown irrelevant since cleared).
        det.on_frame_detections([make_detection("chair", region="center", distance="near")])
        # Last entry should be back at level 0.
        assert cb.obstacles[-1]["escalation_level"] == 0


# ======================================================================
# D1 — preemption: obstacle suppresses reached announcement
# ======================================================================

class TestPreemption:
    def test_obstacle_suppresses_reached_announcement(self):
        # Set up: chair is the target. Pre-fill temporal buffer so the
        # NEXT centered+near chair would normally fire reached.
        det, ctx, cb = make_detector(["chair"])
        # First two frames: chair visible, satisfies temporal consistency.
        # Person also in path -> obstacle fires, reached does NOT.
        det.on_frame_detections([
            make_detection("chair", region="center", distance="near"),
            make_detection("person", region="center", distance="near"),
        ])
        det.on_frame_detections([
            make_detection("chair", region="center", distance="near"),
            make_detection("person", region="center", distance="near"),
        ])
        # Obstacle warning fired at least once.
        assert len(cb.obstacles) >= 1
        # But reached was suppressed in the same frames.
        assert cb.reached == []

    def test_reached_fires_once_obstacle_clears(self):
        det, ctx, cb = make_detector(["chair"])
        # Frame 1: obstacle + chair -> reached suppressed, chair buffered.
        det.on_frame_detections([
            make_detection("chair", region="center", distance="near"),
            make_detection("person", region="center", distance="near"),
        ])
        # Frame 2: same -> reached suppressed again, but temporal buffer
        # has both frames now.
        det.on_frame_detections([
            make_detection("chair", region="center", distance="near"),
            make_detection("person", region="center", distance="near"),
        ])
        assert cb.reached == []
        # Frame 3: person cleared. Chair still center+near, buffer satisfies
        # consistency -> reached should fire.
        det.on_frame_detections([
            make_detection("chair", region="center", distance="near"),
        ])
        assert len(cb.reached) == 1


# ======================================================================
# G4 — missing fields: defensive behavior
# ======================================================================

class TestMissingFields:
    def test_missing_region_no_warning(self):
        # G4 = A: without region, we can't tell if it's in path -> no warn.
        det, _ctx, cb = make_detector(["chair"])
        det.on_frame_detections([
            make_detection("person", region=None, distance="near"),
        ])
        assert cb.obstacles == []

    def test_missing_distance_no_warning(self):
        det, _ctx, cb = make_detector(["chair"])
        det.on_frame_detections([
            make_detection("person", region="center", distance=None),
        ])
        assert cb.obstacles == []
