"""LandmarkDetector — per-frame engine for the Navigation task (Part 2).

Consumes detection dicts produced by partner Ahmad's Sprint 3 spatial-
reasoning module. Applies temporal consistency, picks the right detection
when the active waypoint maps to several classes, and fires three kinds of
events through callbacks the manager hands in at construction time:

    on_reached(waypoint)
        Fires when the active waypoint's reached criterion (E1 = A) is met:
        temporal-consistency confirmation in region=center and distance=near.

    on_disambiguation_needed(choices: list[str])
        Fires when more than one alias class is present at the same time
        with temporal consistency on each (C7 = D). The detector pauses
        further reached-checks until the manager calls disambiguate(class).

    on_detection_timeout()
        Fires once when the active waypoint has been searched for longer
        than LANDMARK_DETECTION_TIMEOUT_SECONDS without any detection
        entering the buffer (G1).

The detector does not speak, does not touch the FSM, does not advance
waypoints — that's the manager's job. It is a pure observer of detection
dicts and an emitter of high-level events.

Detection input contract (C1 = A, documented for partner Ahmad):

    {
      "class_name": str,                        REQUIRED
      "confidence": float,                      REQUIRED, 0..1
      "bbox": [x1, y1, x2, y2],                 REQUIRED
      "region": "left" | "center" | "right",    OPTIONAL (G5 = B fallback)
      "distance_category": "near" | "medium" | "far",   OPTIONAL (G5 = B)
    }

Missing region or distance_category fields are tolerated: a warning is
logged once per session and the detection enters the buffer but fails the
reached criterion (which requires center + near).
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional

from .context import (
    DETECTION_MIN_HITS,
    DETECTION_WINDOW,
    NavigationContext,
    Waypoint,
)
from .obstacle_map import (
    is_obstacle,
    is_person,
    is_small_object,
    obstacle_priority,
    passes_size_gate,
)


# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

# C4 = B: minimum confidence to consider a detection at all.
CONFIDENCE_THRESHOLD: float = 0.6

# G1: detection-timeout for "I'm looking but nothing matches".
LANDMARK_DETECTION_TIMEOUT_SECONDS: float = 60.0

# C7 sub-decision 4: disambiguation prompt timeout (then fall back to
# highest-confidence pick silently).
DISAMBIGUATION_TIMEOUT_SECONDS: float = 10.0

# Part 3 — obstacle warning throttling and escalation.
# E3: cooldown before re-warning about the same obstacle class.
OBSTACLE_WARNING_COOLDOWN_SECONDS: float = 3.0
# Issue 3 resolution: 3-level constrained escalation ladder.
MAX_OBSTACLE_ESCALATION_LEVEL: int = 2


# Callback types (just aliases — Python doesn't enforce these at runtime).
ReachedCallback = Callable[[Waypoint], None]
DisambiguationCallback = Callable[[list[str]], None]
DetectionTimeoutCallback = Callable[[], None]
# F1+F2: obstacle callback receives a dict {class_name, region,
# distance_category, confidence, escalation_level, is_person}.
ObstacleCallback = Callable[[dict], None]
# Part 4 D2 = A: guidance callback receives a structured info dict so the
# manager can choose phrasing without the detector knowing strings.
# {class_name, region, distance_category, change_type, is_class_change}
# change_type is one of: "first_cue", "region_changed", "distance_changed",
# "both_changed", "lost_sight".
GuidanceCallback = Callable[[dict], None]


class LandmarkDetector:
    """Per-frame detection consumer attached to a NavigationContext.

    Lifecycle:
      - Constructed by the manager when a Navigation task begins.
      - on_frame_detections() called once per frame by the frame handler.
      - Manager calls disambiguate(class_name) once the user resolves a
        chair-vs-couch prompt. Detector then locks the active waypoint
        to that class and resumes reached-checks.
      - check_timeout() polled from the session loop (alongside Part 1's
        waypoint-prompt timeout) — fires the detection-timeout callback.
      - Discarded when the manager cancels or the task completes.
    """

    def __init__(
        self,
        context: NavigationContext,
        on_reached: ReachedCallback,
        on_disambiguation_needed: DisambiguationCallback,
        on_detection_timeout: DetectionTimeoutCallback,
        on_obstacle: Optional[ObstacleCallback] = None,
        on_guidance: Optional[GuidanceCallback] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.context = context
        self._on_reached = on_reached
        self._on_disambiguation_needed = on_disambiguation_needed
        self._on_detection_timeout = on_detection_timeout
        # Part 3: obstacle callback is optional so tests that don't care
        # about obstacles can omit it. When None, obstacle warnings are
        # detected internally but no audible alert fires.
        self._on_obstacle = on_obstacle
        # Part 4: guidance callback similarly optional.
        self._on_guidance = on_guidance
        self._log = logger or logging.getLogger(__name__)
        # G5 = B: warn-once tracking so we don't spam logs at 5 FPS.
        self._warned_missing_region = False
        self._warned_missing_distance = False
        # E6 = A: once we've fired reached for the active waypoint, the
        # detector goes silent on it until the manager calls advance().
        self._reached_already_fired = False
        # G1: detection-timeout fires once per waypoint, not repeatedly.
        self._timeout_already_fired = False

    # -----------------------------------------------------------------
    # Main entry point — called by the frame handler at ~5 FPS
    # -----------------------------------------------------------------

    def on_frame_detections(self, detections: list[dict]) -> None:
        """Process one frame of detections.

        Order matters for Part 3 preemption (D1 = A):
          0. PART 3 — obstacle pass first. Any active obstacle warning
             SUPPRESSES the reached announcement for this frame, but the
             reached state machine keeps tracking (D2 = A).
          1. Filter landmark detections by confidence + class membership.
          2. Push to temporal-history buffer.
          3. Update last_detection_at.
          4. If awaiting disambiguation, stop.
          5. If multiple alias classes confirmed -> disambiguation prompt.
          6. Otherwise apply reached criterion (unless preempted).
        """
        if not self.context.frame_processing_enabled:
            return
        wp = self.context.current_waypoint()
        if wp is None:
            return

        # ---- PART 3: obstacle check (always first, runs even during
        # disambiguation per D3 = A) ----
        obstacle_fired = self._check_obstacles(detections, wp)

        if self._reached_already_fired:
            # Detector is idle on reached for this waypoint until advance().
            return

        # ---- Landmark filter ----
        relevant_detections = self._filter_relevant(detections, wp)
        classes_this_frame: set[str] = {
            d["class_name"] for d in relevant_detections
        }
        self.context.detection_history.append(classes_this_frame)

        if classes_this_frame:
            self.context.last_detection_at = time.time()

        if self.context.awaiting_disambiguation:
            return

        if wp.locked_class is None and len(wp.target_classes) > 1:
            confirmed = self._classes_passing_consistency(wp.target_classes)
            if len(confirmed) >= 2:
                self._trigger_disambiguation(confirmed)
                return

        # D1 = A: if an obstacle warning fired this frame, suppress the
        # reached *announcement*. A4 = A: also suppress guidance in the
        # same frame. The reached state machine has already updated.
        if obstacle_fired:
            self._log.debug("guidance_and_reached_suppressed_by_obstacle")
            # F2 = B: still update last-visible flag so the lost-sight
            # detector doesn't get confused on the next frame.
            self.context.landmark_visible_last_frame = bool(relevant_detections)
            return

        # Part 4: guidance check (A3 = A: only after temporal consistency).
        self._check_guidance(wp, relevant_detections)

        self._check_reached(wp, relevant_detections)

    # -----------------------------------------------------------------
    # Manager-facing actions
    # -----------------------------------------------------------------

    def disambiguate(self, chosen_class: str) -> None:
        """Manager calls this when the user has picked a class (C7 sub 1)."""
        wp = self.context.current_waypoint()
        if wp is None:
            return
        wp.locked_class = chosen_class
        self.context.awaiting_disambiguation = False
        self.context.disambiguation_choices = []
        self.context.disambiguation_started_at = None
        self.context.disambiguation_retry_count = 0
        # D6 = A philosophy: history is stale, clear so the buffer reflects
        # only what's seen *after* the user committed to one class.
        self.context.detection_history.clear()
        self._log.info(
            "nav_disambiguated waypoint=%r class=%r",
            wp.normalized_text, chosen_class,
        )

    def disambiguate_fallback(self) -> None:
        """C7 sub 4: timeout fallback. Pick highest-confidence class from the
        current history and lock to it.
        """
        wp = self.context.current_waypoint()
        if wp is None or not self.context.disambiguation_choices:
            return
        # Pick the most-recent-most-frequent of the choices.
        scores: dict[str, int] = {c: 0 for c in self.context.disambiguation_choices}
        for frame_set in self.context.detection_history:
            for c in frame_set:
                if c in scores:
                    scores[c] += 1
        pick = max(scores, key=lambda c: scores[c])
        self._log.info(
            "nav_disambiguation_fallback waypoint=%r pick=%r scores=%s",
            wp.normalized_text, pick, scores,
        )
        self.disambiguate(pick)

    def check_timeout(self, now: Optional[float] = None) -> bool:
        """Poll for the 60s detection timeout (G1) and the 10s disambiguation
        timeout (C7 sub 4). Returns True if any timeout fired.
        """
        t = now if now is not None else time.time()
        fired = False

        # Disambiguation timeout first (it short-circuits the reached path).
        if (
            self.context.awaiting_disambiguation
            and self.context.disambiguation_started_at is not None
            and (t - self.context.disambiguation_started_at) >= DISAMBIGUATION_TIMEOUT_SECONDS
        ):
            self._log.info("nav_disambiguation_timeout — falling back to highest-confidence")
            self.disambiguate_fallback()
            fired = True

        # Detection timeout.
        if (
            not self._timeout_already_fired
            and self.context.current_waypoint_started_at is not None
            and (t - self.context.current_waypoint_started_at)
                >= LANDMARK_DETECTION_TIMEOUT_SECONDS
            and self.context.last_detection_at is None
        ):
            self._timeout_already_fired = True
            self._log.info("nav_detection_timeout — fired on_detection_timeout callback")
            try:
                self._on_detection_timeout()
            except Exception:
                self._log.exception("on_detection_timeout callback raised")
            fired = True

        return fired

    def reset_for_new_waypoint(self) -> None:
        """Called by the manager after advance(). Clears 'fired' latches."""
        self._reached_already_fired = False
        self._timeout_already_fired = False

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _filter_relevant(self, detections: list[dict], wp: Waypoint) -> list[dict]:
        """Keep only detections whose class is in the waypoint's active set
        AND confidence >= threshold (C4). Region/distance are passed through
        with a logged warning if missing (G5 = B).
        """
        active = set(wp.active_classes())
        kept: list[dict] = []
        for d in detections:
            cls = d.get("class_name")
            conf = d.get("confidence", 0.0)
            if cls is None or cls not in active:
                continue
            if conf < CONFIDENCE_THRESHOLD:
                continue
            # G5 = B: warn once if fields missing.
            if "region" not in d and not self._warned_missing_region:
                self._log.warning(
                    "detection_missing_region: partner pipeline did not "
                    "supply 'region'. Treating as 'unknown'; reached check "
                    "will fail until field is present."
                )
                self._warned_missing_region = True
            if "distance_category" not in d and not self._warned_missing_distance:
                self._log.warning(
                    "detection_missing_distance_category: partner pipeline "
                    "did not supply 'distance_category'. Treating as 'unknown'."
                )
                self._warned_missing_distance = True
            kept.append(d)
        return kept

    def _classes_passing_consistency(self, candidate_classes: list[str]) -> list[str]:
        """Return the subset of `candidate_classes` that appear in at least
        DETECTION_MIN_HITS frames of the current history buffer.
        """
        if len(self.context.detection_history) < DETECTION_MIN_HITS:
            return []
        confirmed = []
        for cls in candidate_classes:
            hits = sum(1 for frame_set in self.context.detection_history if cls in frame_set)
            if hits >= DETECTION_MIN_HITS:
                confirmed.append(cls)
        return confirmed

    def _trigger_disambiguation(self, choices: list[str]) -> None:
        wp = self.context.current_waypoint()
        if wp is None:
            return
        self.context.awaiting_disambiguation = True
        self.context.disambiguation_choices = list(choices)
        self.context.disambiguation_started_at = time.time()
        self._log.info(
            "nav_disambiguation_needed waypoint=%r choices=%s",
            wp.normalized_text, choices,
        )
        try:
            self._on_disambiguation_needed(list(choices))
        except Exception:
            self._log.exception("on_disambiguation_needed callback raised")

    def _check_obstacles(self, detections: list[dict], wp: Waypoint) -> bool:
        """Part 3 obstacle pass. Runs every frame.

        Returns True if at least one obstacle warning fired (which the
        caller uses to suppress the reached announcement per D1 = A).

        Pipeline per frame:
          1. Filter to detections that are obstacle classes, NOT the active
             landmark class (A2), with center region (B1=A, B3=A) and near
             distance (B2=A), confidence >= threshold (G5).
          2. Apply size gate for small objects (A5 + Issue 1 resolution).
          3. Update active_obstacle_classes (used for escalation reset).
          4. Reset escalation for any obstacle that's no longer active
             (it "cleared").
          5. Pick the highest-priority + nearest detection (E6 + B4=A).
          6. Throttle by per-class cooldown (E2+E3+E4); allow a different
             class to bypass the cooldown (E5=A).
          7. Compute the escalation level for the warning.
          8. Fire callback.
        """
        # Skip during waypoint-collection (decision 10 + Issue 2 resolution).
        # If frame processing is off (which it is until first waypoint
        # added), we never even get here. Belt-and-suspenders:
        if not self.context.frame_processing_enabled:
            return False

        landmark_classes = set(wp.active_classes())

        # Step 1+2: filter to actionable obstacle candidates.
        candidates: list[dict] = []
        for d in detections:
            cls = d.get("class_name")
            if cls is None or not is_obstacle(cls):
                continue
            # A2: never warn about the thing we're trying to reach.
            if cls in landmark_classes:
                continue
            if d.get("confidence", 0.0) < CONFIDENCE_THRESHOLD:
                continue
            # B1+B3: only obstacles in the walking corridor.
            if d.get("region") != "center":
                continue
            # B2: only "near" obstacles (advisory-not-realtime: we accept
            # being conservative-early per H2).
            if d.get("distance_category") != "near":
                continue
            # A5 size gate for small objects.
            if not passes_size_gate(cls, d.get("bbox", [])):
                continue
            candidates.append(d)

        # Step 3+4: update active set and reset escalation on cleared
        # obstacles. When an obstacle clears (no longer present), its
        # cooldown also resets so a re-appearance starts a fresh episode
        # at level 0, not a continuation of the prior one.
        new_active: set = {d["class_name"] for d in candidates}
        cleared = self.context.active_obstacle_classes - new_active
        for c in cleared:
            self.context.obstacle_escalation_level.pop(c, None)
            self.context.last_obstacle_warning_at.pop(c, None)
            self._log.debug("obstacle_cleared class=%r", c)
        self.context.active_obstacle_classes = new_active

        if not candidates:
            return False

        # Step 5: prioritize. Higher obstacle_priority first; tie-break by
        # bbox area (proxy for "nearer within the 'near' bucket").
        def _sort_key(d):
            area = (
                (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1])
                if d.get("bbox") else 0
            )
            return (-obstacle_priority(d["class_name"]), -area)

        candidates.sort(key=_sort_key)
        chosen = candidates[0]
        chosen_class = chosen["class_name"]

        # Step 6: cooldown / E5 bypass. We track the last-warned timestamp
        # per class. If the same class is on cooldown, suppress the SPOKEN
        # warning — but still return True so the caller knows an obstacle
        # is in the path (preemption suppresses reached even if we're not
        # speaking about the obstacle this exact frame).
        now = time.time()
        last_at = self.context.last_obstacle_warning_at.get(chosen_class)
        on_cooldown = (
            last_at is not None
            and (now - last_at) < OBSTACLE_WARNING_COOLDOWN_SECONDS
        )
        if on_cooldown:
            return True  # obstacle present, just not announcing this frame

        # Step 7: escalation level.
        # Issue 3 resolution: constrained escalation. Each successive warning
        # within an active episode bumps level by 1, capped at MAX.
        current_level = self.context.obstacle_escalation_level.get(chosen_class, 0)
        # If this is the FIRST warning in an episode (escalation was reset
        # because the obstacle had cleared), current_level is 0 — good.
        # Otherwise advance.
        next_level = min(current_level + (0 if last_at is None else 1),
                         MAX_OBSTACLE_ESCALATION_LEVEL)
        self.context.obstacle_escalation_level[chosen_class] = next_level
        self.context.last_obstacle_warning_at[chosen_class] = now

        # Step 8: fire.
        payload = {
            "class_name": chosen_class,
            "region": chosen.get("region"),
            "distance_category": chosen.get("distance_category"),
            "confidence": chosen.get("confidence", 0.0),
            "escalation_level": next_level,
            "is_person": is_person(chosen_class),
        }
        self._log.info(
            "obstacle_warning class=%r escalation=%d is_person=%s conf=%.2f",
            chosen_class, next_level, payload["is_person"],
            payload["confidence"],
        )
        if self._on_obstacle is not None:
            try:
                self._on_obstacle(payload)
            except Exception:
                self._log.exception("on_obstacle callback raised")
        return True

    def _check_guidance(self, wp: Waypoint, frame_detections: list[dict]) -> None:
        """Part 4: emit a guidance cue when meaningful state changes.

        Decisions implemented:
          - A1 = B: speak only when region OR distance changes (or class on
            disambig).
          - A3 = A: require temporal consistency (2-of-3) before any cue.
          - C5 = A: state cleared on advance, so first confirmed detection
            after a new waypoint produces a "first_cue".
          - D4 = A: when distance == "near" but region != "center", still
            speak so user can course-correct.
          - E5 = A: missing region/distance => no cue.
          - F2 = B: when landmark was visible last frame and isn't now,
            emit a "lost_sight" cue once.
          - B6 = C + Interpretation 1: class-name change forces a fresh
            full cue even if region/distance haven't changed.
        """
        # F2 = B: lost-sight detection (visible -> not visible).
        currently_visible = bool(frame_detections)
        if self.context.landmark_visible_last_frame and not currently_visible:
            # Only speak lost-sight if we had previously spoken about it
            # (otherwise we never claimed it was found in the first place).
            if self.context.last_spoken_region is not None:
                self._fire_guidance(
                    class_name=self.context.last_spoken_class or wp.normalized_text,
                    region=None,
                    distance=None,
                    change_type="lost_sight",
                    is_class_change=False,
                )
                # Reset spoken state so when the landmark comes back we get
                # a fresh first_cue (and the test for "next is fresh" works).
                self.context.last_spoken_region = None
                self.context.last_spoken_distance = None
        self.context.landmark_visible_last_frame = currently_visible

        if not currently_visible:
            return

        # A3 = A: need temporal consistency confirmation before any cue.
        active = wp.active_classes()
        if not self._classes_passing_consistency(active):
            return

        # Pick the detection to describe: prefer the highest-confidence
        # match in the current frame. If multiple alias classes (rare here
        # because disambiguation would have triggered), pick highest conf.
        best = max(
            frame_detections,
            key=lambda d: d.get("confidence", 0.0),
        )
        region = best.get("region")
        distance = best.get("distance_category")
        cls = best["class_name"]

        # E5 = A: missing region/distance => no cue.
        if region is None or distance is None:
            return

        # Detect what kind of change this is.
        was_class = self.context.last_spoken_class
        was_region = self.context.last_spoken_region
        was_distance = self.context.last_spoken_distance
        is_class_change = (was_class is not None) and (cls != was_class)
        is_first_cue = was_region is None and was_distance is None

        if is_first_cue:
            change_type = "first_cue"
        elif is_class_change:
            # B6 Interpretation 1: class changed -> force full fresh cue.
            change_type = "first_cue"
        elif region != was_region and distance != was_distance:
            change_type = "both_changed"
        elif region != was_region:
            change_type = "region_changed"
        elif distance != was_distance:
            change_type = "distance_changed"
        else:
            # No change -> no speech. C2 = A: change-triggered only, no timer.
            return

        self._fire_guidance(
            class_name=cls,
            region=region,
            distance=distance,
            change_type=change_type,
            is_class_change=is_class_change,
        )

    def _fire_guidance(
        self,
        class_name: str,
        region: Optional[str],
        distance: Optional[str],
        change_type: str,
        is_class_change: bool,
    ) -> None:
        """Update last-spoken state and invoke the guidance callback."""
        # Update last-spoken state BEFORE firing so the manager sees a
        # consistent context if it inspects it.
        if change_type != "lost_sight":
            self.context.last_spoken_region = region
            self.context.last_spoken_distance = distance
            self.context.last_spoken_class = class_name

        payload = {
            "class_name": class_name,
            "region": region,
            "distance_category": distance,
            "change_type": change_type,
            "is_class_change": is_class_change,
        }
        self._log.info(
            "guidance_cue class=%r region=%r distance=%r change=%r",
            class_name, region, distance, change_type,
        )
        if self._on_guidance is not None:
            try:
                self._on_guidance(payload)
            except Exception:
                self._log.exception("on_guidance callback raised")

    def _check_reached(self, wp: Waypoint, frame_detections: list[dict]) -> None:
        """Apply E1 = A: temporal-consistency-confirmed AND center AND near.

        The "center + near" check uses the *current* frame's matching
        detection that has the highest confidence (C6 / C7 = first match
        from a single class after lock). The temporal history governs the
        confirmation half of the rule, not the centered/near half.
        """
        # E1 needs (a) temporal confirmation and (b) a centered+near detection
        # right now.
        active = wp.active_classes()
        if not self._classes_passing_consistency(active):
            return

        # Pick the detection most worth firing on: prefer one that is
        # near AND center; among those, highest confidence (C6 / C7 sub-tie-
        # break).
        candidates = [
            d for d in frame_detections
            if d.get("region") == "center" and d.get("distance_category") == "near"
        ]
        if not candidates:
            return
        best = max(candidates, key=lambda d: d.get("confidence", 0.0))

        # Fire.
        wp.status = self._mark_reached_status()
        self._reached_already_fired = True
        # D6 = A: clear history so a re-arming detector after advance has a
        # clean slate.
        self.context.detection_history.clear()
        self._log.info(
            "nav_waypoint_reached waypoint=%r class=%r conf=%.2f",
            wp.normalized_text, best["class_name"], best.get("confidence", 0.0),
        )
        try:
            self._on_reached(wp)
        except Exception:
            self._log.exception("on_reached callback raised")

    @staticmethod
    def _mark_reached_status():
        # Import lazily to avoid circular imports at module load time.
        from .context import WaypointStatus
        return WaypointStatus.REACHED
