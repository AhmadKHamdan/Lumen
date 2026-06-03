"""Navigation task context and waypoint data structures.

This module owns the state of a single Navigation task:
- The destination the user requested (raw label, e.g. "kitchen").
- The ordered list of waypoints the user has provided so far.
- Sub-state flags that govern the dialog phase (awaiting_waypoint,
  frame_processing_enabled, retry_count, last_prompt_at).

Design decisions implemented here:
- 3:  Waypoint is a @dataclass with raw_text, normalized_text,
      target_class, status, added_at.
- 4:  Container is a plain list[Waypoint] with a current_index pointer,
      so completed waypoints stay around for logs and evaluation.
- 16: NavigationContext is intended to live as a single attribute on the
      Session object (one per WebSocket). This module holds no reference
      back to the Session itself.

target_class is populated later by the landmark-mapping layer (Part 2)
and starts as None.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# D2 = A: 3-frame temporal-consistency window. D3 = B: majority threshold.
DETECTION_WINDOW: int = 3
DETECTION_MIN_HITS: int = 2  # majority of 3


class WaypointStatus(str, Enum):
    """Lifecycle status of a single waypoint."""

    PENDING = "pending"    # Stored but not yet the active target.
    ACTIVE = "active"      # Currently being searched for / guided to.
    REACHED = "reached"    # Detected centered + near, announced (Part 2/4).
    SKIPPED = "skipped"    # Cancelled or aborted before being reached.


@dataclass
class Waypoint:
    """A single user-provided navigation milestone."""

    raw_text: str
    normalized_text: str
    # B3 = A: a waypoint can map to several COCO classes (e.g. "seat" ->
    # ["chair", "couch", "bench"]). Filled in by landmark mapping at add time
    # (B1 = A). Empty list means resolution failed; under A4 = A the manager
    # rejects such waypoints before they're stored, so an active Waypoint
    # always has at least one class here.
    target_classes: list[str] = field(default_factory=list)
    status: WaypointStatus = WaypointStatus.PENDING
    added_at: float = field(default_factory=time.time)
    # C7 sub-decision 1: when disambiguation pins this waypoint to a single
    # class for the rest of its life, that class is stored here. None means
    # "no disambiguation has happened; treat all target_classes equally."
    locked_class: Optional[str] = None

    def active_classes(self) -> list[str]:
        """Classes the detector should actually look for right now.

        After disambiguation (C7), only the locked_class is searched;
        otherwise every entry in target_classes is fair game.
        """
        if self.locked_class is not None:
            return [self.locked_class]
        return list(self.target_classes)


@dataclass
class NavigationContext:
    """All state for one in-flight Navigation task.

    One instance per active task; lives on the per-WebSocket Session.
    """

    destination: str                                 # Raw destination, e.g. "kitchen".
    waypoints: list[Waypoint] = field(default_factory=list)
    current_index: int = 0

    # Dialog sub-state.
    awaiting_waypoint: bool = True                   # True from entry until first waypoint stored.
    retry_count: int = 0                             # Re-prompts after empty/unrecognized input.
    last_prompt_at: float = field(default_factory=time.time)

    # Perception gate (decision 10): YOLO inference is OFF until first waypoint stored.
    frame_processing_enabled: bool = False

    # ---- Detection layer state (Part 2) ----

    # D5 = A: per-frame "what classes did we see?" history. We push a set of
    # class names per frame and look at the last N frames for temporal
    # consistency (D2 = A: window = 3). Bounded via maxlen=DETECTION_WINDOW.
    detection_history: deque = field(default_factory=lambda: deque(maxlen=DETECTION_WINDOW))
    last_detection_at: Optional[float] = None        # Used for G1 detection timeout.

    # C7 sub-decision 3: while we're awaiting the user's chair-vs-couch
    # answer, this is True and "reached" detection is paused. Frame
    # processing stays on for obstacle awareness (Part 3).
    awaiting_disambiguation: bool = False
    disambiguation_choices: list[str] = field(default_factory=list)
    disambiguation_started_at: Optional[float] = None
    disambiguation_retry_count: int = 0              # C7 sub-decision 2.

    # G1: track when the current waypoint became active so we can detect
    # "I've been looking for the chair for 60s and nothing".
    current_waypoint_started_at: Optional[float] = None

    # ---- Obstacle-warning sub-state (Part 3) ----
    # E2/E3: cooldown per obstacle class. Stores last-warned-at timestamp.
    last_obstacle_warning_at: dict[str, float] = field(default_factory=dict)
    # Issue 3 resolution: constrained escalation ladder. Level 0 = mild,
    # 1 = "still there" reminder, 2 = "still in path, please be careful".
    # Level increments on each cooldown re-fire while the same obstacle
    # remains in the path; resets when the obstacle clears.
    obstacle_escalation_level: dict[str, int] = field(default_factory=dict)
    # The set of obstacle classes "currently active" (seen in the most
    # recent frame in the center, near). Used to detect "obstacle cleared"
    # so escalation can reset.
    active_obstacle_classes: set = field(default_factory=set)

    # ---- Guidance sub-state (Part 4) ----
    # C4 = A: track the last-spoken cue so we only re-speak when region OR
    # distance changes. Cleared on advance() so the next waypoint always
    # gets a fresh first cue (C5 = A).
    last_spoken_region: Optional[str] = None
    last_spoken_distance: Optional[str] = None
    # B6 = C (Interpretation 1): when the class name changes (e.g. after a
    # disambiguation lock), force a fresh full cue even if region/distance
    # match what was last spoken.
    last_spoken_class: Optional[str] = None
    # F2 = B: track whether the landmark was visible last frame, so we can
    # speak a "lost sight" cue when it transitions visible -> not visible.
    landmark_visible_last_frame: bool = False

    # ---- Part 5 sub-state ----
    # When True, system has just announced "What's next?" after reaching a
    # waypoint with an empty queue. User can say "done" / a new landmark /
    # "stop".
    awaiting_next_or_done: bool = False

    # When True, user said "I'm here" but the detector hasn't seen the
    # current waypoint as centered+near. System asked back; awaiting yes/no.
    awaiting_arrival_confirmation: bool = False

    # Captures the start timestamp of the whole task for the duration_seconds
    # summary metric. Set in NavigationTaskManager.on_navigation_command.
    task_started_at: Optional[float] = None

    # ---- Waypoint queue helpers ----

    def add_waypoint(
        self,
        raw_text: str,
        normalized_text: str,
        target_classes: Optional[list[str]] = None,
    ) -> Waypoint:
        """Append a waypoint. First added becomes ACTIVE; others queue as PENDING.

        target_classes is the result of landmark_map.resolve() at the caller
        side (B1 = A, eager resolution). An empty list reaching here is a
        bug: under A4 = A the manager should reject unmappable waypoints
        before calling add_waypoint.
        """
        wp = Waypoint(
            raw_text=raw_text,
            normalized_text=normalized_text,
            target_classes=list(target_classes or []),
        )
        if not self.waypoints:
            wp.status = WaypointStatus.ACTIVE
            self.current_waypoint_started_at = time.time()
        self.waypoints.append(wp)
        return wp

    def current_waypoint(self) -> Optional[Waypoint]:
        if 0 <= self.current_index < len(self.waypoints):
            return self.waypoints[self.current_index]
        return None

    def has_more_waypoints(self) -> bool:
        return self.current_index + 1 < len(self.waypoints)

    def mark_current_reached(self) -> Optional[Waypoint]:
        wp = self.current_waypoint()
        if wp is not None:
            wp.status = WaypointStatus.REACHED
        return wp

    def advance(self) -> Optional[Waypoint]:
        """Advance to the next waypoint. Returns the new active one, or None.

        D6 = A: detection history is cleared on every transition so stale
        observations of the previous landmark do not bleed into the next.
        """
        self.detection_history.clear()
        self.last_detection_at = None
        self.disambiguation_choices = []
        self.awaiting_disambiguation = False
        self.disambiguation_started_at = None
        self.disambiguation_retry_count = 0
        # Part 3: obstacle warnings are per-task, but cooldowns roll forward
        # across waypoints (the chair you were just warned about is still
        # the same chair on the way to the next landmark).
        # We deliberately do NOT clear last_obstacle_warning_at on advance.
        # Part 4 (C5 = A): guidance "last spoken" state IS cleared on
        # advance so the next waypoint's first cue is always full.
        self.last_spoken_region = None
        self.last_spoken_distance = None
        self.last_spoken_class = None
        self.landmark_visible_last_frame = False

        if not self.has_more_waypoints():
            return None
        self.current_index += 1
        nxt = self.waypoints[self.current_index]
        nxt.status = WaypointStatus.ACTIVE
        self.current_waypoint_started_at = time.time()
        return nxt

    def skip_current(self) -> Optional[Waypoint]:
        """G3 = A: mark current waypoint SKIPPED, then advance to next."""
        wp = self.current_waypoint()
        if wp is not None and wp.status != WaypointStatus.REACHED:
            wp.status = WaypointStatus.SKIPPED
        return self.advance()

    def cancel_remaining(self) -> None:
        """Mark every not-yet-reached waypoint as SKIPPED (called on cancel/timeout)."""
        for wp in self.waypoints[self.current_index:]:
            if wp.status != WaypointStatus.REACHED:
                wp.status = WaypointStatus.SKIPPED
        self.detection_history.clear()
        self.last_detection_at = None

    def summary(self) -> dict:
        """Build the completion/cancellation summary payload (C2 = A).

        Used by the manager when emitting task_completed / task_cancelled
        FSM events. Sprint 5 evaluation reads these fields directly.
        """
        reached = sum(1 for w in self.waypoints if w.status == WaypointStatus.REACHED)
        skipped = sum(1 for w in self.waypoints if w.status == WaypointStatus.SKIPPED)
        duration = None
        if self.task_started_at is not None:
            duration = time.time() - self.task_started_at
        return {
            "destination": self.destination,
            "waypoints_total": len(self.waypoints),
            "waypoints_reached": reached,
            "waypoints_skipped": skipped,
            "duration_seconds": duration,
        }
