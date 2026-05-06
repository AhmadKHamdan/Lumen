"""
Navigation task — Sprint 1 stub.

Sprint 4 will fill this with:
    * Initial-landmark prompt ("what's the first landmark?")
    * Waypoint storage in task context (list of strings)
    * Landmark detection from YOLOv8n stream using COCO furniture as proxies
    * Turn-by-turn template guidance ("continue forward", "turn slightly left")
    * Obstacle warnings that preempt navigation cues
    * Waypoint-reached detection → prompt for next waypoint
    * Completion: user confirmation ("I'm here"), cancellation, timeout

For Sprint 1 we just log the task initiation.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from api.session import Session

log = logging.getLogger("lumen.task.navigation")


def start(session: "Session", destination: str) -> None:
    """Stub entry hook called when the FSM enters NavigationActive.

    Sprint 1: just logs. Sprint 4 will spawn a navigation loop that reads
    frames, detects landmarks, and emits turn-by-turn guidance.
    """
    log.info(
        "Session %s: would start navigation to destination=%r (Sprint 1 stub)",
        session.id, destination,
    )


def stop(session: "Session") -> None:
    """Stub exit hook called when the FSM leaves NavigationActive.

    Sprint 4 will cancel the running navigation loop here.
    """
    log.info("Session %s: stopping navigation (Sprint 1 stub)", session.id)
