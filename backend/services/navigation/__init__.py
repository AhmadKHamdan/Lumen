"""
Navigation task (Sprint 4) — goal-directed exploration.

The user names only a destination ("navigate to the kitchen") and Lumen guides
them there room-by-room: a compass-tracked 360 scan of the room, semantic arrival
detection (fridge + oven => kitchen), a three-layer door perception funnel
(custom door detector -> geometry gates -> 4-class semantic verifier), guided
door approach with step-count distances, doorway-transit detection, and an
obstacle watchdog (YOLO named objects + floor segmentation) while walking.

Ported from the standalone webdemo prototype (see docs/Exploration_Navigation_
Design.md for the design and its rationale). The decision logic is byte-for-byte
the field-tested prototype's; what changed is the plumbing: per-session state
instead of module globals, the session's WebSocket frame stream instead of HTTP
POSTs, and server-side gTTS instead of browser speechSynthesis.

Public API (FSM entry/exit hooks, mirroring object_allocation):
    start(session, destination)  — spawn the navigation loop
    stop(session)                — cancel it
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from .goals import resolve_goal, is_known_goal  # re-exported for callers/tests

if TYPE_CHECKING:
    from api.session import Session

log = logging.getLogger("lumen.task.navigation")

__all__ = ["start", "stop", "resolve_goal", "is_known_goal"]


def start(session: "Session", destination: str) -> None:
    """FSM entry hook: launch the navigation loop toward ``destination``."""
    # Deferred import: engine pulls cv2 + the TTS/vision stack, which pure-logic
    # consumers of this package (unit tests, FSM wiring) must not pay for.
    from . import engine

    loop = asyncio.get_running_loop()
    _cancel_existing(session)
    log.info("Session %s: starting navigation to destination=%r",
             session.id, destination)
    session.detection_task = loop.create_task(engine.run(session, destination))


def stop(session: "Session") -> None:
    """FSM exit hook: cancel the running navigation loop, if any."""
    log.info("Session %s: stopping navigation", session.id)
    _cancel_existing(session)


def _cancel_existing(session: "Session") -> None:
    task = getattr(session, "detection_task", None)
    if task is not None and not task.done():
        task.cancel()
    session.detection_task = None
