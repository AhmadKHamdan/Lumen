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

__all__ = ["start", "stop", "resolve_goal", "is_known_goal",
           "preload_in_background"]


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


def preload_in_background() -> None:
    """Warm the slow paths before the first user speaks (opt-in via the
    LUMEN_NAV_PRELOAD env var - see main.py):

    - load + warm the navigation models (first-ever run also downloads
      YOLOv8m ~50 MB and SegFormer-B0 ~15 MB), so the first "navigate to..."
      doesn't sit in a minute of silence;
    - pre-synthesize the FIXED guidance phrases into the TTS LRU cache, so
      demo-day wifi hiccups can't silence the common lines. Best effort: any
      failure is logged and ignored (dynamic phrases still synthesize live).

    The phrase list mirrors controller.py's fixed strings; drift only costs a
    cache miss, never a wrong utterance.
    """
    import threading

    def _work() -> None:
        try:
            from . import models
            models.ensure_loaded()
        except Exception:
            log.exception("Navigation model preload failed (will retry lazily)")
        try:
            from services import tts_service
            from .goals import GOAL_INDICATORS
            fixed = [
                "Give me a moment while I get ready.",
                "Slow down. Move the phone slowly.",
                "Keep scanning.",
                "Good, keep scanning.",
                "Almost done — keep turning until you face where you started.",
                "I don't see a door yet. Keep scanning the room slowly.",
                "Still looking for a door. Keep moving the camera around the room.",
                "No door yet — keep scanning the walls slowly.",
                "Okay, the way ahead is clear.",
                "You're right at the door. Reach out with your hand, open it, "
                "walk through the doorway, and take two or three steps into the room.",
                "You're through. Now slowly turn to your right, all the way "
                "around, until you are facing where you started, so I can scan "
                "this room.",
            ] + [
                f"Looking for the {g}. Let's scan the room — slowly turn to "
                "your right, all the way around, until you are facing where "
                "you started."
                for g in GOAL_INDICATORS
            ]
            for phrase in fixed:
                try:
                    tts_service.synthesize(phrase)
                except Exception:
                    log.warning("TTS warmup failed at %r - offline? Stopping "
                                "warmup; live synthesis unaffected.", phrase[:40])
                    break
            log.info("Navigation preload complete (%d phrases cached).", len(fixed))
        except Exception:
            log.exception("TTS warmup failed")

    threading.Thread(target=_work, name="lumen-nav-preload", daemon=True).start()
