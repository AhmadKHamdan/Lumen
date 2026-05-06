"""
Object Allocation task — Sprint 1 stub.

Sprint 3 will fill this with:
    * YOLOv8n inference on the latest frame
    * Spatial reasoning (left/center/right, near/medium/far)
    * Temporal consistency check (target detected in 3 of last 5 frames)
    * Template-based guidance ("the cup is to your left, near you")
    * Throttled updates (only when region or distance changes)
    * Edge cases: never detected, multiple instances, lost from view
    * Completion logic: user confirmation, cancellation, 60s timeout

For Sprint 1 we just log the task initiation - the FSM transition into
ObjectAllocationActive happens correctly, the confirmation TTS plays, but
the actual detection loop is empty.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from api.session import Session

log = logging.getLogger("lumen.task.object_allocation")


def start(session: "Session", target: str) -> None:
    """Stub entry hook called when the FSM enters ObjectAllocationActive.

    Sprint 1: just logs. Sprint 3 will spawn an asyncio task that reads
    ``session.latest_frame`` on a tick, runs YOLO, and emits TTS guidance.
    """
    log.info(
        "Session %s: would start object allocation for target=%r (Sprint 1 stub)",
        session.id, target,
    )


def stop(session: "Session") -> None:
    """Stub exit hook called when the FSM leaves ObjectAllocationActive.

    Sprint 3 will cancel the running detection loop here.
    """
    log.info("Session %s: stopping object allocation (Sprint 1 stub)", session.id)
