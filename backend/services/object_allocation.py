"""
Object Allocation task (Sprint 3).

When the FSM enters ``ObjectAllocationActive``, :func:`start` spawns an asyncio
loop that, a few times per second:

    1. grabs the latest decoded camera frame off the session,
    2. runs YOLOv8n on it (in a thread, so the event loop keeps serving),
    3. keeps only detections of the requested target class,
    4. feeds the result to a :class:`GuidanceTracker`, which applies a
       temporal-consistency filter (target must appear in >= 3 of the last 5
       frames before we trust it), picks the most head-on instance, classifies
       direction + distance, and decides what (if anything) to say,
    5. speaks whatever phrase the tracker returns.

The decision logic lives in :class:`GuidanceTracker` - a pure, clock-injected
state machine with no I/O - so it can be unit-tested deterministically. The
async ``_run`` shell only does the I/O (frame grab, YOLO call, TTS, sleep).

Tracker outcomes:
    - ``("guide", phrase)``   normal directional guidance (throttled)
    - ``("scan",  phrase)``   periodic "slowly turn around" while never seen
    - ``("lost",  phrase)``   one announcement when a seen target drops out
    - ``("timeout", phrase)`` give up after never seeing it within 60 s

Completion is **user-driven**: the tracker never declares success. The user
says "got it" (handled in audio_handler -> FSM task_complete), which tears this
loop down via :func:`stop`. The only autonomous exit is the not-found timeout.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import TYPE_CHECKING, Optional, Sequence

from services import guidance_generator, spatial_reasoning, tts_service, yolo_service

if TYPE_CHECKING:
    from api.session import Session

log = logging.getLogger("lumen.task.object_allocation")

# ---------- tuning knobs ----------

DETECTION_HZ = 5                 # frames analysed per second
_LOOP_INTERVAL = 1.0 / DETECTION_HZ

TEMPORAL_WINDOW = 5              # remember presence over the last N frames
TEMPORAL_MIN_HITS = 3           # ... and require this many to "confirm" the target

CONF_THRESHOLD = 0.35           # YOLO confidence floor for this task

GUIDANCE_REAFFIRM_SEC = 6.0     # re-speak unchanged guidance at most this often
SCAN_PROMPT_INTERVAL_SEC = 8.0  # cadence of "turn around" prompts while unseen
TASK_TIMEOUT_SEC = 60.0         # give up if never seen within this many seconds


class GuidanceTracker:
    """Pure decision state machine for Object Allocation guidance.

    Call :meth:`update` once per analysed frame with the current detections,
    the frame shape, and a monotonic timestamp. It returns an optional
    ``(action, phrase)`` describing what to speak, or ``None`` to stay quiet.

    No I/O, no global clock - the caller injects ``now`` - so the whole policy
    (temporal filter, throttling, scan/lost/timeout edges) is unit-testable.
    """

    def __init__(
        self,
        target: str,
        *,
        window: int = TEMPORAL_WINDOW,
        min_hits: int = TEMPORAL_MIN_HITS,
        reaffirm_sec: float = GUIDANCE_REAFFIRM_SEC,
        scan_interval_sec: float = SCAN_PROMPT_INTERVAL_SEC,
        timeout_sec: float = TASK_TIMEOUT_SEC,
    ) -> None:
        self.target = target
        self.min_hits = min_hits
        self.reaffirm_sec = reaffirm_sec
        self.scan_interval_sec = scan_interval_sec
        self.timeout_sec = timeout_sec

        self.presence: deque[bool] = deque(maxlen=window)
        self.started: Optional[float] = None
        self.ever_seen = False
        self.last_region: Optional[str] = None
        self.last_spoken_key: Optional[tuple[str, str]] = None
        self.last_spoken_at = 0.0
        self.last_scan_at = 0.0

    def update(
        self,
        detections: Sequence,
        frame_shape: Optional[Sequence[int]],
        now: float,
    ) -> Optional[tuple[str, str]]:
        if self.started is None:
            self.started = now
            # Delay the first scan prompt by a full interval so it doesn't talk
            # over the "Looking for your cup" confirmation at task start.
            self.last_scan_at = now

        # Safety net: give up if we've never seen the target in time.
        if not self.ever_seen and (now - self.started) >= self.timeout_sec:
            return ("timeout", guidance_generator.timeout_phrase(self.target))

        present = len(detections) > 0
        self.presence.append(present)
        hits = sum(self.presence)
        confirmed = hits >= self.min_hits

        if confirmed and present and frame_shape is not None:
            h, w = int(frame_shape[0]), int(frame_shape[1])
            best = spatial_reasoning.most_centered(detections, w)
            # Pass label so per-class size priors drive distance bucketing -
            # a laptop occupying 40 % of frame width is "near", a cup
            # occupying 18 % is also "near", same image -> different verdict.
            info = spatial_reasoning.locate(best.box, w, h, label=best.label)
            self.ever_seen = True
            self.last_region = info.region
            key = (info.region, info.distance)
            stale = (now - self.last_spoken_at) >= self.reaffirm_sec
            if key != self.last_spoken_key or stale:
                first = self.last_spoken_key is None
                phrase = (
                    guidance_generator.first_seen_phrase(self.target, info)
                    if first
                    else guidance_generator.guidance_phrase(self.target, info)
                )
                self.last_spoken_key = key
                self.last_spoken_at = now
                return ("guide", phrase)
            return None

        if self.ever_seen and hits == 0:
            # Was visible, now gone for the whole window -> announce once.
            if self.last_spoken_key is not None:
                self.last_spoken_key = None
                self.last_spoken_at = now
                return ("lost", guidance_generator.lost_phrase(self.target, self.last_region))
            return None

        if not self.ever_seen:
            # Still hunting for the first sighting -> periodic scan prompt.
            if (now - self.last_scan_at) >= self.scan_interval_sec:
                self.last_scan_at = now
                return ("scan", guidance_generator.scanning_phrase(self.target))
            return None

        # Transient miss (1-2 of last 5) or post-lost silence -> stay quiet.
        return None


def start(session: "Session", target: str) -> None:
    """FSM entry hook: launch the detection loop for ``target``."""
    loop = asyncio.get_running_loop()
    _cancel_existing(session)
    log.info("Session %s: starting object allocation for target=%r", session.id, target)
    session.detection_task = loop.create_task(_run(session, target))


def stop(session: "Session") -> None:
    """FSM exit hook: cancel the running detection loop, if any."""
    log.info("Session %s: stopping object allocation", session.id)
    _cancel_existing(session)


def _cancel_existing(session: "Session") -> None:
    task = getattr(session, "detection_task", None)
    if task is not None and not task.done():
        task.cancel()
    session.detection_task = None


def _detect_target(frame, target: str) -> list:
    """Blocking helper run in a thread: detect ``target`` instances in ``frame``."""
    return yolo_service.detect(
        frame, conf_threshold=CONF_THRESHOLD, target_labels=[target],
    )


async def _speak(session: "Session", text: str) -> None:
    """Synthesize ``text`` (in a thread) and push the MP3 to the client."""
    if not text:
        return
    loop = asyncio.get_running_loop()
    try:
        mp3 = await loop.run_in_executor(None, tts_service.synthesize, text)
    except Exception:
        log.exception("Session %s: guidance TTS failed for %r", session.id, text)
        return
    await session.send_tts(mp3)
    log.info("Session %s: OA guidance: %r", session.id, text)


async def _run(session: "Session", target: str) -> None:
    """The detection + guidance loop. Runs until cancelled or it times out."""
    loop = asyncio.get_running_loop()
    tracker = GuidanceTracker(target)

    try:
        while True:
            now = time.monotonic()

            frame = session.latest_frame
            detections: list = []
            if frame is not None and getattr(frame, "size", 0):
                try:
                    detections = await loop.run_in_executor(
                        None, _detect_target, frame, target,
                    )
                except Exception:
                    log.exception("Session %s: detection failed", session.id)
                    detections = []

            frame_shape = frame.shape if frame is not None else None
            result = tracker.update(detections, frame_shape, now)

            if result is not None:
                action, phrase = result
                await _speak(session, phrase)
                if action == "timeout":
                    # Failure exit (user-driven completion is the normal path).
                    session.fsm.handle_event("user_stop")
                    return

            await asyncio.sleep(_LOOP_INTERVAL)

    except asyncio.CancelledError:
        log.info("Session %s: object allocation loop cancelled", session.id)
        raise
    except Exception:
        log.exception("Session %s: object allocation loop crashed", session.id)
