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

from services import (
    guidance_generator,
    hand_service,
    reach_guidance,
    spatial_reasoning,
    tts_service,
    yolo_service,
)

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

# Debounce spatial changes so YOLO's frame-to-frame box jitter around a
# bucket boundary doesn't cause "medium -> near -> medium -> ..." spam.
# We require the same (region, distance) for this many consecutive frames
# before switching the "official" bucket. 2 frames ~= 400 ms at 5 FPS.
SPATIAL_STREAK_TO_ACCEPT = 2

# Reach mode (Sprint 5: hand guidance, only fires when target is near AND a
# hand is detected).
REACH_REAFFIRM_SEC = 5.0

# Once reach mode has spoken, don't fall back to body-direction guidance
# until the hand has been absent for this long. Handles the case where
# MediaPipe momentarily fails to detect the hand between two good frames.
REACH_GRACE_SEC = 3.0

# Frozen-feed guard (same rationale as the navigation engine's): if the phone
# stops sending frames, re-analysing the stale image would keep re-confirming a
# detection that may no longer be in front of the user. A stale frame is
# treated as no frame - the tracker then ages it out via its presence window.
FRAME_STALE_SEC = 2.0


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
        self.last_distance: Optional[str] = None
        self.last_spoken_key: Optional[tuple[str, str]] = None
        self.last_spoken_at = 0.0
        self.last_scan_at = 0.0
        # Reach-mode state (Sprint 5). last_reach_key is the (state, direction)
        # tuple of the most recently spoken reach cue, used for throttling.
        self.last_reach_key: Optional[tuple[str, str]] = None
        self.last_reach_at = 0.0

        # Spatial debouncing (Sprint 5 refinement). YOLO's box size flickers
        # frame-to-frame near bucket boundaries; we require SPATIAL_STREAK_TO_ACCEPT
        # consecutive frames of the same (region, distance) to switch the
        # "official" bucket used for phrase selection.
        self._raw_spatial_key: Optional[tuple[str, str]] = None
        self._spatial_streak = 0

        # Haptic-pulse hint the loop should ship after this update(). None
        # means "no vibration this frame". The loop reads and clears it every
        # tick. Set by _update_reach when the fingertip enters "almost" or
        # "touching" states.
        self.pending_haptic: Optional[str] = None

        # Previous motion state, used to detect a still -> moving transition
        # and reset the smoothed spatial state so the guidance re-issues
        # immediately after the user actually moved.
        self._prev_motion_state: str = "still"

    def should_check_hand(self) -> bool:
        """Cheap pre-check for the detection loop: only worth running
        MediaPipe when the target was most recently classified as 'near'.
        Saves ~15 ms per frame in the common 'still looking' case."""
        return self.last_distance == "near"

    def update(
        self,
        detections: Sequence,
        frame_shape: Optional[Sequence[int]],
        now: float,
        hand_pose=None,
        motion_state: str = "still",
    ) -> Optional[tuple[str, Optional[str]]]:
        if self.started is None:
            self.started = now
            # Delay the first scan prompt by a full interval so it doesn't talk
            # over the "Looking for your cup" confirmation at task start.
            self.last_scan_at = now

        # Motion-aware: if the user has just started moving, reset the
        # smoothed spatial state so guidance responds immediately to their
        # new position. The debounce is there to filter YOLO jitter while
        # the user is still, not to hide real position changes.
        entered_motion = (motion_state != "still"
                          and self._prev_motion_state == "still")
        if entered_motion:
            self._spatial_streak = 0
            self._raw_spatial_key = None
            self.last_spoken_key = None   # so next detection is announced fresh
        self._prev_motion_state = motion_state
        force_accept_bucket = motion_state != "still"

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

            # Spatial debounce: don't accept a new (region, distance) bucket
            # until we've seen it SPATIAL_STREAK_TO_ACCEPT times in a row.
            # This filters YOLO's per-frame box-size jitter that used to fire
            # 3 different phrases in 600 ms.
            raw_key = (info.region, info.distance)
            if raw_key == self._raw_spatial_key:
                self._spatial_streak += 1
            else:
                self._raw_spatial_key = raw_key
                self._spatial_streak = 1
            first_bucket_ever = self.last_distance is None
            if (first_bucket_ever
                    or force_accept_bucket
                    or self._spatial_streak >= SPATIAL_STREAK_TO_ACCEPT):
                self.last_region = info.region
                self.last_distance = info.distance
            # Everything downstream uses the smoothed self.last_* values.
            effective_info = _replace_info(info, self.last_region, self.last_distance)

            # Reach mode: target is near AND we can see the user's hand.
            if self.last_distance == "near" and hand_pose is not None:
                return self._update_reach(best, hand_pose, w, h, now)

            # Grace period: hand momentarily lost while we're still near?
            # Stay silent for a few seconds rather than switching back to
            # body-direction mode (which would then alternate with reach
            # mode every ~200 ms on hand-detection flicker).
            if (self.last_distance == "near" and hand_pose is None
                    and self.last_reach_key is not None
                    and (now - self.last_reach_at) < REACH_GRACE_SEC):
                return None

            # Otherwise: standard direction + distance guidance.
            key = (self.last_region, self.last_distance)
            stale = (now - self.last_spoken_at) >= self.reaffirm_sec
            if key != self.last_spoken_key or stale:
                first = self.last_spoken_key is None
                phrase = (
                    guidance_generator.first_seen_phrase(self.target, effective_info)
                    if first
                    else guidance_generator.guidance_phrase(self.target, effective_info)
                )
                self.last_spoken_key = key
                self.last_spoken_at = now
                # Leaving reach mode -> forget the prior reach cue so re-entering
                # later doesn't get suppressed by the throttle key.
                self.last_reach_key = None
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

    def _update_reach(self, best_det, hand_pose, w, h, now):
        """Decide the next reach-mode cue (called only when near + hand seen).

        Returns
        -------
        ("touch", phrase) - fingertip is inside the box. The loop must
                            fire FSM task_complete after speaking the phrase.
                            Only emitted once per session.
        ("reach", phrase) - directional or "almost" cue, throttled.
        None              - same cue as last time and still within the
                            re-affirm window.

        Also sets ``self.pending_haptic`` when the reach state changes into
        "almost" or "touching" - the loop ships it as a vibration cue.
        """
        reach = reach_guidance.assess_reach(
            best_det.box, hand_pose.fingertip, w, h,
        )

        if reach.state == "touching":
            # Touch fires task_complete; only emit once even if the loop
            # ticks before the FSM transition cancels us.
            if self.last_reach_key == ("touching", "center"):
                return None
            self.last_reach_key = ("touching", "center")
            self.last_reach_at = now
            self.pending_haptic = "long"  # the "you got it" pulse
            return ("touch", reach_guidance.reach_phrase(self.target, reach))

        key = (reach.state, reach.direction)
        stale = (now - self.last_reach_at) >= REACH_REAFFIRM_SEC
        if key != self.last_reach_key or stale:
            # Transition INTO "almost" gets a short haptic pulse - a
            # non-audio cue that the user is on target and just needs to
            # push forward. Directional changes stay purely voice.
            entered_almost = (
                reach.state == "almost"
                and (self.last_reach_key is None or self.last_reach_key[0] != "almost")
            )
            if entered_almost:
                self.pending_haptic = "short"
            self.last_reach_key = key
            self.last_reach_at = now
            return ("reach", reach_guidance.reach_phrase(self.target, reach))
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


def _replace_info(info, region: str, distance: str):
    """Return a SpatialInfo copy with a smoothed region/distance override.

    Used by GuidanceTracker to feed guidance_generator the *debounced*
    bucket rather than the raw per-frame one, so phrases don't ping-pong
    when YOLO's box size jitters across a threshold.
    """
    return spatial_reasoning.SpatialInfo(
        region=region,
        distance=distance,
        cx_frac=info.cx_frac,
        area_frac=info.area_frac,
        apparent_frac=info.apparent_frac,
        label=info.label,
    )


def _detect_target(frame, target: str) -> list:
    """Blocking helper run in a thread: detect ``target`` instances in ``frame``."""
    return yolo_service.detect(
        frame, conf_threshold=CONF_THRESHOLD, target_labels=[target],
    )


async def _speak(session: "Session", text: str) -> None:
    """Synthesize ``text`` (in a thread) and push the MP3 to the client.

    Passes ``text`` to send_tts so the "repeat that" voice command can replay
    any guidance line.
    """
    if not text:
        return
    loop = asyncio.get_running_loop()
    try:
        mp3 = await loop.run_in_executor(None, tts_service.synthesize, text)
    except Exception:
        log.exception("Session %s: guidance TTS failed for %r", session.id, text)
        return
    await session.send_tts(mp3, text=text)
    log.info("Session %s: OA guidance: %r", session.id, text)


async def _run(session: "Session", target: str) -> None:
    """The detection + guidance loop. Runs until cancelled or it times out."""
    loop = asyncio.get_running_loop()
    tracker = GuidanceTracker(target)

    try:
        while True:
            now = time.monotonic()

            frame = session.latest_frame
            frame_at = session.latest_frame_at
            if (frame is not None and frame_at is not None
                    and (time.time() - frame_at) > FRAME_STALE_SEC):
                frame = None   # feed frozen -> no observation this tick
            detections: list = []
            if frame is not None and getattr(frame, "size", 0):
                try:
                    detections = await loop.run_in_executor(
                        None, _detect_target, frame, target,
                    )
                except Exception:
                    log.exception("Session %s: detection failed", session.id)
                    detections = []

            # Hand detection only when the tracker says we're in (or just left)
            # reach-distance. MediaPipe is cheap (~15 ms) but pointless when
            # the target is still across the room.
            hand_pose = None
            if tracker.should_check_hand() and frame is not None and getattr(frame, "size", 0):
                try:
                    hand_pose = await loop.run_in_executor(
                        None, hand_service.detect, frame,
                    )
                except Exception:
                    log.exception("Session %s: hand detection failed", session.id)
                    hand_pose = None

            frame_shape = frame.shape if frame is not None else None
            result = tracker.update(
                detections, frame_shape, now,
                hand_pose=hand_pose,
                motion_state=getattr(session, "motion_state", "still"),
            )

            # Ship any pending haptic pulse the tracker set this tick, before
            # the phrase - a short vibration a beat before the voice cue is
            # a nicer "you're close" signal than after.
            if tracker.pending_haptic is not None:
                try:
                    await session.send_haptic(tracker.pending_haptic)
                except Exception:
                    log.exception("Session %s: haptic send failed", session.id)
                tracker.pending_haptic = None

            if result is not None:
                action, phrase = result
                if phrase:
                    await _speak(session, phrase)
                if action == "touch":
                    # Auto-complete: fingertip is inside the target box. This
                    # is the only autonomous task exit on success - everything
                    # else waits for the user to say "got it".
                    log.info(
                        "Session %s: hand touched %r, auto-completing",
                        session.id, target,
                    )
                    session.fsm.handle_event("task_complete")
                    return
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
