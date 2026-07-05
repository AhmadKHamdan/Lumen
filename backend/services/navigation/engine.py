"""The navigation task engine: the async per-session loop.

When the FSM enters ``NavigationActive``, :func:`services.navigation.start` spawns
:func:`run` which, a few times per second:

    1. grabs the latest decoded camera frame off the session (+ compass heading),
    2. applies the motion gate (blurred frames are skipped, the user coached),
    3. runs the perception stack in a thread (COCO indicators, the 3-layer door
       funnel, door geometry, obstacle watchdog),
    4. feeds the result to the exploration controller (discover / face_target /
       go_indicator / go_door state machine), which decides what to say,
    5. hands the guidance to a per-session speech queue (server-side gTTS).

This is the integrated version of the webdemo prototype's ``server.py::detect``
endpoint: one HTTP-POST-per-frame became a server-driven loop over the session's
WebSocket frame stream, browser speechSynthesis became gTTS, and the module-global
state became a per-session :class:`~services.navigation.state.NavState`.

Timing: the controller's pacing constants are SECONDS, and each tick passes the
measured ``dt`` - clamped to two nominal ticks, because time during a stall (TTS
synthesis, slow inference) contains no frames and therefore is not evidence that
anything appeared or disappeared.

Speech (adapted from the webdemo frontend's speechSynthesis queueing rules):
  - a dedicated :class:`_Speech` worker synthesizes and sends in the background,
    so gTTS latency NEVER stalls the perception loop;
  - ``priority`` lines (scan summaries, arrival, obstacle warnings) always queue;
  - ambient lines (keep-scanning nudges, door countdowns) are dropped when the
    pipe is busy, when they repeat too soon, or hot on the heels of another line
    - the client plays MP3s back-to-back, so ambient chatter must never queue up
    and lag behind reality.

Exits:
  - arrival        -> flushes speech, speaks the arrival phrase, fires FSM
                      ``task_complete``.
  - journey timeout (NAV_TIMEOUT_SEC) -> speaks, fires ``user_stop``.
  - unknown destination -> speaks an apology, fires ``user_stop``.
  - cancellation (user "stop", disconnect) -> the task is cancelled by
    :func:`services.navigation.stop` via the FSM exit hook.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

from services import tts_service
from . import controller, models, obstacles, perception
from .config import MOTION_COACH_FRAMES, MOTION_MAX, NAV_TIMEOUT_SEC
from .geometry import _track_turn
from .goals import indicators_for, resolve_goal
from .state import NavState

if TYPE_CHECKING:
    from api.session import Session

log = logging.getLogger("lumen.task.navigation")

DETECTION_HZ = 3                 # heavier models than Object Allocation's 5 Hz
_LOOP_INTERVAL = 1.0 / DETECTION_HZ
# dt clamp: a stalled tick must not fast-forward the controller's timers - time
# without observed frames is not evidence of absence (a "door gone for 2 s" timer
# needs 2 s of FRAMES without the door, not 2 s of wall clock).
_DT_MAX = 2.0 * _LOOP_INTERVAL

# Ambient-speech throttling (see module docstring).
AMBIENT_MIN_GAP_SEC = 2.5        # no ambient line within this of ANY queued line
AMBIENT_REPEAT_SEC = 8.0         # identical ambient line at most this often

# Frozen-feed guard: if the phone stops sending frames (screen lock, backgrounded
# tab, dropped wifi), latest_frame goes stale. Re-analysing the same image would
# read motion~0 and keep ACCUMULATING indicator evidence from a frozen picture -
# a stale fridge frame could confirm an arrival while the user stands in a
# hallway. A frame older than this, or one we already processed, is skipped and
# the controller's timers stay frozen (no frames = no evidence, either way).
FRAME_STALE_SEC = 2.0


async def _speak_now(session: "Session", text: str) -> None:
    """Direct, awaited speech - only for lines OUTSIDE the perception loop
    (task start/end), where blocking is fine and ordering must be exact."""
    if not text:
        return
    loop = asyncio.get_running_loop()
    try:
        mp3 = await loop.run_in_executor(None, tts_service.synthesize, text)
    except Exception:
        log.exception("Session %s: nav TTS failed for %r", session.id, text)
        return
    await session.send_tts(mp3, text=text)
    log.info("Session %s: NAV guidance: %r", session.id, text)


class _Speech:
    """Per-session speech pipe: an asyncio.Queue drained by one worker task, so
    the perception loop enqueues and moves on. Ambient lines are gated at the
    enqueue side (busy pipe / repeats / min-gap -> dropped, never queued)."""

    def __init__(self, session: "Session") -> None:
        self.session = session
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.busy = False                 # worker is synthesizing/sending now
        self.last_enqueued_at = 0.0
        self.last_ambient: tuple[str, float] = ("", 0.0)
        self._task = asyncio.get_running_loop().create_task(self._run())

    def say_priority(self, text: str, now: float) -> None:
        """Priority lines must be spoken - always enqueue."""
        self.last_enqueued_at = now
        self.queue.put_nowait(text)

    def say_ambient(self, text: str, now: float) -> bool:
        """Ambient lines are droppable by design. Returns True if enqueued."""
        if self.busy or not self.queue.empty():
            return False                  # pipe busy -> ambient never queues
        if (text == self.last_ambient[0]
                and now - self.last_ambient[1] < AMBIENT_REPEAT_SEC):
            return False                  # same nudge too soon
        if now - self.last_enqueued_at < AMBIENT_MIN_GAP_SEC:
            return False                  # hot on the heels of another line
        self.last_ambient = (text, now)
        self.last_enqueued_at = now
        self.queue.put_nowait(text)
        return True

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            text = await self.queue.get()
            self.busy = True
            try:
                mp3 = await loop.run_in_executor(
                    None, tts_service.synthesize, text)
                await self.session.send_tts(mp3, text=text)
                log.info("Session %s: NAV guidance: %r", self.session.id, text)
            except Exception:
                log.exception("Session %s: nav TTS failed for %r",
                              self.session.id, text)
            finally:
                self.busy = False

    async def flush(self, timeout: float = 6.0) -> None:
        """Wait (bounded) for queued lines to finish - used before a terminal
        line so the arrival phrase doesn't overtake its predecessors."""
        deadline = time.monotonic() + timeout
        while (not self.queue.empty() or self.busy) and time.monotonic() < deadline:
            await asyncio.sleep(0.1)

    def close(self) -> None:
        self._task.cancel()


def _perceive(st: NavState, img: np.ndarray, heading: Optional[float],
              indicators: set, dt: float) -> Optional[dict]:
    """Blocking per-frame perception, run in a worker thread.

    Returns None when the motion gate skipped the frame (with ``st`` updated), or a
    dict with everything the controller needs. Mirrors the webdemo's ``detect()``
    pipeline: motion gate -> objects -> door funnel -> door geometry -> transit ->
    obstacles.
    """
    h, w = img.shape[:2]

    # --- Motion gate: if the phone is panning too fast the frame is blurred and
    # detections would be unreliable, so skip detection and coach the user to slow
    # down. Also avoids wasting a (latency-costly) inference on a bad frame. ---
    small = cv2.cvtColor(cv2.resize(img, (64, 48)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    motion = float(np.abs(small - st.prev_small).mean()) if st.prev_small is not None else 0.0
    st.prev_small = small
    if motion > MOTION_MAX:
        # Keep the turn total advancing even though we skip detection on this
        # blurred frame, so a fast segment doesn't stall the full-circle completion.
        if st["mode"] == "discover":
            _track_turn(st, heading)
        # One blurred frame (autofocus hunt, exposure change) is not the user moving
        # fast — skip it silently, and only COACH after several consecutive ones.
        st["fast_frames"] += 1
        return None
    st["fast_frames"] = 0

    # Perception: one frame -> trustworthy detections.
    obj_dets, obstacle_dets = perception.perceive_objects(st, img, w, h, indicators)
    obj_dets, door_dets, near_box, vdoor = perception.process_doors(st, img, w, h, obj_dets)
    seen = {cls for cls, _conf, _xy in obj_dets}

    # Interpretation: arrival evidence, door geometry, transit, obstacles.
    confirmed = controller.accumulate_indicator_evidence(st, seen, indicators)
    dg = perception.door_geometry(st, door_dets, vdoor, w, h)
    transit, just_near = controller.detect_transit(st, near_box, dg.door_confirmed,
                                                   dg.cur_frac, motion, dt)
    obst_guidance, obst_priority, obst_blocking = obstacles.evaluate(
        st, obstacle_dets, img, w, h, dt)

    return {
        "motion": motion, "w": w, "seen": seen, "obj_dets": obj_dets,
        "confirmed": confirmed, "dg": dg, "transit": transit, "just_near": just_near,
        "obst_guidance": obst_guidance, "obst_priority": obst_priority,
        "obst_blocking": obst_blocking,
    }


async def run(session: "Session", destination: str) -> None:
    """The navigation loop. Runs until arrival, timeout, cancellation, or abort."""
    loop = asyncio.get_running_loop()

    # 1. Resolve the spoken destination to a goal we have room indicators for.
    goal = resolve_goal(destination)
    if goal is None:
        log.info("Session %s: unknown navigation destination %r",
                 session.id, destination)
        await _speak_now(session,
                         f"I don't know how to find the {destination} yet. "
                         "Try the kitchen, bathroom, bedroom, living room, "
                         "office, or dining room.")
        session.fsm.handle_event("user_stop")
        return

    # 2. First navigation task in this server's lifetime loads the models
    # (~seconds, GPU init included). Tell the user rather than going silent.
    if not models.is_loaded():
        await _speak_now(session, "Give me a moment while I get ready.")
        try:
            await loop.run_in_executor(None, models.ensure_loaded)
        except Exception:
            log.exception("Session %s: navigation model load failed", session.id)
            await _speak_now(session, "Sorry, navigation isn't available right now.")
            session.fsm.handle_event("user_stop")
            return

    st = NavState()
    st["goal"] = goal
    primary, secondary = indicators_for(goal)
    indicators = set(primary) | set(secondary)

    speech = _Speech(session)
    journey_start = time.monotonic()
    last_tick = journey_start
    processed_frame_at: Optional[float] = None   # wall-clock stamp of last frame used

    try:
        while True:
            now = time.monotonic()
            dt = min(now - last_tick, _DT_MAX)
            last_tick = now

            # Whole-journey belt-and-braces stop (the ROOM_CAP check-in is the
            # polite per-room version; this catches a journey stuck in ONE room).
            if now - journey_start >= NAV_TIMEOUT_SEC:
                log.info("Session %s: navigation timeout after %.0fs",
                         session.id, now - journey_start)
                await speech.flush()
                await _speak_now(session,
                                 f"We've been looking for the {goal} for a while "
                                 "without luck, so I'll stop here. Say navigate "
                                 f"to the {goal} anytime to try again.")
                session.fsm.handle_event("user_stop")
                return

            frame = session.latest_frame
            frame_at = session.latest_frame_at   # time.time(), set by frame_handler
            if (frame is None or not getattr(frame, "size", 0)
                    or frame_at is None
                    or frame_at == processed_frame_at          # no NEW frame yet
                    or (time.time() - frame_at) > FRAME_STALE_SEC):  # feed frozen
                await asyncio.sleep(_LOOP_INTERVAL)
                continue
            processed_frame_at = frame_at

            heading = session.heading

            # frame_handler stores RGB; the ported perception stack (and its tuned
            # thresholds) speak cv2's BGR. Convert once here.
            try:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                result = await loop.run_in_executor(
                    None, _perceive, st, bgr, heading, indicators, dt)
            except Exception:
                log.exception("Session %s: navigation perception failed", session.id)
                await asyncio.sleep(_LOOP_INTERVAL)
                continue

            if result is None:
                # Motion gate skipped the frame; coach only after several
                # consecutive too-fast frames (repeat-gating is say_ambient's job).
                if st["fast_frames"] >= MOTION_COACH_FRAMES:
                    speech.say_ambient("Slow down. Move the phone slowly.", now)
                await asyncio.sleep(_LOOP_INTERVAL)
                continue

            dg = result["dg"]
            guidance, priority, announce_arrival, phrase, _matched = controller.step(
                st, goal=goal, heading=heading, motion=result["motion"],
                w=result["w"], seen=result["seen"], indicators=indicators,
                obj_dets=result["obj_dets"], door_confirmed=dg.door_confirmed,
                door_cx_frac=dg.door_cx_frac, door_corro=dg.corro,
                region=dg.region, door_dist=dg.door_dist,
                transit=result["transit"], just_near=result["just_near"],
                confirmed=result["confirmed"],
                obst_guidance=result["obst_guidance"],
                obst_priority=result["obst_priority"],
                obst_blocking=result["obst_blocking"], dt=dt)

            if announce_arrival:
                # Terminal: let queued lines drain, then speak the arrival
                # directly (task_complete tears this task down via the FSM).
                await speech.flush()
                await _speak_now(session, phrase or f"We've reached the {goal}.")
                log.info("Session %s: navigation arrived at %r", session.id, goal)
                session.fsm.handle_event("task_complete")
                return

            if guidance:
                if priority:
                    speech.say_priority(guidance, now)
                else:
                    speech.say_ambient(guidance, now)

            await asyncio.sleep(_LOOP_INTERVAL)

    except asyncio.CancelledError:
        log.info("Session %s: navigation loop cancelled", session.id)
        raise
    except Exception:
        log.exception("Session %s: navigation loop crashed", session.id)
    finally:
        speech.close()
