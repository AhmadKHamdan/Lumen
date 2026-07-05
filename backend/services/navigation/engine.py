"""The navigation task engine: the async per-session loop.

When the FSM enters ``NavigationActive``, :func:`services.navigation.start` spawns
:func:`run` which, a few times per second:

    1. grabs the latest decoded camera frame off the session (+ compass heading),
    2. applies the motion gate (blurred frames are skipped, the user coached),
    3. runs the perception stack in a thread (COCO indicators, the 3-layer door
       funnel, door geometry, obstacle watchdog),
    4. feeds the result to the exploration controller (discover / face_target /
       go_indicator / go_door state machine), which decides what to say,
    5. speaks the guidance via server-side gTTS (same pipeline as every other task).

This is the integrated version of the webdemo prototype's ``server.py::detect``
endpoint: one HTTP-POST-per-frame became a server-driven loop over the session's
WebSocket frame stream, browser speechSynthesis became gTTS, and the module-global
state became a per-session :class:`~services.navigation.state.NavState`.

Speech policy (adapted from the webdemo frontend's queueing rules):
  - ``priority`` lines (scan summaries, arrival, obstacle warnings) are always
    spoken.
  - ambient lines (keep-scanning nudges, door countdowns) are dropped if they
    repeat the previous ambient line too soon, or if any line was spoken very
    recently — the browser plays MP3s back-to-back, so unthrottled ambient
    chatter would queue up and lag behind reality.

Exits:
  - arrival  -> speaks the arrival phrase, fires FSM ``task_complete``.
  - unknown destination -> speaks an apology, fires FSM ``user_stop``.
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
from .config import MOTION_COACH_FRAMES, MOTION_MAX
from .geometry import _track_turn
from .goals import indicators_for, resolve_goal
from .state import NavState

if TYPE_CHECKING:
    from api.session import Session

log = logging.getLogger("lumen.task.navigation")

DETECTION_HZ = 3                 # heavier models than Object Allocation's 5 Hz
_LOOP_INTERVAL = 1.0 / DETECTION_HZ

# Ambient-speech throttling (see module docstring).
AMBIENT_MIN_GAP_SEC = 2.5        # no ambient line within this of ANY spoken line
AMBIENT_REPEAT_SEC = 8.0         # identical ambient line at most this often


async def _speak(session: "Session", text: str) -> None:
    """Synthesize ``text`` (in a thread) and push the MP3 to the client."""
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


def _perceive(st: NavState, img: np.ndarray, heading: Optional[float],
              indicators: set) -> Optional[dict]:
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
                                                   dg.cur_frac, motion)
    obst_guidance, obst_priority, obst_blocking = obstacles.evaluate(
        st, obstacle_dets, img, w, h)

    return {
        "motion": motion, "w": w, "seen": seen, "obj_dets": obj_dets,
        "confirmed": confirmed, "dg": dg, "transit": transit, "just_near": just_near,
        "obst_guidance": obst_guidance, "obst_priority": obst_priority,
        "obst_blocking": obst_blocking,
    }


async def run(session: "Session", destination: str) -> None:
    """The navigation loop. Runs until arrival, cancellation, or abort."""
    loop = asyncio.get_running_loop()

    # 1. Resolve the spoken destination to a goal we have room indicators for.
    goal = resolve_goal(destination)
    if goal is None:
        log.info("Session %s: unknown navigation destination %r",
                 session.id, destination)
        await _speak(session,
                     f"I don't know how to find the {destination} yet. "
                     "Try the kitchen, bathroom, bedroom, living room, "
                     "office, or dining room.")
        session.fsm.handle_event("user_stop")
        return

    # 2. First navigation task in this server's lifetime loads the models
    # (~seconds, GPU init included). Tell the user rather than going silent.
    if not models.is_loaded():
        await _speak(session, "Give me a moment while I get ready.")
        try:
            await loop.run_in_executor(None, models.ensure_loaded)
        except Exception:
            log.exception("Session %s: navigation model load failed", session.id)
            await _speak(session, "Sorry, navigation isn't available right now.")
            session.fsm.handle_event("user_stop")
            return

    st = NavState()
    st["goal"] = goal
    primary, secondary = indicators_for(goal)
    indicators = set(primary) | set(secondary)

    last_spoken_at = 0.0
    last_ambient: tuple[str, float] = ("", 0.0)
    coached_at = 0.0

    try:
        while True:
            now = time.monotonic()

            frame = session.latest_frame
            if frame is None or not getattr(frame, "size", 0):
                await asyncio.sleep(_LOOP_INTERVAL)
                continue

            heading = session.heading

            # frame_handler stores RGB; the ported perception stack (and its tuned
            # thresholds) speak cv2's BGR. Convert once here.
            try:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                result = await loop.run_in_executor(
                    None, _perceive, st, bgr, heading, indicators)
            except Exception:
                log.exception("Session %s: navigation perception failed", session.id)
                await asyncio.sleep(_LOOP_INTERVAL)
                continue

            if result is None:
                # Motion gate skipped the frame; coach only after several
                # consecutive too-fast frames, and not more than once per gap.
                if (st["fast_frames"] >= MOTION_COACH_FRAMES
                        and now - coached_at >= AMBIENT_REPEAT_SEC):
                    coached_at = now
                    await _speak(session, "Slow down. Move the phone slowly.")
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
                obst_blocking=result["obst_blocking"])

            if announce_arrival:
                await _speak(session, phrase or f"We've reached the {goal}.")
                log.info("Session %s: navigation arrived at %r", session.id, goal)
                session.fsm.handle_event("task_complete")
                return

            if guidance:
                if priority:
                    await _speak(session, guidance)
                    last_spoken_at = now
                else:
                    # Ambient: drop repeats and anything hot on the heels of the
                    # previous line (MP3s queue client-side; don't build a backlog).
                    recent_same = (guidance == last_ambient[0]
                                   and now - last_ambient[1] < AMBIENT_REPEAT_SEC)
                    too_soon = now - last_spoken_at < AMBIENT_MIN_GAP_SEC
                    if not recent_same and not too_soon:
                        await _speak(session, guidance)
                        last_spoken_at = now
                        last_ambient = (guidance, now)

            await asyncio.sleep(_LOOP_INTERVAL)

    except asyncio.CancelledError:
        log.info("Session %s: navigation loop cancelled", session.id)
        raise
    except Exception:
        log.exception("Session %s: navigation loop crashed", session.id)
