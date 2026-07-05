"""
Per-connection session state.

A Session is created when a client opens a WebSocket and lives until that
WebSocket closes. It holds:

- the WebSocket itself (for sending messages back)
- the FSM instance (authoritative state)
- the latest decoded frame as a numpy array (Sprint 2 will run YOLO on this)
- the latest task context dict (target object, destination, waypoints, ...)
- the connection start time (for diagnostics)
- a unique session id (for log readability)

The Session subscribes to FSM state changes and pushes ``fsm_state`` JSON
messages to the client whenever the FSM transitions.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Optional

import numpy as np
from fastapi import WebSocket

from fsm.task_fsm import TaskFSM, FSMState
from api.router import Router
from services import object_allocation, navigation

log = logging.getLogger("lumen.session")

# Wire-format names for FSM states. The internal enum names are PascalCase
# (Idle, ListeningForCommand, ...). The protocol uses snake_case strings
# from docs/protocol.md.
_STATE_WIRE_NAMES: dict[FSMState, str] = {
    FSMState.IDLE: "idle",
    FSMState.LISTENING: "listening",
    FSMState.OBJECT_ACTIVE: "object_active",
    FSMState.NAV_ACTIVE: "nav_active",
    FSMState.RETURNING: "returning",
}


# ---------- resume pool (session persistence across reconnects) ----------
#
# When a WS drops mid-task, the client's next connect can send a `resume`
# message with the previous session id. If we still have the task target
# saved here, we re-fire the task on the new session and announce it.
# TTL keeps the pool bounded and stale ids from surprising anyone.

RESUME_TTL_SEC = 5 * 60

_resume_pool: dict[str, dict] = {}


def _prune_resume_pool() -> None:
    now = time.time()
    stale = [sid for sid, s in _resume_pool.items() if s["expires_at"] < now]
    for sid in stale:
        del _resume_pool[sid]


def save_task_for_resume(session_id: str, task_type: str, target: str) -> None:
    """Called on graceful disconnect if a task was active."""
    _prune_resume_pool()
    _resume_pool[session_id] = {
        "task_type": task_type,
        "target": target,
        "expires_at": time.time() + RESUME_TTL_SEC,
    }
    log.info("Saved resume state for %s: %s/%s", session_id, task_type, target)


def take_task_for_resume(session_id: str) -> Optional[dict]:
    """Called on a `resume` message. Pops and returns the saved state,
    or None if there's no live entry for this id."""
    _prune_resume_pool()
    return _resume_pool.pop(session_id, None)


class Session:
    """One Session per WebSocket connection."""

    def __init__(self, ws: WebSocket) -> None:
        self.id: str = uuid.uuid4().hex[:8]
        self.ws: WebSocket = ws
        self.connected_at: float = time.time()

        # Per-task scratch state
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_frame_at: Optional[float] = None
        self.task_context: dict[str, Any] = {}

        # Handle to the running async task loop for the active task (e.g. the
        # Object Allocation detection loop). object_allocation.start/stop own
        # this; cleanup() cancels it on disconnect.
        self.detection_task: Optional[asyncio.Task] = None

        # Last spoken TTS phrase, for the voice "repeat that" command. Updated
        # whenever anything calls send_tts(mp3, text="...").
        self.last_spoken_text: str = ""

        # Client-provided session id, if any. Set by the "resume" message
        # handler. When we later save state on disconnect, we key it by this
        # id (falling back to self.id) so the client's next WS connect can
        # find it. Enables cross-connection continuity if the phone drops
        # Wi-Fi mid-task.
        self.client_session_id: Optional[str] = None

        # Coarse motion state reported by the phone's DeviceMotionEvent:
        # "still" | "moving" | "walking". Used by GuidanceTracker to bypass
        # the spatial debounce during real user motion (so bucket changes
        # aren't hidden behind the smoothing when the user actually walked).
        self.motion_state: str = "still"

        # Phone compass heading in degrees (0-360, clockwise), reported by the
        # client's DeviceOrientationEvent. None when the device has no compass
        # (laptop) or permission was denied. The navigation task uses it to
        # track the 360 room scan and to anchor door/indicator bearings; every
        # other task ignores it.
        self.heading: Optional[float] = None

        # FSM owned by this session
        self.fsm: TaskFSM = TaskFSM()
        self.fsm.subscribe(self._on_fsm_change)

        # Router knows how to dispatch incoming messages by type
        self.router: Router = Router(self)

        # Lock to serialize sends (FastAPI WebSocket isn't thread-safe across
        # concurrent send_* calls)
        self._send_lock: asyncio.Lock = asyncio.Lock()

    # ---------- lifecycle ----------

    async def send_initial_state(self) -> None:
        """Send the current FSM state to the client right after connect."""
        await self.send_fsm_state(self.fsm.state)
        log.debug("Session %s sent initial state", self.id)

    async def run(self) -> None:
        """Main receive loop. Returns when the WebSocket closes."""
        while True:
            msg = await self.ws.receive()
            mtype = msg.get("type")

            if mtype == "websocket.disconnect":
                log.info("Session %s: receive() got disconnect", self.id)
                return

            # FastAPI delivers either {"text": "..."} or {"bytes": b"..."}
            if "text" in msg and msg["text"] is not None:
                await self.router.dispatch_text(msg["text"])
            elif "bytes" in msg and msg["bytes"] is not None:
                await self.router.dispatch_binary(msg["bytes"])
            else:
                log.warning("Session %s: unexpected message shape: %s",
                            self.id, list(msg.keys()))

    async def cleanup(self) -> None:
        """Release any resources tied to this session.

        If a task was active at disconnect time, save it to the resume pool
        so the next WS connect can pick up where we left off.
        """
        # Save resume state BEFORE clearing task_context.
        active_type = self.task_context.get("task_type") if self.task_context else None
        if active_type == "object_allocation":
            target = self.task_context.get("target", "")
            if target:
                save_task_for_resume(self.id, "object_allocation", target)
        elif active_type == "navigation":
            dest = self.task_context.get("destination", "")
            if dest:
                save_task_for_resume(self.id, "navigation", dest)

        # Cancel any running task loop (Object Allocation detection, etc.).
        task = self.detection_task
        if task is not None and not task.done():
            task.cancel()
        self.detection_task = None
        self.latest_frame = None
        self.task_context.clear()

    # ---------- send helpers ----------

    async def send_json(self, payload: dict[str, Any]) -> None:
        """Send a JSON control message to the client."""
        async with self._send_lock:
            await self.ws.send_json(payload)

    async def send_binary(self, tag: int, data: bytes) -> None:
        """Send a tagged binary frame.

        The 1-byte ``tag`` is prepended to ``data`` per docs/protocol.md.
        Tag 0x03 = TTS MP3 (server → client).
        """
        if not 0 <= tag <= 0xFF:
            raise ValueError(f"binary tag out of range: {tag}")
        wire = bytes([tag]) + data
        async with self._send_lock:
            await self.ws.send_bytes(wire)

    async def send_fsm_state(self, state: FSMState) -> None:
        wire_name = _STATE_WIRE_NAMES.get(state, str(state))
        await self.send_json({"type": "fsm_state", "state": wire_name})

    async def send_transcription(self, text: str, confidence: float) -> None:
        await self.send_json({
            "type": "transcription",
            "text": text,
            "confidence": round(float(confidence), 3),
        })

    async def send_error(self, code: str, message: str) -> None:
        """Send an error JSON to the client. Client renders it visually; any
        voice feedback comes from the follow-up gTTS clip (or from browser
        speechSynthesis only for connection-level failures where we can't
        reach the client)."""
        await self.send_json({"type": "error", "code": code, "message": message})

    async def send_session_hello(self) -> None:
        """Tell the client its session id right after WS accept, so it can
        stash the id and send a ``resume`` message on the next connect."""
        await self.send_json({"type": "session_hello", "id": self.id})

    async def send_haptic(self, pattern: str) -> None:
        """Ask the client to vibrate. Pattern names are symbolic strings the
        client maps to a ``navigator.vibrate`` millisecond sequence:

            "short"   ~40 ms tap - enters reach "almost" zone
            "medium"  ~80 ms tap - generic acknowledgement
            "long"    ~200 ms  - fingertip touched the target box
            "success" double-tap - task_complete via voice "got it"
            "cancel"  single medium - task cancelled

        The client silently ignores unknown patterns and iOS Safari - which
        has no Vibration API at all - degrades to a no-op. Voice cues remain
        the primary channel; haptics are complementary."""
        await self.send_json({"type": "haptic", "pattern": pattern})

    async def send_tts(self, mp3_bytes: bytes, text: str = "") -> None:
        """Send a TTS MP3 clip to the client (binary tag 0x03).

        If ``text`` is passed we also record it as ``last_spoken_text`` so the
        voice "repeat that" command can replay it. Callers that don't care
        about repeat (e.g. non-linguistic clips) can omit it.
        """
        if text:
            self.last_spoken_text = text
        await self.send_binary(0x03, mp3_bytes)

    def _fire_cleanup_done(self) -> None:
        """Self-fire cleanup_done one tick after entering RETURNING."""
        try:
            self.fsm.handle_event("cleanup_done")
        except Exception:
            log.exception("Session %s: cleanup_done failed", self.id)

    # ---------- FSM subscriber ----------

    def _on_fsm_change(self, old: FSMState, new: FSMState) -> None:
        """FSM state changed - run side effects, then push to client.

        Called synchronously from inside fsm.handle_event(). We:
          1. fire exit hooks for the old state
          2. fire entry hooks for the new state
          3. schedule the fsm_state push onto the event loop
        """
        log.info("Session %s: FSM %s -> %s", self.id, old.name, new.name)

        # Exit hook for old state
        if old == FSMState.OBJECT_ACTIVE:
            object_allocation.stop(self)
        elif old == FSMState.NAV_ACTIVE:
            navigation.stop(self)

        # Entry hook for new state
        if new == FSMState.OBJECT_ACTIVE:
            target = self.task_context.get("target", "<unknown>")
            object_allocation.start(self, target)
        elif new == FSMState.NAV_ACTIVE:
            destination = self.task_context.get("destination", "<unknown>")
            navigation.start(self, destination)
        elif new == FSMState.RETURNING:
            # Sprint 1: clean up task context immediately. The cleanup_done
            # event must be deferred to the next loop tick so the
            # send_fsm_state(RETURNING) below fires before send_fsm_state(IDLE).
            self.task_context.clear()
        elif new == FSMState.LISTENING:
            # task_abort skips RETURNING and drops straight to LISTENING, so
            # we have to clear leftover task_context ourselves. Safe to clear
            # unconditionally - LISTENING is "no active task" by definition.
            self.task_context.clear()

        # Push state to client (async, scheduled on the loop)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            log.warning("Session %s: no running loop for FSM notify", self.id)
            return
        loop.create_task(self.send_fsm_state(new))

        # Defer cleanup_done so RETURNING is visible to the client first.
        if new == FSMState.RETURNING:
            loop.call_soon(self._fire_cleanup_done)
