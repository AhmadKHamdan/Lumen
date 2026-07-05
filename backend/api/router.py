"""
WebSocket message router.

Parses incoming messages by type and dispatches to handlers. Out-of-protocol
messages are logged and ignored (per docs/protocol.md).

Two channels share the same WebSocket:
- text frames carry JSON control messages
- binary frames carry tagged media payloads (1-byte tag + bytes)
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from api.frame_handler import handle_frame
from api.audio_handler import handle_command_audio

if TYPE_CHECKING:
    from api.session import Session

log = logging.getLogger("lumen.router")

# Binary frame tags (see docs/protocol.md)
TAG_FRAME = 0x01          # client → server: JPEG camera frame
TAG_COMMAND_AUDIO = 0x02  # client → server: WebM/Opus PTT recording
TAG_TTS = 0x03            # server → client: MP3 TTS clip (not received)


class Router:
    """Dispatches incoming messages on a single Session."""

    def __init__(self, session: "Session") -> None:
        self.session = session

    # ---------- text (JSON) ----------

    async def dispatch_text(self, text: str) -> None:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as e:
            log.warning("Session %s: bad JSON: %s", self.session.id, e)
            await self.session.send_error("protocol_violation",
                                          "Malformed JSON payload.")
            return

        if not isinstance(payload, dict):
            log.warning("Session %s: JSON not an object: %r",
                        self.session.id, payload)
            return

        mtype = payload.get("type")
        if mtype == "user_event":
            await self._handle_user_event(payload)
        elif mtype == "resume":
            await self._handle_resume(payload)
        elif mtype == "motion":
            # Phone accelerometer state: "still" | "moving" | "walking".
            # Guides the tracker's spatial debounce behaviour.
            state = payload.get("state", "still")
            if state in ("still", "moving", "walking"):
                self.session.motion_state = state
        else:
            log.info("Session %s: unknown JSON type %r (ignored)",
                     self.session.id, mtype)

    async def _handle_resume(self, payload: dict) -> None:
        """Restore the task from the previous connection, if it's still in
        the resume pool. Client sends {type:"resume", id:"<previous id>"}.

        We re-drive the FSM through user_start + command_recognized so the
        object_allocation detection loop restarts identically to a fresh
        request, and announce the resumption in voice.
        """
        old_id = payload.get("id")
        if not old_id:
            return
        # Local imports to avoid circular deps at module load.
        from api.session import take_task_for_resume
        from services import tts_service
        import time as _time

        self.session.client_session_id = str(old_id)
        state = take_task_for_resume(str(old_id))
        if state is None:
            log.info("Session %s: resume requested for %s but no live state",
                     self.session.id, old_id)
            return

        task_type = state["task_type"]
        target = state["target"]
        log.info("Session %s: resuming %s task for target=%r",
                 self.session.id, task_type, target)

        # Announce first (small nicety - user isn't left wondering what happened).
        resume_phrase = (
            f"Resuming search for your {target}."
            if task_type == "object_allocation"
            else f"Resuming navigation to the {target}."
        )
        try:
            mp3 = tts_service.synthesize(resume_phrase)
        except Exception:
            log.exception("Session %s: resume TTS failed", self.session.id)
        else:
            await self.session.send_tts(mp3, text=resume_phrase)

        # Drive FSM through IDLE -> LISTENING -> ACTIVE, mirroring the normal
        # command flow (audio_handler's happy path).
        self.session.fsm.handle_event("user_start")
        ctx_key = "target" if task_type == "object_allocation" else "destination"
        self.session.task_context = {
            "task_type": task_type,
            ctx_key: target,
            "started_at": _time.time(),
        }
        ok = self.session.fsm.handle_event(
            "command_recognized",
            payload={"task_type": task_type, "target": target},
        )
        if not ok:
            log.warning("Session %s: FSM refused resume transition",
                        self.session.id)
            self.session.task_context = {}

    async def _handle_user_event(self, payload: dict) -> None:
        event = payload.get("event")
        if event == "start":
            self.session.fsm.handle_event("user_start")
        elif event in ("stop", "cancel"):
            self.session.fsm.handle_event("user_stop")
        elif event == "confirm":
            # User signalled task completion via the UI (the voice path "got
            # it" routes through audio_handler instead). task_complete is only
            # valid from an active task; the FSM rejects it otherwise.
            accepted = self.session.fsm.handle_event("task_complete")
            log.info("Session %s: user confirm -> task_complete accepted=%s",
                     self.session.id, accepted)
        else:
            log.warning("Session %s: unknown user_event %r",
                        self.session.id, event)

    # ---------- binary (tagged) ----------

    async def dispatch_binary(self, data: bytes) -> None:
        if not data:
            log.warning("Session %s: empty binary frame", self.session.id)
            return
        tag = data[0]
        payload = data[1:]

        if tag == TAG_FRAME:
            await handle_frame(self.session, payload)
        elif tag == TAG_COMMAND_AUDIO:
            await handle_command_audio(self.session, payload)
        else:
            log.warning("Session %s: unknown binary tag 0x%02x (%d bytes ignored)",
                        self.session.id, tag, len(payload))
            await self.session.send_error(
                "protocol_violation",
                f"Unknown binary tag 0x{tag:02x}",
            )
