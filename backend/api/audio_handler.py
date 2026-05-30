"""
Push-to-talk audio handler.

Receives binary `command_audio` messages (WebM/Opus blobs from the browser's
MediaRecorder), runs them through:

    speech_service.transcribe()  →  command_parser.parse()  →  FSM event
                                                                      ↓
                              tts_service.synthesize()  ←  confirmation phrase
                                          ↓
                                   send_tts() back to client

The full integration glue is in this file (matches Sprint 1's "voice round-trip"
goal). The blob is also saved to backend/captured_audio/<timestamp>.webm for
offline inspection.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from fsm.task_fsm import FSMState
from services import command_parser, guidance_generator, speech_service, tts_service

if TYPE_CHECKING:
    from api.session import Session

log = logging.getLogger("lumen.audio")

CAPTURED_AUDIO_DIR = Path(__file__).resolve().parent.parent / "captured_audio"
CAPTURED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def _save_for_inspection(session_id: str, blob: bytes) -> Path:
    """Drop the raw WebM/Opus blob to disk for debugging.

    Returns the path written. Filename includes session id and a millisecond
    timestamp so concurrent uploads from the same session don't collide.
    """
    ts = int(time.time() * 1000)
    path = CAPTURED_AUDIO_DIR / f"{ts}_{session_id}.webm"
    path.write_bytes(blob)
    return path


async def handle_command_audio(session: "Session", blob: bytes) -> None:
    """Handle a PTT recording end-to-end.

    Sprint 1 contract:
      * save blob to disk (debug aid)
      * transcribe via Whisper
      * push transcription JSON to client
      * parse intent
      * drive FSM into the right active state (or stay listening on unknown)
      * synthesize the confirmation phrase via gTTS and send back as MP3
    """
    if not blob:
        log.warning("Session %s: empty command_audio payload", session.id)
        return

    saved = _save_for_inspection(session.id, blob)
    log.info("Session %s: saved PTT audio (%d bytes) -> %s",
             session.id, len(blob), saved.name)

    # 1. Transcribe
    try:
        result = speech_service.transcribe(blob, "audio/webm;codecs=opus")
    except Exception:
        log.exception("Session %s: transcription failed", session.id)
        await session.send_error("transcription_failed",
                                 "Couldn't understand audio. Please try again.")
        # Synthesize a clarification phrase too so the user hears feedback.
        try:
            mp3 = tts_service.synthesize("Sorry, I didn't catch that. Please try again.")
            await session.send_tts(mp3)
        except Exception:
            log.exception("Session %s: clarification TTS also failed", session.id)
        return

    text = (result.get("text") or "").strip()
    confidence = float(result.get("confidence", 0.0))
    log.info("Session %s: transcribed %.2f conf: %r",
             session.id, confidence, text)
    await session.send_transcription(text, confidence)

    if not text:
        await session.send_error("transcription_failed",
                                 "Heard silence; please try again.")
        return

    # 2. Parse intent
    intent = command_parser.parse(text)
    log.info("Session %s: parsed intent: %s", session.id, intent)

    # 3. Drive FSM and pick a confirmation phrase
    if intent["task_type"] == "completion":
        # A confirm/cancel command said during (or just before) a task. This
        # path handles its own TTS and returns.
        await _handle_completion(session, intent)
        return

    if intent["task_type"] == "object_allocation":
        # Populate task context BEFORE firing the event: the FSM entry hook
        # (Session._on_fsm_change -> object_allocation.start) reads the target
        # off task_context synchronously inside handle_event().
        session.task_context = {
            "task_type": "object_allocation",
            "target": intent["target"],
            "started_at": time.time(),
        }
        ok = session.fsm.handle_event("command_recognized", payload=intent)
        if not ok:
            log.warning("Session %s: FSM rejected object_allocation from %s",
                        session.id, session.fsm.state.name)
            session.task_context = {}
            await session.send_error("protocol_violation",
                                     "Not ready to start a task yet.")
            return
        confirm_text = command_parser.confirmation_phrase(intent)
    elif intent["task_type"] == "navigation":
        session.task_context = {
            "task_type": "navigation",
            "destination": intent["target"],
            "started_at": time.time(),
        }
        ok = session.fsm.handle_event("command_recognized", payload=intent)
        if not ok:
            log.warning("Session %s: FSM rejected navigation from %s",
                        session.id, session.fsm.state.name)
            session.task_context = {}
            await session.send_error("protocol_violation",
                                     "Not ready to start a task yet.")
            return
        confirm_text = command_parser.confirmation_phrase(intent)
    else:
        # Unknown intent: stay in ListeningForCommand (or whatever current
        # state) and prompt the user to repeat.
        await session.send_error("parse_unknown",
                                 "I didn't catch that, please repeat.")
        confirm_text = command_parser.confirmation_phrase(intent)

    # 4. Synthesize and send confirmation
    try:
        mp3 = tts_service.synthesize(confirm_text)
    except Exception:
        log.exception("Session %s: TTS synthesis failed for %r",
                      session.id, confirm_text)
        return
    await session.send_tts(mp3)
    log.info("Session %s: sent TTS (%d bytes) for %r",
             session.id, len(mp3), confirm_text)


async def _handle_completion(session: "Session", intent: dict) -> None:
    """Handle a confirm/cancel voice command spoken during a task.

    * "confirm" (e.g. "got it") while a task is active -> task_complete, then
      voice a closing phrase naming the object.
    * "cancel" (e.g. "never mind") while listening or active -> user_stop.
    * Either one with nothing to act on -> a gentle "no active task" reply.

    The object name is read from task_context *before* firing the FSM event,
    because the RETURNING transition clears task_context synchronously.
    """
    target_word = intent.get("target")  # "confirm" | "cancel"
    state = session.fsm.state
    active = state in (FSMState.OBJECT_ACTIVE, FSMState.NAV_ACTIVE)

    # Best-effort object/destination label for a natural closing phrase.
    obj = (
        session.task_context.get("target")
        or session.task_context.get("destination")
        or "task"
    )

    if target_word == "confirm":
        if active:
            session.fsm.handle_event("task_complete")
            phrase = guidance_generator.complete_phrase(obj)
            log.info("Session %s: user confirmed completion of %r", session.id, obj)
        else:
            await session.send_error("protocol_violation",
                                     "No active task to complete.")
            phrase = "There's no active task right now."
    else:  # cancel
        if active:
            # Voice cancel during a running task: abort the task but STAY in
            # the session (LISTENING), so the user can immediately speak a
            # new command. user_stop would dump us to IDLE, after which the
            # next command_recognized gets rejected and the user has to find
            # the Start button - bad UX for a blind user mid-flow.
            session.fsm.handle_event("task_abort")
            phrase = guidance_generator.cancel_phrase(obj)
            log.info("Session %s: user aborted task -> LISTENING", session.id)
        elif state == FSMState.LISTENING:
            # Cancel said while we were waiting for a command: end the session.
            session.fsm.handle_event("user_stop")
            phrase = guidance_generator.cancel_phrase(obj)
            log.info("Session %s: user cancelled while listening", session.id)
        else:
            phrase = "Okay."

    try:
        mp3 = tts_service.synthesize(phrase)
    except Exception:
        log.exception("Session %s: completion TTS failed for %r",
                      session.id, phrase)
        return
    await session.send_tts(mp3)
    log.info("Session %s: sent completion TTS (%d bytes) for %r",
             session.id, len(mp3), phrase)
