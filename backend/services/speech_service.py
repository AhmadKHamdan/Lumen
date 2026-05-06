"""
Speech-to-text service.

Wraps OpenAI Whisper (the local ``whisper-base`` model, English only).
Designed to be standalone and importable - run from a Python REPL.

Whisper expects 16 kHz mono WAV input. The browser sends WebM/Opus, so we
decode it via pydub (which shells out to ffmpeg). ffmpeg must be on PATH.

The first call lazy-loads the model (~140MB download on first run, cached
to ``~/.cache/whisper/``). Subsequent calls reuse the loaded model.

Returns a dict ``{"text": str, "confidence": float}`` where confidence is a
heuristic in [0, 1] derived from Whisper's average log-probability:

    confidence = exp(avg_logprob) clamped to [0, 1]

This is a rough proxy, not a calibrated probability. Don't gate critical
logic on a hard threshold.
"""
from __future__ import annotations

import io
import logging
import math
import tempfile
import threading
from pathlib import Path

import numpy as np
from pydub import AudioSegment

log = logging.getLogger("lumen.stt")

# Model name. ``base`` is ~140MB and runs in CPU-friendly time. ``tiny`` is
# faster but markedly less accurate on short utterances.
_MODEL_NAME = "base"

_model = None  # whisper.model.Whisper - lazy-loaded
_model_lock = threading.Lock()


def _get_model():
    """Lazy-load the Whisper model on first call."""
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        log.info("Loading Whisper model %r (this may take a while on first run)", _MODEL_NAME)
        import whisper  # local import - ~1s import cost
        _model = whisper.load_model(_MODEL_NAME)
        log.info("Whisper model loaded")
    return _model


def _decode_to_pcm(audio_bytes: bytes, mime_type: str) -> np.ndarray:
    """Decode an arbitrary audio blob to 16 kHz mono float32 PCM in [-1, 1].

    pydub handles WebM/Opus, MP3, WAV, etc. via ffmpeg.
    """
    fmt_hint = _format_hint_from_mime(mime_type)
    log.debug("Decoding %d bytes as fmt=%s", len(audio_bytes), fmt_hint)

    seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt_hint)
    seg = seg.set_frame_rate(16000).set_channels(1).set_sample_width(2)  # 16-bit PCM

    # Convert to float32 in [-1, 1]
    samples = np.array(seg.get_array_of_samples(), dtype=np.int16)
    pcm = samples.astype(np.float32) / 32768.0
    return pcm


def _format_hint_from_mime(mime_type: str) -> str:
    """Guess the pydub format hint from the MIME type.

    pydub accepts: 'webm', 'ogg', 'mp3', 'wav', 'm4a', etc.
    """
    mime = (mime_type or "").lower()
    if "webm" in mime:
        return "webm"
    if "ogg" in mime or "opus" in mime:
        return "ogg"
    if "mp3" in mime or "mpeg" in mime:
        return "mp3"
    if "wav" in mime or "wave" in mime:
        return "wav"
    if "m4a" in mime or "mp4" in mime:
        return "m4a"
    # Default - let ffmpeg sniff it
    return "webm"


def _avg_logprob_to_confidence(avg_logprob: float) -> float:
    """Map Whisper's avg log-probability to a [0, 1] confidence proxy."""
    if avg_logprob is None or not math.isfinite(avg_logprob):
        return 0.0
    p = math.exp(avg_logprob)
    return max(0.0, min(1.0, p))


def transcribe(audio_bytes: bytes, mime_type: str) -> dict:
    """Transcribe ``audio_bytes`` to text + confidence proxy.

    Parameters
    ----------
    audio_bytes : bytes
        The raw audio blob (typically WebM/Opus from MediaRecorder).
    mime_type : str
        e.g. ``"audio/webm;codecs=opus"`` (used as a hint for the decoder).

    Returns
    -------
    dict
        ``{"text": str, "confidence": float}``. Empty / silent input returns
        ``{"text": "", "confidence": 0.0}``.
    """
    if not audio_bytes:
        return {"text": "", "confidence": 0.0}

    model = _get_model()

    pcm = _decode_to_pcm(audio_bytes, mime_type)
    if pcm.size == 0:
        return {"text": "", "confidence": 0.0}

    # Whisper accepts a 1-D float32 numpy array directly.
    result = model.transcribe(
        pcm,
        language="en",
        fp16=False,                # CPU-friendly default
        condition_on_previous_text=False,
    )

    text = (result.get("text") or "").strip()
    # Aggregate avg_logprob across segments (Whisper returns a list).
    segments = result.get("segments") or []
    if segments:
        logprobs = [s.get("avg_logprob") for s in segments
                    if s.get("avg_logprob") is not None]
        if logprobs:
            avg = sum(logprobs) / len(logprobs)
        else:
            avg = float("-inf")
    else:
        avg = float("-inf")
    confidence = _avg_logprob_to_confidence(avg)

    return {"text": text, "confidence": confidence}
