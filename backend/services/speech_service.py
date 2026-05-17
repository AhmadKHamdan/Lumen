"""
Speech-to-text service.

Wraps faster-whisper (the local ``base`` model, English only). Designed to be
standalone and importable - run from a Python REPL.

Why faster-whisper instead of openai-whisper?
- Same Whisper models under the hood (model weights from HuggingFace).
- Ships pre-built wheels for Python 3.13 (openai-whisper's setup.py breaks
  on 3.13 due to PEP 667; see requirements.txt comment).
- ~4x faster CPU inference via CTranslate2.
- No torch dependency (smaller install footprint).

Whisper expects 16 kHz mono float32 PCM input. The browser sends WebM/Opus,
so we decode it via PyAV - the FFmpeg Python bindings, which ship the FFmpeg
shared libraries inside their wheel. No external ``ffmpeg.exe`` on PATH
required.

The first call lazy-loads the model. faster-whisper downloads the model from
HuggingFace (~150 MB for ``base``) into the HF cache the first time. Subsequent
calls reuse the loaded model.

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
import threading

import av
import numpy as np

log = logging.getLogger("lumen.stt")

# Model size. ``base`` is ~150MB and runs in CPU-friendly time. ``tiny`` is
# faster but markedly less accurate on short utterances.
_MODEL_NAME = "base"

# CPU inference quantization. ``int8`` is the fastest CPU-friendly option;
# ``float16`` or ``float32`` are options if you need higher accuracy.
_COMPUTE_TYPE = "int8"

# Whisper always wants 16 kHz mono input.
_TARGET_SAMPLE_RATE = 16000

_model = None  # faster_whisper.WhisperModel - lazy-loaded
_model_lock = threading.Lock()


def _get_model():
    """Lazy-load the faster-whisper model on first call."""
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        log.info(
            "Loading faster-whisper model %r compute=%s "
            "(downloads ~150MB from HuggingFace on first run)",
            _MODEL_NAME, _COMPUTE_TYPE,
        )
        # Local import keeps module import cost low until the first transcribe call.
        from faster_whisper import WhisperModel
        _model = WhisperModel(_MODEL_NAME, device="cpu", compute_type=_COMPUTE_TYPE)
        log.info("faster-whisper model loaded")
    return _model


def _decode_to_pcm(audio_bytes: bytes) -> np.ndarray:
    """Decode an arbitrary audio blob to 16 kHz mono float32 PCM in [-1, 1].

    Uses PyAV (FFmpeg bindings, with FFmpeg shared libs bundled into the
    wheel). No external ``ffmpeg.exe`` is required.

    PyAV happily auto-detects WebM/Opus, OGG/Opus, MP3, WAV, M4A and friends
    from the bytes themselves, so we don't need a MIME-type hint.

    Returns an empty array if the blob is empty or has no audio stream.
    """
    if not audio_bytes:
        return np.empty(0, dtype=np.float32)

    log.debug("PyAV decode: %d bytes", len(audio_bytes))

    container = av.open(io.BytesIO(audio_bytes))
    try:
        audio_streams = [s for s in container.streams if s.type == "audio"]
        if not audio_streams:
            log.warning("PyAV: no audio stream found in blob")
            return np.empty(0, dtype=np.float32)
        audio_stream = audio_streams[0]

        # Resample to 16 kHz mono float32. PyAV format strings:
        #   "flt"  = AV_SAMPLE_FMT_FLT (float32 packed)
        #   "fltp" = AV_SAMPLE_FMT_FLTP (float32 planar)
        # Packed is fine for mono; the resulting ndarray comes out as (1, N).
        resampler = av.AudioResampler(
            format="flt",
            layout="mono",
            rate=_TARGET_SAMPLE_RATE,
        )

        chunks: list[np.ndarray] = []
        for frame in container.decode(audio_stream):
            for resampled in resampler.resample(frame):
                arr = resampled.to_ndarray()
                # to_ndarray returns shape (channels, samples) for mono packed.
                # Flatten regardless to be defensive.
                chunks.append(arr.flatten())

        # Flush any trailing samples held by the resampler.
        for resampled in resampler.resample(None):
            arr = resampled.to_ndarray()
            chunks.append(arr.flatten())
    finally:
        container.close()

    if not chunks:
        return np.empty(0, dtype=np.float32)
    pcm = np.concatenate(chunks).astype(np.float32, copy=False)
    return pcm


def _avg_logprob_to_confidence(avg_logprob: float) -> float:
    """Map Whisper's avg log-probability to a [0, 1] confidence proxy."""
    if avg_logprob is None or not math.isfinite(avg_logprob):
        return 0.0
    p = math.exp(avg_logprob)
    return max(0.0, min(1.0, p))


def transcribe(audio_bytes: bytes, mime_type: str = "") -> dict:
    """Transcribe ``audio_bytes`` to text + confidence proxy.

    Parameters
    ----------
    audio_bytes : bytes
        The raw audio blob (typically WebM/Opus from MediaRecorder).
    mime_type : str
        Kept for API compatibility. Ignored - PyAV auto-detects the format
        from the bytes.

    Returns
    -------
    dict
        ``{"text": str, "confidence": float}``. Empty / silent input returns
        ``{"text": "", "confidence": 0.0}``.
    """
    if not audio_bytes:
        return {"text": "", "confidence": 0.0}

    model = _get_model()

    try:
        pcm = _decode_to_pcm(audio_bytes)
    except av.AVError as e:
        log.warning("PyAV decode failed: %s", e)
        return {"text": "", "confidence": 0.0}

    if pcm.size == 0:
        return {"text": "", "confidence": 0.0}

    # faster-whisper accepts a 1-D float32 numpy array directly.
    # ``transcribe()`` returns (segments_generator, info). We have to
    # materialize the generator to actually run inference.
    segments_gen, _info = model.transcribe(
        pcm,
        language="en",
        beam_size=5,
        condition_on_previous_text=False,
        vad_filter=False,  # PTT chunks are already trimmed by the user's button release
    )
    segments = list(segments_gen)

    text = "".join(s.text for s in segments).strip()

    if segments:
        logprobs = [s.avg_logprob for s in segments if s.avg_logprob is not None]
        avg = sum(logprobs) / len(logprobs) if logprobs else float("-inf")
    else:
        avg = float("-inf")
    confidence = _avg_logprob_to_confidence(avg)

    return {"text": text, "confidence": confidence}
