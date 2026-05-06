"""
Text-to-speech service.

Wraps gTTS (Google TTS) and returns MP3 bytes ready to be sent over the
WebSocket as a binary `tts` message.

We cache by exact text key in a bounded LRU. Most of our TTS strings are a
small set of templates ("looking for your cup", "navigating to the kitchen",
"I didn't catch that, please repeat", ...) so caching avoids redundant
network calls to Google TTS.

Designed to be standalone and importable - run from a Python REPL with no
FastAPI involvement so it can be developed and tested in isolation.
"""
from __future__ import annotations

import io
import logging
from collections import OrderedDict
from threading import Lock

from gtts import gTTS

log = logging.getLogger("lumen.tts")

# Bounded LRU cache. Most callers hit ~10 distinct strings, so 50 is plenty
# and prevents unbounded growth if dynamic phrases are passed in.
_CACHE_MAX_ENTRIES = 50

_cache: OrderedDict[str, bytes] = OrderedDict()
_cache_lock = Lock()


def synthesize(text: str, *, lang: str = "en") -> bytes:
    """Synthesize ``text`` to MP3 bytes via gTTS.

    Cache hits are returned immediately. Cache misses call gTTS (which makes
    an HTTPS request to translate.google.com) and store the result.

    Raises whatever gTTS raises on network/auth/empty-text errors. Callers
    should catch broadly because gTTS exceptions vary by version.
    """
    if not text or not text.strip():
        raise ValueError("synthesize() called with empty text")

    key = f"{lang}::{text}"

    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            log.debug("TTS cache hit: %r", text)
            return _cache[key]

    log.info("TTS synthesize: %r (lang=%s)", text, lang)
    buf = io.BytesIO()
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.write_to_fp(buf)
    except Exception:
        log.exception("gTTS failed for %r", text)
        raise
    mp3 = buf.getvalue()
    if not mp3:
        raise RuntimeError(f"gTTS returned empty bytes for {text!r}")

    with _cache_lock:
        _cache[key] = mp3
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_MAX_ENTRIES:
            _cache.popitem(last=False)

    return mp3


def cache_stats() -> dict:
    """For diagnostics / tests."""
    with _cache_lock:
        return {"entries": len(_cache), "max": _CACHE_MAX_ENTRIES}


def clear_cache() -> None:
    """Test helper."""
    with _cache_lock:
        _cache.clear()
