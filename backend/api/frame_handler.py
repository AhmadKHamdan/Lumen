"""
Frame ingestion handler.

Receives binary `frame` messages, decodes the JPEG to a numpy array via
Pillow, and stores it on the session for downstream consumers (Sprint 2 will
add YOLO inference here).

For Sprint 1 we just log the decode time and frame size periodically so we
can verify frames are flowing.
"""
from __future__ import annotations

import io
import logging
import time
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, UnidentifiedImageError

if TYPE_CHECKING:
    from api.session import Session

log = logging.getLogger("lumen.frame")

# Throttle log spam: with 5 FPS we'd otherwise log 5 lines/sec per session.
_LOG_EVERY_N_FRAMES = 25


# Per-session counter for log throttling. Keyed by session id (string).
_frame_counter: dict[str, int] = {}


async def handle_frame(session: "Session", jpeg_bytes: bytes) -> None:
    if not jpeg_bytes:
        log.warning("Session %s: empty frame payload", session.id)
        return

    t_start = time.perf_counter()
    try:
        with Image.open(io.BytesIO(jpeg_bytes)) as img:
            img.load()
            arr = np.asarray(img.convert("RGB"))
    except (UnidentifiedImageError, OSError) as e:
        log.warning("Session %s: failed to decode JPEG (%d bytes): %s",
                    session.id, len(jpeg_bytes), e)
        return
    decode_ms = (time.perf_counter() - t_start) * 1000.0

    session.latest_frame = arr
    session.latest_frame_at = time.time()

    n = _frame_counter.get(session.id, 0) + 1
    _frame_counter[session.id] = n

    if n == 1 or n % _LOG_EVERY_N_FRAMES == 0:
        log.info(
            "Session %s: frame #%d shape=%s size=%dB decode=%.1fms",
            session.id, n, arr.shape, len(jpeg_bytes), decode_ms,
        )
