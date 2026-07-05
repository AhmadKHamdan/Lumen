"""
Hand detection service (MediaPipe Tasks - HandLandmarker).

Wraps Google's MediaPipe HandLandmarker to locate the user's reaching hand
in a single RGB frame. Designed to be standalone and importable - run from a
Python REPL.

Why the Tasks API (not solutions.hands)?
---------------------------------------
MediaPipe 0.10.14+ shipped Python 3.13 wheels that DROP the legacy
``mediapipe.solutions.*`` namespace and expose only the new Tasks API
(``mediapipe.tasks.python.vision.HandLandmarker``). Our project runs on
Python 3.13, so we use the Tasks API. It produces the same 21 landmarks;
the only user-visible difference is a one-time download of the
``hand_landmarker.task`` model file (~10 MB, same auto-cache pattern as
YOLO's ``yolov8n.pt``).

Use case
--------
During the "reach" phase of Object Allocation, once YOLO has locked on the
target and the user is within arm's reach, they reach with their free hand
toward the object. The reaching hand enters the bottom of the camera frame.
We detect it, take the index-fingertip pixel position, and feed that into
``reach_guidance.assess_reach`` to decide what to say next.

Lazy loading
------------
MediaPipe Tasks imports are heavy. Model creation is deferred to the first
``detect()`` call so importing this module is cheap. The
``_landmarks_to_pose`` helper is pure (takes any objects with ``.x`` /
``.y``), so it unit-tests without MediaPipe installed at all.
"""
from __future__ import annotations

import logging
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

log = logging.getLogger("lumen.hand")

# MediaPipe HandLandmark indices we care about.
_WRIST_IDX = 0
_INDEX_TIP_IDX = 8

# Detection tuning. We only track one hand (the reaching one). Detection
# confidence is at 0.5 (MediaPipe default). Tracking / presence confidence
# is held slightly lower so we don't drop the hand mid-reach as it tilts.
_NUM_HANDS = 1
_MIN_DETECTION_CONFIDENCE = 0.5
_MIN_PRESENCE_CONFIDENCE = 0.5
_MIN_TRACKING_CONFIDENCE = 0.4

# Model file for MediaPipe HandLandmarker. Auto-downloaded on first use and
# cached in the process's current working directory (which is ``backend/`` when
# uvicorn is launched the standard way), mirroring how ultralytics caches
# yolov8n.pt right there.
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
_MODEL_FILENAME = "hand_landmarker.task"

_model = None  # mediapipe.tasks.python.vision.HandLandmarker - lazy-loaded
_model_lock = threading.Lock()


@dataclass(frozen=True)
class HandPose:
    """One detected hand in a single frame.

    All coordinates are in pixel space (top-left origin), so they're directly
    comparable to YOLO's bounding boxes.

    Attributes
    ----------
    fingertip : (x, y)
        Index-finger tip pixel position. This is the "reaching point".
    wrist : (x, y)
        Wrist pixel position. More stable across motion than fingertip; used
        as a fallback when the tip is occluded.
    bbox : (x1, y1, x2, y2)
        Tight box around all 21 landmarks.
    score : float
        Detection confidence in [0, 1] (1.0 when not reported by MediaPipe).
    """

    fingertip: tuple[float, float]
    wrist: tuple[float, float]
    bbox: tuple[float, float, float, float]
    score: float


def _ensure_model_file() -> Path:
    """Download ``hand_landmarker.task`` if we don't have it yet.

    Cached in the current working directory (typically ``backend/``), so the
    same file is reused across restarts and doesn't pollute a global cache.
    """
    path = Path.cwd() / _MODEL_FILENAME
    if path.exists():
        return path
    log.info("Downloading MediaPipe Hands model (~10 MB) to %s", path)
    urllib.request.urlretrieve(_MODEL_URL, path)
    log.info("MediaPipe Hands model cached at %s", path)
    return path


def _get_model():
    """Lazy-load the HandLandmarker on first call."""
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        log.info(
            "Loading MediaPipe HandLandmarker (num_hands=%d, det_conf=%.2f)",
            _NUM_HANDS, _MIN_DETECTION_CONFIDENCE,
        )
        # Deferred imports - MediaPipe Tasks is heavy.
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            HandLandmarker,
            HandLandmarkerOptions,
            RunningMode,
        )
        model_path = str(_ensure_model_file())
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_hands=_NUM_HANDS,
            min_hand_detection_confidence=_MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=_MIN_PRESENCE_CONFIDENCE,
            min_tracking_confidence=_MIN_TRACKING_CONFIDENCE,
        )
        _model = HandLandmarker.create_from_options(options)
        log.info("MediaPipe HandLandmarker loaded")
    return _model


def _landmarks_to_pose(
    landmarks: Sequence,
    frame_w: float,
    frame_h: float,
    score: float = 1.0,
) -> Optional[HandPose]:
    """Convert a 21-landmark sequence into a :class:`HandPose`.

    Pure helper - takes any sequence of objects with ``.x`` / ``.y`` in
    normalised [0, 1] coords. Works with MediaPipe's NormalizedLandmark, the
    Tasks API's Landmark, or a mock object for tests.
    """
    if not landmarks or len(landmarks) <= max(_WRIST_IDX, _INDEX_TIP_IDX):
        return None
    wrist = (landmarks[_WRIST_IDX].x * frame_w, landmarks[_WRIST_IDX].y * frame_h)
    tip = (landmarks[_INDEX_TIP_IDX].x * frame_w, landmarks[_INDEX_TIP_IDX].y * frame_h)
    xs = [lm.x * frame_w for lm in landmarks]
    ys = [lm.y * frame_h for lm in landmarks]
    bbox = (min(xs), min(ys), max(xs), max(ys))
    return HandPose(fingertip=tip, wrist=wrist, bbox=bbox, score=float(score))


def detect(frame_rgb) -> Optional[HandPose]:
    """Run MediaPipe HandLandmarker on a single RGB frame.

    Parameters
    ----------
    frame_rgb : np.ndarray, shape (H, W, 3), dtype uint8
        RGB image (the format ``frame_handler`` stores on the session).

    Returns
    -------
    Optional[HandPose]
        ``None`` if no hand is detected or the frame is empty/degenerate.
    """
    if frame_rgb is None or getattr(frame_rgb, "size", 0) == 0:
        return None
    model = _get_model()

    # Wrap the numpy frame in the MediaPipe Image container that the Tasks
    # API expects. SRGB tells MediaPipe the layout is uint8 RGB, which is
    # what frame_handler decoded via Pillow.
    import mediapipe as mp
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = model.detect(mp_image)
    if not getattr(result, "hand_landmarks", None):
        return None

    h, w = frame_rgb.shape[:2]
    # Tasks API returns a list of hands, each a list of 21 landmarks.
    return _landmarks_to_pose(result.hand_landmarks[0], w, h)


def warm_up() -> None:
    """Eagerly load the model. Optional - call at server start to move the
    ~1 s import + model creation cost off the first user command. Idempotent.
    """
    _get_model()
