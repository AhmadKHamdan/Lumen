"""
Hand detection service (MediaPipe Hands).

Wraps Google's MediaPipe Hands solution to locate the user's reaching hand
in a single RGB frame. Designed to be standalone and importable - run from a
Python REPL.

Why MediaPipe Hands?
- 21 hand landmarks per detected hand, robust on partial occlusion / motion.
- CPU realtime (~15-30 fps on a modest laptop).
- Bundled models inside the pip wheel (no auto-download on first run, unlike
  YOLO).
- No torch dependency - keeps the install delta small.

Use case: during the "reach" phase of Object Allocation, once YOLO has locked
on the target and the user is within arm's reach, the user reaches with their
free hand toward the object. The reaching hand enters the bottom of the
camera frame. We detect it, take the index-fingertip pixel position, and feed
that into ``reach_guidance.assess_reach`` to decide what to say next.

Lazy loading
------------
MediaPipe takes ~1 second to import and instantiate. The model loader is
deferred to the first ``detect()`` call so import is cheap. The
``_landmarks_to_pose`` helper is pure (no MediaPipe types in its signature -
it just needs objects with .x / .y), so it's unit-testable without the
library installed.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Optional, Sequence

log = logging.getLogger("lumen.hand")

# MediaPipe HandLandmark indices we care about.
_WRIST_IDX = 0
_INDEX_TIP_IDX = 8

# We only ever track one hand at a time (the reaching one). Detection
# confidence at 0.5 is the MediaPipe default and works well for handheld
# phone scenarios. Tracking confidence is held below detection so we don't
# drop the hand mid-reach as soon as it tilts.
_MAX_HANDS = 1
_MIN_DETECTION_CONFIDENCE = 0.5
_MIN_TRACKING_CONFIDENCE = 0.4

_model = None  # mediapipe.solutions.hands.Hands - lazy-loaded
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


def _get_model():
    """Lazy-load MediaPipe Hands on first call."""
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        log.info("Loading MediaPipe Hands (max_hands=%d, det_conf=%.2f)",
                 _MAX_HANDS, _MIN_DETECTION_CONFIDENCE)
        import mediapipe as mp  # heavy, deferred
        _model = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=_MAX_HANDS,
            min_detection_confidence=_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=_MIN_TRACKING_CONFIDENCE,
        )
        log.info("MediaPipe Hands loaded")
    return _model


def _landmarks_to_pose(
    landmarks: Sequence,
    frame_w: float,
    frame_h: float,
    score: float = 1.0,
) -> Optional[HandPose]:
    """Convert a 21-landmark sequence into a :class:`HandPose`.

    Pure helper - takes any sequence of objects with ``.x`` / ``.y`` in
    normalised [0, 1] coords (MediaPipe's NormalizedLandmark, or a mock for
    tests). Returns ``None`` if the input is empty or malformed.
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
    """Run MediaPipe Hands on a single RGB frame.

    Parameters
    ----------
    frame_rgb : np.ndarray, shape (H, W, 3)
        RGB image (the format ``frame_handler`` stores on the session).

    Returns
    -------
    Optional[HandPose]
        ``None`` if no hand is detected or the frame is empty/degenerate.
    """
    if frame_rgb is None or getattr(frame_rgb, "size", 0) == 0:
        return None
    model = _get_model()

    # MediaPipe wants RGB and is happy with a contiguous numpy array.
    results = model.process(frame_rgb)
    if not getattr(results, "multi_hand_landmarks", None):
        return None

    h, w = frame_rgb.shape[:2]
    return _landmarks_to_pose(results.multi_hand_landmarks[0].landmark, w, h)


def warm_up() -> None:
    """Eagerly load the model. Optional - called at server start to amortise
    the ~1 s import cost off the first user command. Idempotent."""
    _get_model()
