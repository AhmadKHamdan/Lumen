"""
Spatial reasoning: turn a bounding box into human directions.

Maps a detection's pixel-space bounding box into two coarse, speakable
buckets:

- ``region``   : ``"left"`` | ``"center"`` | ``"right"`` - from the box's
                 horizontal center relative to the frame width.
- ``distance`` : ``"near"`` | ``"medium"`` | ``"far"`` - estimated from how
                 large the box appears compared to a *per-class* expectation
                 of "at arm's reach" size.

Why per-class distance?
-----------------------
A monocular RGB camera gives no true depth. Apparent box size is the only
proxy. But objects have wildly different *physical* sizes: at arm's reach
(~40 cm) on a typical phone rear camera (~67 deg horizontal FOV):

    object       physical width   width as fraction of frame
    -----------  ---------------  -------------------------
    cell phone   ~7 cm            ~0.12
    cup          ~10 cm           ~0.18
    book         ~15 cm           ~0.22
    bottle       ~8 cm wide       ~0.12  (but ~25 cm tall -> use max)
    laptop       ~33 cm           ~0.40
    monitor/tv   ~50 cm           ~0.60

The old "area_frac >= 0.20 = near" rule sent a laptop right in front of you
to "medium" ("a few steps ahead") because its area was 16 %, and sent a
mug-near-the-face to "far". This module classifies distance using
``max(width_frac, height_frac)`` (which handles both wide and tall objects)
compared against a per-COCO-class threshold. Same image, correct bucket per
object.

The thresholds are deliberately wide so the bucket doesn't flicker as the
user moves. Approximate is fine - the spoken guidance only has three buckets.

Pure module: numpy / math only, no torch / no model. Unit-tested in
isolation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

# ---------- region (left / center / right) ----------

# Fraction-of-width split points for the box center.
# [0, 0.35) -> left, [0.35, 0.65] -> center, (0.65, 1.0] -> right.
REGION_LEFT_MAX = 0.35
REGION_RIGHT_MIN = 0.65


# ---------- distance (near / medium / far) ----------

# For each COCO class we care about, the expected "apparent size" (= max of
# width-fraction and height-fraction of the frame) when the object is at
# arm's reach (~40 cm) on a typical phone camera. Tuned for our OBJECT_NOUNS
# list in command_parser.py. Anything not listed falls back to the default.
PER_CLASS_NEAR_APPARENT: dict[str, float] = {
    # Small handheld objects
    "cell phone": 0.12,
    "remote": 0.12,
    "mouse": 0.10,
    "scissors": 0.10,
    "cup": 0.18,
    "bottle": 0.18,   # narrow but tall - max of w/h works
    "clock": 0.18,
    "vase": 0.18,
    "book": 0.22,
    # Mid-sized objects
    "keyboard": 0.40,
    "laptop": 0.40,
    "microwave": 0.40,
    "sink": 0.40,
    "toilet": 0.40,
    "chair": 0.40,
    # Large objects / appliances / furniture
    "oven": 0.50,
    "dining table": 0.50,
    "tv": 0.55,
    "couch": 0.60,
    "bed": 0.60,
    "refrigerator": 0.60,
}

# Fallback for unknown classes. Roughly book-sized.
DEFAULT_NEAR_APPARENT = 0.22

# How close the observed apparent fraction must come to the class's
# "near-at-arm's-reach" expectation. ratio = apparent / class_threshold.
#   ratio >= NEAR_RATIO_MIN -> the user can reach it -> "near"
#   ratio <= FAR_RATIO_MAX  -> clearly out of reach   -> "far"
#   otherwise               -> "medium" (a few steps away)
NEAR_RATIO_MIN = 0.85
FAR_RATIO_MAX = 0.35


@dataclass(frozen=True)
class SpatialInfo:
    """Coarse spatial description of a detection within a frame.

    ``apparent_frac`` is ``max(width_frac, height_frac)`` - the cue we
    actually classify distance from. ``area_frac`` is kept for diagnostics
    and backwards compatibility with existing tests.
    """

    region: str           # "left" | "center" | "right"
    distance: str         # "near" | "medium" | "far"
    cx_frac: float        # box center x as fraction of frame width
    area_frac: float      # box area as fraction of frame area
    apparent_frac: float = 0.0  # max(w_frac, h_frac), the new distance cue
    label: str = ""       # which COCO class informed the distance call


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _classify_region(cx_frac: float) -> str:
    if cx_frac < REGION_LEFT_MAX:
        return "left"
    if cx_frac > REGION_RIGHT_MIN:
        return "right"
    return "center"


def near_threshold_for(label: str) -> float:
    """The "at arm's reach" apparent-size threshold for a COCO class name.

    Returns the default if the class isn't in :data:`PER_CLASS_NEAR_APPARENT`.
    """
    return PER_CLASS_NEAR_APPARENT.get((label or "").lower(), DEFAULT_NEAR_APPARENT)


def _classify_distance(apparent_frac: float, label: str) -> str:
    threshold = near_threshold_for(label)
    if threshold <= 0:
        return "medium"
    ratio = apparent_frac / threshold
    if ratio >= NEAR_RATIO_MIN:
        return "near"
    if ratio <= FAR_RATIO_MAX:
        return "far"
    return "medium"


def locate(
    box: Sequence[float],
    frame_w: float,
    frame_h: float,
    label: str = "",
) -> SpatialInfo:
    """Describe ``box`` within a ``frame_w`` x ``frame_h`` frame.

    Parameters
    ----------
    box : (x1, y1, x2, y2)
        Pixel-space bounding box, top-left origin.
    frame_w, frame_h : float
        Frame dimensions in pixels.
    label : str
        COCO class name of the detected object. Drives per-class distance
        bucketing. Pass ``""`` to fall back to default thresholds.
    """
    x1, y1, x2, y2 = (float(v) for v in box[:4])

    if frame_w <= 0 or frame_h <= 0:
        return SpatialInfo(
            region="center", distance="medium",
            cx_frac=0.5, area_frac=0.0,
            apparent_frac=0.0, label=label,
        )

    cx = (x1 + x2) / 2.0
    cx_frac = _clamp01(cx / frame_w)

    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    w_frac = _clamp01(bw / frame_w)
    h_frac = _clamp01(bh / frame_h)
    apparent_frac = max(w_frac, h_frac)
    area_frac = _clamp01((bw * bh) / (frame_w * frame_h))

    return SpatialInfo(
        region=_classify_region(cx_frac),
        distance=_classify_distance(apparent_frac, label),
        cx_frac=cx_frac,
        area_frac=area_frac,
        apparent_frac=apparent_frac,
        label=label,
    )


def most_centered(detections: Sequence, frame_w: float):
    """Pick the detection whose horizontal center is closest to the frame center.

    Used when several instances of the target are visible at once - we guide
    the user to the most head-on one. ``detections`` is any sequence of objects
    exposing a ``center_x`` attribute (e.g. yolo_service.Detection).

    Returns the chosen detection, or ``None`` if the sequence is empty.
    """
    if not detections:
        return None
    if frame_w <= 0:
        return detections[0]
    target_cx = frame_w / 2.0
    return min(detections, key=lambda d: abs(d.center_x - target_cx))


def closest(detections: Sequence):
    """Pick the detection with the largest box area (the apparent-nearest one).

    Returns the chosen detection, or ``None`` if the sequence is empty.
    """
    if not detections:
        return None
    return max(detections, key=lambda d: d.area)
