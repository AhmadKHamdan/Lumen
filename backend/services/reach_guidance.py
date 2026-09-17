"""
Reach guidance: turn (target box, fingertip) into a hand-relative direction.

Once Object Allocation has located the target and the user is within arm's
reach, the user reaches with their free hand. Their reaching hand enters the
camera frame from below. We compute the vector from the index fingertip to
the target's box centroid in pixel space and decide what to say:

    state       what's true                                       phrase
    ----------- ------------------------------------------------- ------------------------------
    "approach"  finger is outside the box, far from centroid      "Move your hand left/right/up/down."
    "almost"    finger is outside the box but close to centroid    "Almost there. Reach forward."
    "touching"  fingertip is inside the box                        "Your hand is on the cup. Grasp it."

The touch case is what trips the auto-completion path: the loop fires
FSM ``task_complete`` and the task ends. That's the only autonomous task
exit in the system - the user can still stop earlier by saying "cancel".

Coordinate semantics
--------------------
The user is looking through the rear camera, so image-right == user's right
and "raise your hand" means moving the hand toward smaller y (up in image).
Frame origin is top-left.

Pure module: math only, no MediaPipe, no model. Unit-tested in isolation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

# ---------- thresholds ----------

# Fingertip-to-target-centroid distance (in fraction of frame width OR height)
# below which we consider the user "almost there" instead of naming a
# direction. Wider than the initial 10% so we don't chatter with tiny left/right
# flips when the fingertip is nearly on target.
ALMOST_CENTER_FRAC = 0.15


@dataclass(frozen=True)
class ReachInfo:
    """Hand position relative to the target box."""

    state: str           # "approach" | "almost" | "touching"
    direction: str       # "left" | "right" | "up" | "down" | "center"
    dx_frac: float       # (target_cx - finger_x) / frame_w
    dy_frac: float       # (target_cy - finger_y) / frame_h


def assess_reach(
    target_box: Sequence[float],
    fingertip_xy: Sequence[float],
    frame_w: float,
    frame_h: float,
) -> ReachInfo:
    """Classify the fingertip's position relative to the target box.

    Parameters
    ----------
    target_box : (x1, y1, x2, y2)
        Target's pixel-space bounding box from YOLO.
    fingertip_xy : (x, y)
        Index fingertip pixel position from hand_service.
    frame_w, frame_h : float
        Frame dimensions in pixels.
    """
    if frame_w <= 0 or frame_h <= 0:
        return ReachInfo("approach", "center", 0.0, 0.0)

    x1, y1, x2, y2 = (float(v) for v in target_box[:4])
    fx, fy = float(fingertip_xy[0]), float(fingertip_xy[1])

    # Touch test: fingertip inside the box -> autocomplete path.
    if x1 <= fx <= x2 and y1 <= fy <= y2:
        return ReachInfo("touching", "center", 0.0, 0.0)

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    dx = cx - fx   # >0: target right of finger
    dy = cy - fy   # >0: target below finger
    dxf = dx / frame_w
    dyf = dy / frame_h

    abs_dxf = abs(dxf)
    abs_dyf = abs(dyf)

    # Close enough to the centroid in both axes -> "almost", no named direction.
    if max(abs_dxf, abs_dyf) < ALMOST_CENTER_FRAC:
        return ReachInfo("almost", "center", dxf, dyf)

    # Pick the dominant axis by PIXEL magnitude, not fractional. Frame width
    # (640) is bigger than height (480), so comparing fractions used to bias
    # toward the vertical axis - a 30 px offset in both x and y came out
    # as dxf=0.047 < dyf=0.063, triggering an up/down cue instead of a
    # left/right cue. Pixel comparison matches human intuition.
    if abs(dx) >= abs(dy):
        direction = "right" if dx > 0 else "left"
    else:
        direction = "down" if dy > 0 else "up"
    return ReachInfo("approach", direction, dxf, dyf)


def grab_phrase(target: str) -> str:
    """Spoken cue once the hand has HOVERED near the target through a full
    re-affirm window. The 2D touch test can't see depth — a fingertip reaching
    past or beside the object may never enter the box — so instead of looping
    "reach forward" forever, hand the final action to the user."""
    target = target or "object"
    return (f"Your hand is right next to the {target}. Reach a little further "
            "and try to pick it up. Say got it when you have it.")


def reach_phrase(target: str, info: ReachInfo) -> str:
    """Compose the spoken cue for the current :class:`ReachInfo`."""
    target = target or "object"

    if info.state == "touching":
        return f"Your hand is on the {target}. Grasp it."
    if info.state == "almost":
        return "Almost there. Push your hand straight forward."
    # Consistent "Move your hand X" wording across all four directions.
    if info.direction == "left":
        return "Move your hand to the left."
    if info.direction == "right":
        return "Move your hand to the right."
    if info.direction == "up":
        return "Move your hand up."
    if info.direction == "down":
        return "Move your hand down."
    # Defensive default.
    return "Move your hand straight forward."
