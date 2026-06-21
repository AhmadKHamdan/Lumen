"""The obstacle watchdog — armed ONLY while walking to a door (`go_door`), because
that's the only phase the user is moving.

Two fused layers:
- Phase A (`_obstacle_in_corridor`): YOLO named objects in the lower-centre walking
  lane — names the thing ("a chair").
- Phase B (`_depth_tripwire`): a class-agnostic monocular-depth check that catches
  UNNAMED clutter (clothes piles, boxes, a standing fan) COCO has no word for.

`evaluate()` is the entry point the server calls; it gates on phase + near_latch and
returns (guidance, priority, blocking). `blocking=True` makes the controller suppress
door guidance for that frame (safety first).
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .config import (CORRIDOR_X, DEPTH_AREA_FRAC, DEPTH_FAR_ROWS, DEPTH_FLAT_MIN,
                     DEPTH_INPUT_W, DEPTH_LANE_ROWS, DEPTH_NEAR_ROWS, DEPTH_REL_MARGIN,
                     DOOR_DEBUG, OBST_BOTTOM_FRAC, OBST_CLEAR_HITS, OBST_HITS,
                     OBST_MIN_H_FRAC, OBST_MIN_OVERLAP, OBST_NAMES, OBST_REPROMPT)
from .models import _depth_pipe
from .state import _state


def _depth_map(img):
    """Relative depth map (PIL-normalized 0..255, H'xW') for a frame, or None.
    Downscaled for speed; orientation matches the input."""
    if _depth_pipe is None:
        return None
    dw = DEPTH_INPUT_W
    dh = max(1, int(round(dw * img.shape[0] / img.shape[1])))
    small = cv2.resize(img, (dw, dh))
    pim = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    return np.asarray(_depth_pipe(pim)["depth"], dtype=np.float32)


def _depth_tripwire(d):
    """Class-agnostic obstacle check from a depth map. Returns the side to step toward
    ('left'/'right') if something sits close in the walking lane, else None.

    Self-calibrating per frame so the model's near/far sign never matters: the bottom
    strip (floor at the feet) and the top-centre strip anchor near=1 / far=0, then each
    lane pixel is compared to the floor at ITS OWN row (the floor recedes up the frame).
    A patch of the centre lane that is much nearer than its row's side-floor is an
    obstacle — works even when it covers only part of the lane."""
    if d is None:
        return None
    H, W = d.shape
    r0, r1 = int(DEPTH_LANE_ROWS[0] * H), int(DEPTH_LANE_ROWS[1] * H)
    near_ref = float(np.median(d[int(DEPTH_NEAR_ROWS[0] * H):int(DEPTH_NEAR_ROWS[1] * H), :]))
    far_ref = float(np.median(d[int(DEPTH_FAR_ROWS[0] * H):int(DEPTH_FAR_ROWS[1] * H),
                                int(0.30 * W):int(0.70 * W)]))
    denom = near_ref - far_ref
    if abs(denom) < DEPTH_FLAT_MIN:
        return None  # too flat to judge (e.g. facing a near blank wall)

    lane = (d[r0:r1] - far_ref) / denom          # nearness map: 0 far .. 1 near
    cL, cR = int(CORRIDOR_X[0] * W), int(CORRIDOR_X[1] * W)
    sL, sR = int(0.22 * W), int(0.78 * W)
    side_floor = np.median(np.concatenate([lane[:, :sL], lane[:, sR:]], axis=1),
                           axis=1, keepdims=True)  # per-row floor baseline
    centre = lane[:, cL:cR]
    intrude = (centre - side_floor) > DEPTH_REL_MARGIN  # nearer than the floor at that row
    frac = float(intrude.mean())
    if DOOR_DEBUG:
        print(f"[depth] near={near_ref:.0f} far={far_ref:.0f} intrude={frac:.2f}", flush=True)
    if frac < DEPTH_AREA_FRAC:
        return None
    half = intrude.shape[1] // 2  # step away from the half where the intrusion sits
    return "right" if intrude[:, :half].sum() > intrude[:, half:].sum() else "left"


def _obstacle_in_corridor(obstacle_dets, w: int, h: int):
    """Most intrusive known obstacle standing in the walking lane, or None.
    The lane is the lower-centre band of the frame (the strip the user walks into).
    Returns (class_name, center_x_frac) of the worst offender."""
    lo, hi = CORRIDOR_X
    best = None
    for cls, _conf, (x1, y1, x2, y2) in obstacle_dets:
        if (y2 - y1) / h < OBST_MIN_H_FRAC or y2 / h < OBST_BOTTOM_FRAC:
            continue  # too small/far, or sitting high (not on the floor ahead)
        overlap = max(0.0, min(x2, hi * w) - max(x1, lo * w)) / w
        if overlap < OBST_MIN_OVERLAP:
            continue  # off to the side -> the user won't walk into it
        score = overlap * ((y2 - y1) / h)  # more lane coverage + taller (closer) = worse
        if best is None or score > best[0]:
            best = (score, cls, ((x1 + x2) / 2) / w)
    return None if best is None else (best[1], best[2])


def _obstacle_watchdog(obstacle_dets, depth_map, w: int, h: int):
    """Debounced corridor watchdog fusing two layers: the YOLO class layer (names the
    object) and the depth tripwire (catches anything, named or not). Returns
    (guidance, priority, blocking). blocking=True means a confirmed obstacle is in the
    lane now, so the caller suppresses door guidance. Speaks on first confirm, again
    every OBST_REPROMPT frames while still blocked, and once when the path clears."""
    yolo = _obstacle_in_corridor(obstacle_dets, w, h)   # (cls, cx) or None
    if yolo is not None:
        cls, cx = yolo
        name = OBST_NAMES.get(cls, cls)
        art = "an" if name[:1].lower() in "aeiou" else "a"
        what, side = f"{art} {name}", ("right" if cx < 0.5 else "left")
    else:
        dside = _depth_tripwire(depth_map)              # 'left'/'right' or None
        what, side = ("something", dside) if dside else (None, None)

    if what is not None:
        _state["obst_clear"] = 0
        _state["obst_hits"] += 1
        if _state["obst_hits"] < OBST_HITS:
            return "", False, False  # not confirmed yet -> let door guidance run
        if DOOR_DEBUG:
            print(f"[obst] {what} -> blocking, step {side}", flush=True)
        if not _state["obst_active"]:
            _state["obst_active"] = True
            _state["obst_cool"] = OBST_REPROMPT
            return f"Stop. There's {what} in your way. Step to your {side}.", True, True
        if _state["obst_cool"] <= 0:
            _state["obst_cool"] = OBST_REPROMPT
            return f"Still blocked. Step to your {side}, slowly.", True, True
        _state["obst_cool"] -= 1
        return "", False, True  # blocking, mid-cooldown -> stay silent this frame
    # corridor clear this frame
    _state["obst_hits"] = 0
    if _state["obst_active"]:
        _state["obst_clear"] += 1
        if _state["obst_clear"] >= OBST_CLEAR_HITS:
            _state["obst_active"] = False
            _state["obst_clear"] = 0
            return "Okay, the way ahead is clear.", True, False
        return "", False, True  # brief grace before resuming door directions
    return "", False, False


def evaluate(obstacle_dets, img, w: int, h: int):
    """Run the watchdog only while WALKING up to a door (not once we're at it, where the
    door panel itself fills the lane). Off-walk, reset the debounce so the next approach
    starts clean. Returns (guidance, priority, blocking)."""
    if _state["mode"] == "go_door" and not _state["near_latch"]:
        depth_map = _depth_map(img)  # None if the depth model didn't load (YOLO-only)
        return _obstacle_watchdog(obstacle_dets, depth_map, w, h)
    _state["obst_hits"] = 0
    _state["obst_clear"] = 0
    _state["obst_active"] = False
    return "", False, False
