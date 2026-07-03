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
                     DEPTH_INPUT_W, DEPTH_LANE_ROWS, DEPTH_LANE_X, DEPTH_NEAR_ROWS,
                     DEPTH_REL_MARGIN, DOOR_DEBUG, FLOOR_LANE_ROWS, FLOOR_LANE_X,
                     FLOOR_MIN_COVER, OBST_BOTTOM_FRAC, OBST_CLEAR_HITS, OBST_HITS,
                     OBST_MIN_H_FRAC, OBST_MIN_OVERLAP, OBST_NAMES, OBST_REPROMPT,
                     OBST_SIGNAL)
from .models import _depth_pipe, _seg_infer, _seg_walkable_ids
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
    # The depth lane is narrower than the YOLO corridor: side objects (a fan, a bin
    # beside the door) must not graze into it — only a mid-path blocker should.
    cL, cR = int(DEPTH_LANE_X[0] * W), int(DEPTH_LANE_X[1] * W)
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


def _floor_tripwire(img):
    """Floor-segmentation obstacle check (OBST-3). Returns the side to step toward
    ('left'/'right') if the walking lane has stopped being mostly floor, else None.

    A semantic model labels every pixel; the lane (a floor strip just ahead of the
    feet, centre band) must stay mostly walkable — floor/rug, plus the door we are
    deliberately walking toward. Something BESIDE the path leaves the lane's floor
    continuous; something IN the path punches a hole in it. The sidestep suggestion
    points at whichever flank has more visible floor."""
    if _seg_infer is None:
        return None
    ids = _seg_infer(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    walk = np.isin(ids, _seg_walkable_ids)
    H, W = walk.shape
    r0, r1 = int(FLOOR_LANE_ROWS[0] * H), int(FLOOR_LANE_ROWS[1] * H)
    cL, cR = int(FLOOR_LANE_X[0] * W), int(FLOOR_LANE_X[1] * W)
    lane = walk[r0:r1, cL:cR]
    cover = float(lane.mean()) if lane.size else 1.0
    left = walk[r0:r1, int(0.10 * W):cL]
    right = walk[r0:r1, cR:int(0.90 * W)]
    lcov = float(left.mean()) if left.size else 0.0
    rcov = float(right.mean()) if right.size else 0.0
    if DOOR_DEBUG:
        print(f"[floor] lane={cover:.2f} L={lcov:.2f} R={rcov:.2f}"
              f"{'' if cover >= FLOOR_MIN_COVER else ' -> blocked'}", flush=True)
    if cover >= FLOOR_MIN_COVER:
        return None
    return "left" if lcov >= rcov else "right"


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


def _obstacle_watchdog(obstacle_dets, unnamed_side, w: int, h: int):
    """Debounced corridor watchdog fusing two layers: the YOLO class layer (names the
    object) and the class-agnostic unnamed signal (floor or depth — precomputed by
    the caller as a step-aside side, or None). Returns (guidance, priority, blocking).
    blocking=True means a confirmed obstacle is in the lane now, so the caller
    suppresses door guidance. Speaks on first confirm, again every OBST_REPROMPT
    frames while still blocked, and once when the path clears."""
    yolo = _obstacle_in_corridor(obstacle_dets, w, h)   # (cls, cx) or None
    if yolo is not None:
        cls, cx = yolo
        name = OBST_NAMES.get(cls, cls)
        art = "an" if name[:1].lower() in "aeiou" else "a"
        what, side = f"{art} {name}", ("right" if cx < 0.5 else "left")
    else:
        what, side = ("something", unnamed_side) if unnamed_side else (None, None)

    if what is not None:
        _state["obst_clear"] = 0
        _state["obst_hits"] += 1
        if _state["obst_hits"] < OBST_HITS:
            return "", False, False  # not confirmed yet -> let door guidance run
        if _state["obst_hold"] > 0:
            # The door call-out ("...let me check the path ahead") is still playing —
            # confirmed, but hold the SPEECH so the sentence finishes; the warning
            # becomes the path verdict right after. Calm tone, never a barked "Stop".
            return "", False, True
        if DOOR_DEBUG:
            print(f"[obst] {what} -> blocking, step {side}", flush=True)
        if not _state["obst_active"]:
            _state["obst_active"] = True
            _state["obst_cool"] = OBST_REPROMPT
            return f"There's {what} in your path. Step to your {side}, where it's clear.", True, True
        if _state["obst_cool"] <= 0:
            _state["obst_cool"] = OBST_REPROMPT
            return f"It's still in your path — step more to your {side}.", True, True
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
    """Run the watchdog while WALKING in go_door. Off-walk (scanning/turning phases),
    reset the debounce so the next approach starts clean.
    Returns (guidance, priority, blocking)."""
    if _state["mode"] == "go_door":
        if not _state["near_latch"]:
            # Approach: both layers. The class-agnostic unnamed signal per OBST_SIGNAL
            # ("off" or a model that failed to load -> None -> YOLO only).
            if OBST_SIGNAL == "floor":
                unnamed_side = _floor_tripwire(img)
            elif OBST_SIGNAL == "depth":
                unnamed_side = _depth_tripwire(_depth_map(img))
            else:
                unnamed_side = None
            return _obstacle_watchdog(obstacle_dets, unnamed_side, w, h)
        # AT/THROUGH the door: the panel fills the frame, so the floor/depth signal
        # is meaningless here and pauses — but a PERSON stepping into the doorway is
        # still a named YOLO box. The named layer stays armed through the transit.
        return _obstacle_watchdog(obstacle_dets, None, w, h)
    _state["obst_hits"] = 0
    _state["obst_clear"] = 0
    _state["obst_active"] = False
    return "", False, False
