"""The perception layer: one camera frame -> trustworthy detections.

Pipeline per frame (called in this order by the server):
  1. perceive_objects  — COCO indicators (+ obstacle classes while walking), with the
                         whole-frame "wall latch" gate.
  2. process_doors     — the door stack: geometry gates -> 4-class semantic
                         verification -> fridge arbitration -> cross-model de-confusion.
  3. build_boxes       — normalize everything kept into overlay boxes + a `seen` set.
  4. door_geometry     — the strongest door's bearing/distance + 2-of-3 confirmation.

Everything here is about *what is in the frame*; deciding what to DO with it (and what
to say) is the controller's job.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .config import (ASSUMED_HFOV_DEG, CONF, DIST_CAL, DOOR_CONF, DOOR_DEBUG,
                     DOOR_EDGE_MIN, DOOR_FILL_FRAC, DOOR_HEIGHT_M, DOOR_MAX_FRAME_FRAC,
                     DOOR_MAX_WH, DOOR_OBJ_IOU, DOOR_STRONG_CONF, MIN_HITS,
                     OBSTACLE_CLASSES, STEP_LENGTH_M, VERIFY_CONF, VERIFY_IOU)
from .models import _door_model, _model, _name_to_id, _names, _verify_model
from .state import _door_hist, _state


# --- small geometric helpers ----------------------------------------------------

def _iou(a: list, b: list) -> float:
    """Intersection-over-union of two [x1,y1,x2,y2] boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _edge_density(gray, xy) -> float:
    """Fraction of Canny-edge pixels inside a box. Blank walls ~0; doors much higher.
    Used to reject wall false-positives that the detector is (wrongly) confident on."""
    x1, y1, x2, y2 = (int(round(v)) for v in xy)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(gray.shape[1], x2), min(gray.shape[0], y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return 0.0
    edges = cv2.Canny(gray[y1:y2, x1:x2], 50, 150)
    return float(np.count_nonzero(edges)) / edges.size


def _door_region(cx: float) -> str:
    """Spoken bearing for a door, from its horizontal centre (0=left .. 1=right)."""
    if cx < 0.40:
        return "on your left"
    if cx > 0.60:
        return "on your right"
    return "ahead"


def _door_distance_m(px_height: float, img_w_px: int, img_h_px: int) -> float | None:
    """Estimate door distance (m) from its pixel height via a pinhole model.
    The camera's quoted FOV belongs to its WIDER axis = the image's LONGER side.
    Phones stream portrait (h > w), so deriving focal from the width understated
    the focal length — and therefore every distance — by ~33%."""
    if px_height <= 0:
        return None
    long_side = max(img_w_px, img_h_px)
    f_px = (long_side / 2) / math.tan(math.radians(ASSUMED_HFOV_DEG / 2))
    return DIST_CAL * DOOR_HEIGHT_M * f_px / px_height


# --- stage 1: COCO objects -------------------------------------------------------

def perceive_objects(img, w: int, h: int, indicators: set):
    """Run the COCO model (indicators, plus obstacle classes while walking) and apply
    the whole-frame wall-latch gate. Returns (obj_dets, obstacle_dets).

    obj_dets: [(cls_name, conf, [x1,y1,x2,y2])]. obstacle_dets is the subset of known
    obstacle classes, captured before door de-confusion so a chair overlapping the
    doorway isn't suppressed away."""
    # Class-filter to this goal's indicators -> faster, cleaner overlay. While walking
    # to a door we ALSO request the obstacle classes so the corridor watchdog can see
    # chairs/people/etc. in the path.
    wanted = set(indicators)
    if _state["mode"] == "go_door":
        wanted |= OBSTACLE_CLASSES
    wanted_ids = [_name_to_id[c] for c in wanted if c in _name_to_id]
    res = _model.predict(img, verbose=False, conf=CONF, classes=wanted_ids or None)[0]
    obj_dets = []  # (cls_name, conf, [x1,y1,x2,y2])
    for b in res.boxes:
        ocls = _names[int(b.cls[0])]
        oconf = float(b.conf[0])
        oxy = [float(v) for v in b.xyxy[0]]
        # Whole-scene latch gate — same pathology as walls-as-doors, via COCO this
        # time: a blank wall boxed edge-to-edge as a "refrigerator" is a fake kitchen
        # indicator -> FALSE ARRIVAL. While scanning or confirming, the user stands
        # mid-room, so a real indicator never fills the whole frame. (go_door close
        # approaches are exempt, where filling the frame is legitimate.)
        if _state["mode"] in ("discover", "go_indicator", "face_target"):
            ofrac = ((oxy[2] - oxy[0]) * (oxy[3] - oxy[1])) / float(w * h)
            if ofrac > DOOR_MAX_FRAME_FRAC:
                if DOOR_DEBUG:
                    print(f"[obj] {ocls} conf={oconf:.2f} fill={ofrac:.2f} "
                          "-> reject (whole-frame latch)", flush=True)
                continue
        obj_dets.append((ocls, oconf, oxy))

    obstacle_dets = ([d for d in obj_dets if d[0] in OBSTACLE_CLASSES]
                     if _state["mode"] == "go_door" else [])
    return obj_dets, obstacle_dets


# --- stage 2: the door stack -----------------------------------------------------

def process_doors(img, w: int, h: int, obj_dets):
    """Run the door model and the 3-layer door funnel (geometry gates -> semantic
    verification -> fridge arbitration -> cross-model de-confusion).

    Returns (obj_dets, door_dets, near_box, vdoor): the (possibly trimmed) obj_dets,
    the verified doors [(conf, [x1,y1,x2,y2])], whether a close-range panel saturates
    the frame, and the verifier's door boxes (for the tighter distance measurement)."""
    dres = _door_model.predict(img, verbose=False, conf=DOOR_CONF)[0]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # for the blank-wall edge gate
    door_cands = []  # geometric-gate survivors, pending semantic verification
    door_dets = []   # (conf, [x1,y1,x2,y2]) — verified doors
    near_box = False  # a huge door box saturating the frame (close-range panel view)
    for b in dres.boxes:
        cname = _door_model.names[int(b.cls[0])]
        conf = float(b.conf[0])
        xy = [float(v) for v in b.xyxy[0]]
        bw, bh = xy[2] - xy[0], xy[3] - xy[1]
        if cname == "door":
            # Geometric wall gates: very-wide boxes (wall span), too little internal
            # structure (smooth blank wall), and near-full-frame boxes (the detector
            # latching onto the whole scene). The frame-fill test only applies during the
            # SCAN -- once we're approaching a door the box is meant to fill the view.
            dens = _edge_density(gray, xy)
            wh = bw / bh if bh > 0 else 99.0
            frame_frac = (bw * bh) / float(w * h) if (w and h) else 0.0
            too_big = frame_frac > DOOR_MAX_FRAME_FRAC and _state["mode"] == "discover"
            # Close-range panel view: a door at arm's length fills the frame height but
            # is SMOOTH inside, so the edge gate rejects it. Flag it for the transit
            # logic — only honoured there if a tracked approach got us close first.
            if _state["mode"] == "go_door" and conf >= 0.5 and bh / h >= DOOR_FILL_FRAC:
                near_box = True
            # Edge density is NOT a hard gate any more: plain doors in dim light score
            # as low as blank walls. Smooth candidates go to the verifier instead,
            # where only semantic proof (a door/handle seen there) can pass them.
            if bh > 0 and wh <= DOOR_MAX_WH and not too_big:
                door_cands.append((conf, xy, dens))  # pending semantic verification below
            elif DOOR_DEBUG:
                print(f"[door] conf={conf:.2f} edge={dens:.3f} wh={wh:.2f} "
                      f"fill={frame_frac:.2f} -> reject (geometry)", flush=True)
        elif cname == "refrigerator door":
            obj_dets.append(("refrigerator", conf, xy))  # corroborates the fridge

    # --- Semantic verification (4-class DoorDetect second opinion). Geometry can't
    # tell a lace curtain from a door, and COCO sometimes calls a real door a fridge.
    vdoor, vhandle, vfridge = [], [], []
    if _verify_model is not None and (door_cands or any(o[0] == "refrigerator" for o in obj_dets)):
        vres = _verify_model.predict(img, verbose=False, conf=VERIFY_CONF)[0]
        for b in vres.boxes:
            vname = _verify_model.names[int(b.cls[0])]
            vxy = [float(v) for v in b.xyxy[0]]
            if vname == "door":
                vdoor.append(vxy)
            elif vname == "handle":
                vhandle.append(vxy)
            elif vname == "refrigerator door":
                vfridge.append(vxy)

    for conf, xy, dens in door_cands:
        # Corroboration = the verifier also sees a door there, OR a handle inside the
        # box. A weak candidate (curtain-level confidence) without either is dropped;
        # a verifier 'refrigerator door' on top of it means it's a fridge, not a door.
        corro = any(_iou(xy, vb) >= VERIFY_IOU for vb in vdoor)
        if not corro:
            corro = any(xy[0] <= (hb[0] + hb[2]) / 2 <= xy[2]
                        and xy[1] <= (hb[1] + hb[3]) / 2 <= xy[3] for hb in vhandle)
        fridge_like = any(_iou(xy, fb) >= VERIFY_IOU for fb in vfridge)
        # The fridge veto only counts when the verifier saw fridge evidence WITHOUT
        # door evidence. It hallucinates 'refrigerator door' on real doors too, and
        # vetoing frames it SIMULTANEOUSLY corroborated as doors (corro=True) kept
        # breaking detection streaks on the actual door.
        veto = fridge_like and not corro
        if dens >= DOOR_EDGE_MIN or _verify_model is None:
            # Textured interior (frame/seams/handle visible): confidence or
            # corroboration passes it, as before.
            ok = not veto and (_verify_model is None or conf >= DOOR_STRONG_CONF or corro)
        else:
            # Smooth interior: a blank wall OR a plain door in dim light — confidence
            # CANNOT tell them apart (walls score 0.9 too), so only semantic proof
            # (the verifier seeing a door or a handle there) passes it.
            ok = corro
        if DOOR_DEBUG:
            print(f"[door] conf={conf:.2f} edge={dens:.3f} corro={corro} "
                  f"fridge_like={fridge_like} -> {'KEEP' if ok else 'reject (verify)'}",
                  flush=True)
        if ok:
            door_dets.append((conf, xy, corro))  # keep corro: corroborated = strong evidence
        # NOTE: a fridge_like rejection does NOT become a refrigerator sighting.
        # The verifier hallucinates 'refrigerator door' on blank walls, and feeding
        # those into the indicator evidence caused fake fridges in the arrival
        # summary. Vetoing the door is safe; claiming a fridge is not — real
        # fridges are detected by the COCO model on its own.

    # A COCO 'refrigerator' claim that sits on a door candidate is suspect — a real
    # door at an angle often reads as a fridge. Keep it ONLY if the verifier saw a
    # 'refrigerator door' there; otherwise it IS the door (kills false kitchen arrivals).
    # EXCEPT during the directed indicator confirmation: there the user is deliberately
    # pointing at the appliance, the door model routinely fires on real fridges, and
    # this drop was starving the confirmation of its fridge sightings for ~17 s.
    # The confidence-based cross-suppression below still arbitrates the same-box clash.
    confirm_phase = (_state["mode"] == "go_indicator"
                     or (_state["mode"] == "face_target"
                         and _state["target_kind"] == "indicator"))
    if _verify_model is not None and door_cands and not confirm_phase:
        kept_objs = []
        for cls, ocf, oxy in obj_dets:
            if (cls == "refrigerator"
                    and any(_iou(oxy, dxy) >= DOOR_OBJ_IOU for _dc, dxy, _dd in door_cands)
                    and not any(_iou(oxy, fb) >= VERIFY_IOU for fb in vfridge)):
                if DOOR_DEBUG:
                    print(f"[door] COCO fridge conf={ocf:.2f} on a door candidate, "
                          "no 'refrigerator door' backup -> dropped (it's the door)", flush=True)
                continue
            kept_objs.append((cls, ocf, oxy))
        obj_dets = kept_objs

    # Cross-model de-confusion: a fridge and a door look alike and both models ran on
    # the same frame. Where a door box and an object box overlap a lot, keep only the
    # higher-confidence one -> no door<->refrigerator double-claims, while the
    # accurate door model keeps its recall.
    drop_obj, drop_door = set(), set()
    for i, (_ocls, ocf, oxy) in enumerate(obj_dets):
        for j, (dcf, dxy, _dcorro) in enumerate(door_dets):
            if _iou(oxy, dxy) >= DOOR_OBJ_IOU:
                if dcf >= ocf:
                    drop_obj.add(i)
                else:
                    drop_door.add(j)
    obj_dets = [d for i, d in enumerate(obj_dets) if i not in drop_obj]
    door_dets = [d for j, d in enumerate(door_dets) if j not in drop_door]
    return obj_dets, door_dets, near_box, vdoor


# --- stage 3: overlay boxes ------------------------------------------------------

def build_boxes(obj_dets, door_dets, indicators: set, w: int, h: int):
    """Normalize kept detections into overlay boxes (sent to the browser) and the set
    of object classes `seen` this frame (drives indicator evidence)."""
    boxes = []
    seen = set()
    for cls, conf, (x1, y1, x2, y2) in obj_dets:
        seen.add(cls)
        boxes.append({
            "cls": cls, "conf": round(conf, 2),
            "box": [x1 / w, y1 / h, x2 / w, y2 / h],  # normalized
            "indicator": cls in indicators,
        })
    for conf, (x1, y1, x2, y2), _corro in door_dets:
        boxes.append({
            "cls": "door", "conf": round(conf, 2),
            "box": [x1 / w, y1 / h, x2 / w, y2 / h],
            "indicator": False, "door": True,
        })
    return boxes, seen


# --- stage 4: door geometry (bearing / distance / temporal confirmation) ---------

@dataclass
class DoorGeom:
    region: str | None
    dist_m: float | None
    cur_frac: float
    door_cx_frac: float | None
    door_confirmed: bool
    door_dist: float | None
    corro: bool = False  # was this frame's strongest door verifier-corroborated?


def door_geometry(door_dets, vdoor, w: int, h: int) -> DoorGeom:
    """Bearing + distance of the strongest door, its 2-of-3 temporal confirmation, and
    the steadied step distance. Updates the rolling `_door_hist` and remembers how close
    the approach got (`last_door_dist`)."""
    best_corro = False
    if door_dets:
        bx, best_corro = max(((xy, co) for _conf, xy, co in door_dets),
                             key=lambda t: (t[0][2] - t[0][0]) * (t[0][3] - t[0][1]))
        door_cx_frac = ((bx[0] + bx[2]) / 2) / w  # 0 = left edge .. 1 = right edge
        region = _door_region(door_cx_frac)
        # The primary model's boxes run LOOSE at range (wall above/below the door),
        # and distance is inverse to box height — a fat box reads as "near". When the
        # verifier also boxed this door, measure on the tighter (shorter) of the two.
        meas_h = bx[3] - bx[1]
        for vb in vdoor:
            if _iou(list(bx), vb) >= VERIFY_IOU:
                meas_h = min(meas_h, vb[3] - vb[1])
        dist_m = _door_distance_m(meas_h, w, h)
        cur_frac = (bx[3] - bx[1]) / h  # how much of the frame height the door fills
        if DOOR_DEBUG and dist_m is not None:
            print(f"[dist] box_h={meas_h:.0f}px (raw {bx[3] - bx[1]:.0f}) frame={w}x{h} "
                  f"-> {dist_m:.1f} m (~{max(1, round(dist_m / STEP_LENGTH_M))} steps)",
                  flush=True)
    else:
        region, dist_m, cur_frac, door_cx_frac = None, None, 0.0, None
    _door_hist.append((region, dist_m) if region else None)
    confirmed_doors = [d for d in _door_hist if d]
    door_confirmed = len(confirmed_doors) >= MIN_HITS
    # Steady the step count with the median distance over the window (less jitter).
    _dists = sorted(d[1] for d in confirmed_doors if d[1] is not None)
    door_dist = _dists[len(_dists) // 2] if _dists else None
    if door_confirmed and door_dist is not None:
        _state["last_door_dist"] = door_dist  # remember how close the approach got
    return DoorGeom(region, dist_m, cur_frac, door_cx_frac, door_confirmed, door_dist,
                    best_corro)
