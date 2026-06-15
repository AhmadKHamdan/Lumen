"""M1 browser demo — backend.

The browser captures the camera (reliable getUserMedia path) and speaks via the
Web Speech API. This FastAPI server just runs the vision + logic: YOLOv8n (COCO)
detection -> 2-of-3 temporal confirmation -> navigation.goal_map arrival check.

Run:
    python webdemo/server.py
Then open http://localhost:8000 in Chrome/Edge and click Start.

No OpenCV camera access — sidesteps the flaky local webcam entirely.
"""
from __future__ import annotations

import base64
import math
import sys
from collections import Counter, deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from navigation import resolve_goal, indicators_for, evaluate_arrival, arrival_phrase

CONF = 0.45
WINDOW, MIN_HITS = 3, 2  # 2-of-3 temporal rule for DOORS (presence, not accumulation)
# Indicators are accumulated across the whole room-scan instead: an object seen in
# >= this many frames during one room visit counts as confirmed. A slow sweep then
# builds evidence rather than forgetting it after 3 frames. Arrival is declared
# straight from the scan (no second confirmation pass), so the bar is high.
INDICATOR_HITS = 4
# Motion gate: if the phone pans too fast the frame is blurred and useless, so we
# skip detection and coach the user to slow down. Tunable (small-image mean diff).
MOTION_MAX = 60.0

app = FastAPI()

# yolov8m, not n: on a GPU it's ~18 ms/frame slower but far more reliable at
# spotting kitchen/bathroom indicators at angle and distance. Detection accuracy
# is the bottleneck here, not local inference time.
print("Loading YOLOv8m (COCO)...")
from ultralytics import YOLO  # noqa: E402
_model = YOLO("yolov8m.pt")
_names = _model.names
_name_to_id = {v: k for k, v in _names.items()}  # COCO class name -> id (for class filtering)

# Custom single-class door detector (trained, mAP50 ~0.95). Resolved relative to
# the repo root so it works regardless of the shell's cwd.
DOOR_CONF = 0.45
# Shape sanity check: doors aren't extremely wide. Rejecting very wide boxes cuts
# some wall/furniture false positives without hurting normal door recall.
DOOR_MAX_WH = 1.4  # max width/height ratio to still count as a door
# Blank walls fire as doors with door-level confidence. A real door has internal
# structure (frame, panel seams, hinges, handle) -> many edges; a blank wall is
# smooth -> almost none. Reject door boxes whose interior edge density is too low.
DOOR_EDGE_MIN = 0.030  # fraction of Canny-edge pixels inside the box (tunable)
# The strongest wall tell: the detector boxes the WHOLE scene (full width AND height)
# and calls it a door. A real door you'd route toward always leaves wall/floor margin
# around it -> it never fills the frame during the scan. Above this area fraction we
# treat a "door" as a whole-wall latch and drop it. Skipped once we're committed to
# approaching a door (go_door/face_target), where the box legitimately grows as we near it.
DOOR_MAX_FRAME_FRAC = 0.80
DOOR_DEBUG = True  # print each door candidate's conf/edge/fill + keep/reject to the terminal
# Second-opinion verification (4-class DoorDetect model: door/handle/cabinet/fridge door).
# Geometry can't separate a lace curtain from a door (curtains are edge-rich, tall,
# door-sized), so weak candidates must be corroborated SEMANTICALLY: either the
# verifier also sees a door there, or it sees a handle inside the box. Curtains have
# neither. Strong candidates (conf above DOOR_STRONG_CONF) pass on their own.
DOOR_STRONG_CONF = 0.60  # accept a door on primary-model confidence alone above this
VERIFY_CONF = 0.30       # verifier runs permissive; it only corroborates, never detects alone
VERIFY_IOU = 0.40        # verifier box overlapping a candidate this much = same object
# A fridge and a door look alike and both models see the same frame. If a door box
# and an object box overlap this much, treat them as the same thing and keep only
# the higher-confidence one (kills door<->refrigerator double-claims).
DOOR_OBJ_IOU = 0.5

# Distance-from-known-size: a door is ~2 m tall, so its pixel height tells us
# roughly how far away it is. HFOV ~60 deg is typical for a laptop/phone webcam;
# focal length in pixels is derived per-frame from the image width.
DOOR_HEIGHT_M = 2.0
ASSUMED_HFOV_DEG = 60.0
STEP_LENGTH_M = 0.75  # average walking step
# One-shot empirical calibration for the whole distance chain (true HFOV, lens
# distortion, box looseness): stand at a KNOWN distance from a door, read the
# [dist] debug line, then set DIST_CAL = true_distance / reported_distance.
DIST_CAL = 1.0
HAND_REACH_STEPS = 3  # within this many steps, ask the user to reach out and feel for the door
_door_path = Path(__file__).resolve().parent.parent / "best.pt"
print(f"Loading door model: {_door_path.name} ...")
_door_model = YOLO(str(_door_path))

# 4-class DoorDetect verifier (door/handle/cabinet door/refrigerator door). Too low
# recall to be the primary detector, but ideal as a SECOND OPINION: corroborate weak
# door candidates and arbitrate door-vs-fridge claims. Optional — absent = old behavior.
_verify_path = (Path(__file__).resolve().parent.parent
                / "door_training" / "runs" / "detect" / "door_yolov8s_4cls" / "weights" / "best.pt")
_verify_model = None
if _verify_path.exists():
    print("Loading 4-class door verifier...")
    _verify_model = YOLO(str(_verify_path))

# Warm up both models so the FIRST real frame isn't stalled by CUDA/kernel init
# (that lag is the long silence at the start of the demo).
print("Warming up models...")
_warm = np.zeros((480, 640, 3), dtype=np.uint8)
_model.predict(_warm, verbose=False)
_door_model.predict(_warm, verbose=False)
if _verify_model is not None:
    _verify_model.predict(_warm, verbose=False)
print("Ready.")

_door_hist: deque = deque(maxlen=WINDOW)  # (region, distance) of strongest door, per frame
_scan_counts: Counter = Counter()        # indicator-class hit counts for the CURRENT room
_prev_small = None                        # last downscaled grayscale frame, for motion gating

# Two-pass, single-concern exploration so prompts never override each other:
#   discover     -> Pass 1: ONE instruction, then scan SILENTLY for a full sweep,
#                   accumulating two flags: indicator_ok and door_seen.
#   go_indicator -> Pass 2a (chosen when indicator_ok): re-confirm the goal's
#                   objects, then announce arrival. Beats doors.
#   go_door      -> Pass 2b (chosen when only door_seen): locate + guide to a door.
# Walking through a near door resets us to discover for the new room.
#   mode       : current phase
#   scan_age   : usable (non-blurred) frames spent in the current scan
#   phase_age  : paces spoken re-prompts (detection runs ~3x/s)
#   near_latch : we got right up to a door (it filled the view)
#   gone       : cycles with no door since near_latch -> infer we walked through
#   door_seen  : a door was confirmed at some point during the current discover
#   ref_heading: compass heading (deg) the user faced when this scan began = "ahead"
#   last_heading / net_rotation: track cumulative turn to know when 360 is done
#   covered    : set of 30-deg sector indices the turn has passed through
#   milestones : which progress lines (25/50/75%) were already spoken
#   sector_objs: {sector -> set of detected classes (incl 'door')} for the summary
#   door_bearings: absolute compass headings (deg) where a door was seen, each
#                  corrected by the door's horizontal position in the frame. Clustered
#                  at scan-end so ONE physical door = one target (not one per sector).
_state = {"goal": None, "mode": "discover", "scan_age": 0,
          "phase": 0, "phase_age": 0, "near_latch": False, "gone": 0,
          "door_seen": False,
          "ref_heading": None, "last_heading": None, "net_rotation": 0.0,
          "covered": set(), "sector_objs": {}, "milestones": set(),
          "door_bearings": [], "ind_bearings": [],
          "target_heading": None, "target_kind": None,
          "skip_scan_prompt": False, "last_door_dist": None,
          "approach_frac": 0.0, "near_age": 0}

REPROMPT = 12          # cycles between spoken re-prompts (~4 s)
ROOM_SCAN_CYCLES = 30  # frames for the go_indicator re-confirm before giving up
DOOR_LOST_CYCLES = 36  # go_door frames with no door re-confirmed -> full rescan (~12 s)

# Sensor-driven discovery: the user does ONE slow guided turn. The phone compass
# tracks rotation so we know when a full circle is done, and tags detections into
# fine BUCKETS sectors for the spatial summary. No rigid stop-and-hold per sector.
BUCKETS = 12                      # 30-deg sectors -> fine coverage, no gaps
BUCKET_DEG = 360.0 / BUCKETS
FULL_TURN_DEG = 350.0             # rotated this far = back to the start direction (full circle)
START_TOL = 25.0                  # within this many deg of the start heading = "back at start"
STILL_MAX = 12.0                  # (fallback path only) motion below = phone steady
FALLBACK_FRAMES = 40             # (no-compass path) usable frames before finishing
FACE_TOL = 25.0                   # within this many deg of the door's heading = "facing it"
DOOR_CLUSTER_DEG = 30.0           # door sightings within this many deg = the SAME door
# Reliability gates for what the scan REPORTS and ACTS ON. Detection isn't as good
# as human eyes: a door or fridge that flickered for a frame or two is noise — never
# speak it, never navigate to it.
DOOR_MIN_SIGHTINGS = 3            # a door cluster needs this many confirmed sightings to be real
OBJ_MIN_SIGHTINGS = 3             # an object mention (per direction) needs this many sightings
DOOR_FILL_FRAC = 0.85  # door height (fraction of frame) meaning "you're at the doorway"
TRANSIT_GONE = 3       # cycles with no door after being at one -> user walked through
# At arm's length a door is a flat panel: the detector still boxes it (huge box) but
# the edge gate rejects it (smooth interior). If we TRACKED an approach down to this
# distance, a saturated door box means "at the door" — walls can't fake that, because
# they were never confirmed as an approaching door first.
NEAR_DOOR_M = 3.5      # last confirmed distance below this = the approach reached the door
APPROACH_FRAC = 0.5    # ...or the confirmed door grew to this frame-height fraction
                       # (distance-scale independent, so calibration can't break it)
# New rooms throw full-frame door candidates too, which would hold the at-the-door
# latch forever (the user already walked through!). Cap how long the latch can hold
# without a properly confirmed door before we infer the transit happened.
NEAR_HOLD_MAX = 20     # ~7 s at ~3 fps


def _reset_scan_fields() -> None:
    _state["scan_age"] = 0
    _state["phase"] = 0
    _state["phase_age"] = 0


def _enter_discover() -> None:
    """Pass 1: one guided 360 turn of a (new) room, collecting flags."""
    _state["mode"] = "discover"
    _state["door_seen"] = False
    _state["ref_heading"] = None
    _state["last_heading"] = None
    _state["net_rotation"] = 0.0
    _state["covered"] = set()
    _state["sector_objs"] = {}
    _state["milestones"] = set()
    _state["door_bearings"] = []
    _state["ind_bearings"] = []
    _state["target_kind"] = None
    _state["skip_scan_prompt"] = False
    _state["last_door_dist"] = None
    _reset_scan_fields()
    _scan_counts.clear()  # fresh room: don't carry indicator evidence across
    _door_hist.clear()


def _signed_from_ref(bucket: int) -> float:
    """Signed degrees of a sector's centre from the START direction (+ right, - left).
    The start direction is our fixed anchor; the summary is described relative to it."""
    deg = (bucket * BUCKET_DEG) % 360.0
    return ((deg + 180.0) % 360.0) - 180.0  # wrap to (-180, 180]


def _signed_from_ref_deg(abs_heading: float) -> float:
    """Signed degrees of an absolute compass heading from the START direction
    (+ right, - left), wrapped to (-180, 180]."""
    ref = _state["ref_heading"] or 0.0
    return ((abs_heading - ref + 180.0) % 360.0) - 180.0


def _cluster_bearings(bearings: list[float]) -> list[tuple[float, int]]:
    """Collapse accumulated sighting bearings into distinct physical objects.

    Sightings of one object land within a few degrees of each other; sightings of two
    different ones are far apart. We sort by angle-from-start and split wherever a
    gap exceeds DOOR_CLUSTER_DEG. Returns [(mean_signed_deg_from_start, n_sightings)],
    so the summary names each object once and we get a precise heading to face."""
    if not bearings:
        return []
    signed = sorted(_signed_from_ref_deg(b) for b in bearings)
    groups: list[list[float]] = [[signed[0]]]
    for s in signed[1:]:
        if s - groups[-1][-1] <= DOOR_CLUSTER_DEG:
            groups[-1].append(s)
        else:
            groups.append([s])
    # A door directly behind the user straddles the +/-180 seam and lands in both the
    # first and last group — merge them across the wrap (shift the top group by -360
    # so the mean comes out right, e.g. [+179, -179] -> -180, not 0).
    if len(groups) > 1 and (signed[0] + 360.0) - signed[-1] <= DOOR_CLUSTER_DEG:
        groups[0] = [s - 360.0 for s in groups.pop()] + groups[0]
    out = []
    for g in groups:
        mean = sum(g) / len(g)
        out.append((((mean + 180.0) % 360.0) - 180.0, len(g)))  # re-wrap to (-180, 180]
    return out


def _cluster_doors() -> list[tuple[float, int]]:
    return _cluster_bearings(_state["door_bearings"])


def _set_door_target(clusters: list[tuple[float, int]]) -> str:
    """Pick the most-sighted door (tie-break: nearest straight-ahead), make it the
    face_target, and return its spoken direction (relative to the start anchor)."""
    mean_signed, _n = max(clusters, key=lambda c: (c[1], -abs(c[0])))
    ref = _state["ref_heading"] or 0.0
    _state["target_heading"] = (ref + mean_signed) % 360.0
    _state["target_kind"] = "door"
    _enter_face_target()
    return _direction(mean_signed)


def _confirmed_from_sectors() -> set:
    """Arrival evidence for the 360 scan, LOCALIZED: a real fridge racks up its
    sightings in one spot (a few adjacent sectors), while detector noise scatters
    around the room. A class is confirmed only when some 90-degree window (a sector
    plus its two neighbours) holds INDICATOR_HITS sightings — scattered one-off
    flickers can never add up to an arrival."""
    per_class: dict[str, dict[int, int]] = {}
    for bucket, counts in _state["sector_objs"].items():
        for c, n in counts.items():
            if c != "door":
                per_class.setdefault(c, {})[bucket] = n
    confirmed = set()
    for c, by_bucket in per_class.items():
        for b in by_bucket:
            window = (by_bucket.get((b - 1) % BUCKETS, 0) + by_bucket.get(b, 0)
                      + by_bucket.get((b + 1) % BUCKETS, 0))
            if window >= INDICATOR_HITS:
                confirmed.add(c)
                break
    return confirmed


def _turn_to(target_heading: float, current_heading: float) -> float:
    """How far to turn from where you face NOW to a heading (+ = right, - = left).
    Needed only to *walk* the user through the turn — the door's position itself
    stays anchored to the start direction."""
    return ((target_heading - current_heading + 180.0) % 360.0) - 180.0


def _track_turn(heading: float | None) -> None:
    """Advance the discover-scan rotation total. Runs on EVERY frame (even blurred
    ones the detector skips) so a fast segment never stalls the full-circle check.
    Ignores large compass glitches."""
    if heading is None or _state["last_heading"] is None:
        return
    d = ((heading - _state["last_heading"] + 180.0) % 360.0) - 180.0
    _state["last_heading"] = heading
    if abs(d) <= 120.0:
        _state["net_rotation"] += d


def _direction(signed_deg: float) -> str:
    """Map a signed angle from 'ahead' to a spoken relative direction."""
    a = signed_deg
    if -30 <= a <= 30:
        return "ahead"
    if 30 < a <= 90:
        return "on your right"
    if 90 < a <= 150:
        return "behind you, to the right"
    if a > 150 or a < -150:
        return "behind you"
    if -150 <= a < -90:
        return "behind you, to the left"
    return "on your left"  # -90 <= a < -30


def _scan_summary(goal: str) -> tuple[str, int]:
    """Spoken spatial map of what the 360 scan found, by direction. Each physical
    door is named once (clustered), and a repeated object in one direction once.
    Returns (text, item_count) so the caller can avoid re-announcing a sole finding."""
    items = []  # doors first, then indicators
    door_dirs = []
    for mean_signed, n in _cluster_doors():
        if n < DOOR_MIN_SIGHTINGS:
            continue  # a flicker, not a door
        where = _direction(mean_signed)
        if where not in door_dirs:
            door_dirs.append(where)
    # "A door nearby" is ONLY for genuinely compass-less scans (no bearings possible).
    # With a compass, doors either have a reliable direction or aren't mentioned.
    if (_state["ref_heading"] is None and not door_dirs
            and sum(c.get("door", 0) for c in _state["sector_objs"].values())
            >= DOOR_MIN_SIGHTINGS):
        items.append("a door nearby")
    items += [f"a door {w}" for w in door_dirs]

    # Object mentions: total sightings per (class, direction); below the minimum it's
    # detector noise and we keep quiet about it.
    dir_counts: dict = {}
    for bucket, counts in _state["sector_objs"].items():
        where = _direction(_signed_from_ref(bucket))
        for c, n in counts.items():
            if c != "door":
                dir_counts[(c, where)] = dir_counts.get((c, where), 0) + n
    items += [f"a {c} {where}" for (c, where), n in dir_counts.items()
              if n >= OBJ_MIN_SIGHTINGS]

    if not items:
        return f"I scanned the whole room but didn't find the {goal} or a door.", 0
    listing = items[0] if len(items) == 1 else ", ".join(items[:-1]) + f", and {items[-1]}"
    return f"Scan complete. I found {listing}.", len(items)


def _finish_discover(goal: str, indicator_ok: bool, confirm_start: bool = False) -> tuple:
    """End of the 360 scan: confirm the user is back at the start, speak the spatial
    summary, then pick the next phase. Returns (guidance, priority)."""
    summary, n_items = _scan_summary(goal)  # built before any _enter_* clears scan state
    if confirm_start:
        # Spoken ONLY when the compass verified the return to the start direction —
        # so every direction in the summary is true of where the user faces RIGHT NOW.
        summary = "You're back where you started — scan complete. " + summary.removeprefix("Scan complete. ")
    if indicator_ok:
        # Branch (a) — ALWAYS beats doors. The 360 scan itself accumulated the evidence
        # (INDICATOR_HITS frames of the goal's objects) — that IS the confirmation.
        # Declare arrival right here, with the directions, and the journey is done.
        _state["mode"] = "arrived"
        if n_items:
            return summary + f" You've reached the {goal}.", True
        lead = ("You're back where you started — scan complete. " if confirm_start
                else "Scan complete. ")
        return lead + f"You've reached the {goal}.", True

    # Branch (a-weak) — indicators were sighted but below the arrival bar. Same
    # directed confirmation we give doors: turn the user toward the sighting and
    # re-check there. Indicator priority holds: this runs BEFORE the door branch.
    ind_clusters = [c for c in _cluster_bearings(_state["ind_bearings"])
                    if c[1] >= OBJ_MIN_SIGHTINGS]
    if ind_clusters:
        mean_signed, _n = max(ind_clusters, key=lambda c: c[1])  # densest sighting area
        ref = _state["ref_heading"] or 0.0
        _state["target_heading"] = (ref + mean_signed) % 360.0
        _state["target_kind"] = "indicator"
        _enter_face_target()
        return (summary + f" That might be the {goal} — let's make sure. Turn toward "
                "it and point the camera there."), True

    # Branch (b) — doors: turn toward the chosen door, then re-confirm it before
    # guiding in. Flickers below DOOR_MIN_SIGHTINGS are noise, never a target.
    clusters = [c for c in _cluster_doors() if c[1] >= DOOR_MIN_SIGHTINGS]
    if clusters:
        where = _set_door_target(clusters)
        if n_items == 1:
            # The summary already named exactly this door — don't announce it twice.
            return (summary + " Turn toward it and point your camera at it, so I can "
                    "guide you in precisely."), True
        return (summary + f" Let's go to the door {where}. Turn that way and point your "
                "camera at it, so I can guide you in precisely."), True

    # No-compass fallback ONLY: doors were confirmed but bearings are impossible.
    # On a compass run, a door without a reliable direction cluster is a flicker —
    # fall through to the rescan instead of vaguely pointing at "the door".
    if (_state["ref_heading"] is None
            and sum(c.get("door", 0) for c in _state["sector_objs"].values()) >= DOOR_MIN_SIGHTINGS):
        _enter_go_door()
        return summary + " Point your camera at the door, and I'll guide you in.", True

    # Nothing useful found. ONE merged utterance (announcement + instruction) — two
    # back-to-back priority lines would cut each other off.
    _enter_discover()
    _state["skip_scan_prompt"] = True
    lead = "You're back where you started. " if confirm_start else ""
    if n_items == 0:
        body = "I couldn't find anything useful in this room. "
    else:  # something was sighted (e.g. a lone fridge glimpse) but no flag was earned
        body = summary.removeprefix("Scan complete. ").rstrip(".") + " — but nothing I can act on yet. "
    return (lead + body + "Let's scan one more time — slowly turn to your right, all "
            "the way around, until you are facing where you started."), True


def _enter_go_indicator() -> None:
    """Pass 2a: re-confirm the goal's indicators with a fresh scan, then arrive."""
    _state["mode"] = "go_indicator"
    _reset_scan_fields()
    _scan_counts.clear()  # fresh evidence so the confirm scan is a real re-check
    _door_hist.clear()


def _enter_face_target() -> None:
    """Pass 2b-pre: actively walk the user through turning to face the chosen door."""
    _state["mode"] = "face_target"
    _reset_scan_fields()


def _enter_go_door() -> None:
    """Pass 2b: locate and guide the user to a door (door-only concern)."""
    _state["mode"] = "go_door"
    _reset_scan_fields()
    _state["phase_age"] = 1  # delay the first "no door yet" so it doesn't double up
    _state["last_door_dist"] = None  # fresh approach: no stale "we were close" memory
    _state["approach_frac"] = 0.0
    _state["near_age"] = 0


def _find_door_phrases() -> list[str]:
    """Nudges while hunting for a door (rotated so re-prompts aren't identical)."""
    return [
        "I don't see a door yet. Keep scanning the room slowly.",
        "Still looking for a door. Keep moving the camera around the room.",
        "No door yet — keep scanning the walls slowly.",
    ]


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


def _steps_word(n: int) -> str:
    return "step" if n == 1 else "steps"


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


def _door_phrase(region: str, dist_m: float | None) -> str:
    """Spoken door call-out: bearing + step distance + a tactile hand cue.
    Within HAND_REACH_STEPS we ask the user to reach out now; farther away we tell
    them how many steps to walk before reaching out to feel for the door."""
    # Definite phrasing on purpose: by the time this speaks, the door has been
    # confirmed — "The door is", never "There's a door" (which sounds like a guess).
    if dist_m is not None and dist_m < 1.0:
        return ("The door is right in front of you. Reach out with your hand to find it."
                if region == "ahead"
                else f"The door is {region}, right next to you. Reach out with your hand to find it.")
    if dist_m is None:
        return f"The door is {region}."

    steps = max(1, round(dist_m / STEP_LENGTH_M))
    unit = _steps_word(steps)
    base = (f"The door is about {steps} {unit} ahead." if region == "ahead"
            else f"The door is {region}, about {steps} {unit} away.")

    if steps <= HAND_REACH_STEPS:
        return base + " You're close — reach out with your hand to find it."
    remaining = steps - HAND_REACH_STEPS
    return base + f" Walk forward, and after about {remaining} {_steps_word(remaining)} reach out with your hand."


class Frame(BaseModel):
    goal: str
    image: str  # data URL (data:image/jpeg;base64,...)
    heading: float | None = None  # phone compass heading in degrees, if available


class GoalText(BaseModel):
    text: str  # raw transcript, e.g. "get me to the kitchen"


@app.get("/")
def index() -> HTMLResponse:
    return HTMLResponse((Path(__file__).parent / "index.html").read_text(encoding="utf-8"))


@app.post("/set_goal")
def set_goal(g: GoalText) -> dict:
    """Resolve a spoken phrase to a canonical room goal (or null if unknown)."""
    goal = resolve_goal(g.text)
    return {"goal": goal, "heard": g.text}


class Spoke(BaseModel):
    text: str
    mode: str = ""


@app.post("/spoke")
def spoke(s: Spoke) -> dict:
    """Client reports a line it actually spoke -> print a transcript to the terminal
    so the whole demo's narration is easy to copy and share."""
    from datetime import datetime
    print(f"[{datetime.now():%H:%M:%S}] [{s.mode or '-':12}] {s.text}", flush=True)
    return {"ok": True}


@app.post("/detect")
def detect(frame: Frame) -> dict:
    goal = resolve_goal(frame.goal) or "kitchen"
    if goal != _state["goal"]:
        _state["goal"] = goal
        _state["near_latch"] = False
        _state["gone"] = 0
        _enter_discover()

    raw = base64.b64decode(frame.image.split(",")[-1])
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "decode failed"}
    h, w = img.shape[:2]

    # --- Motion gate: if the phone is panning too fast the frame is blurred and
    # detections would be unreliable, so skip detection and coach the user to slow
    # down. This also avoids wasting a (latency-costly) inference on a bad frame. ---
    global _prev_small
    small = cv2.cvtColor(cv2.resize(img, (64, 48)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    motion = float(np.abs(small - _prev_small).mean()) if _prev_small is not None else 0.0
    _prev_small = small
    if motion > MOTION_MAX:
        # Keep the turn total advancing even though we skip detection on this blurred
        # frame, so a fast segment doesn't stall the full-circle completion.
        if _state["mode"] == "discover":
            _track_turn(frame.heading)
        return {
            "goal": goal, "mode": _state["mode"], "boxes": [], "arrived": False,
            "matched": [], "phrase": "", "priority": False,
            "guidance": "Slow down. Move the phone slowly.",
        }

    primary, secondary = indicators_for(goal)
    indicators = set(primary) | set(secondary)

    # COCO indicator model (class-filtered to this goal's indicators -> faster, and
    # the overlay stays clean). Collect raw detections; we resolve door<->object
    # overlaps before committing them.
    wanted_ids = [_name_to_id[c] for c in indicators if c in _name_to_id]
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

    # Door model (single-class 'door' today; 4-class compatible). 'door' = navigable;
    # 'refrigerator door' is folded in as a fridge sighting; handle/cabinet ignored.
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
        if dens >= DOOR_EDGE_MIN or _verify_model is None:
            # Textured interior (frame/seams/handle visible): confidence or
            # corroboration passes it, as before.
            ok = not fridge_like and (_verify_model is None or conf >= DOOR_STRONG_CONF or corro)
        else:
            # Smooth interior: a blank wall OR a plain door in dim light — confidence
            # CANNOT tell them apart (walls score 0.9 too), so only semantic proof
            # (the verifier seeing a door or a handle there) passes it.
            ok = not fridge_like and corro
        if DOOR_DEBUG:
            print(f"[door] conf={conf:.2f} edge={dens:.3f} corro={corro} "
                  f"fridge_like={fridge_like} -> {'KEEP' if ok else 'reject (verify)'}",
                  flush=True)
        if ok:
            door_dets.append((conf, xy))
        # NOTE: a fridge_like rejection does NOT become a refrigerator sighting.
        # The verifier hallucinates 'refrigerator door' on blank walls, and feeding
        # those into the indicator evidence caused fake fridges in the arrival
        # summary. Vetoing the door is safe; claiming a fridge is not — real
        # fridges are detected by the COCO model on its own.

    # A COCO 'refrigerator' claim that sits on a door candidate is suspect — a real
    # door at an angle often reads as a fridge. Keep it ONLY if the verifier saw a
    # 'refrigerator door' there; otherwise it IS the door (kills false kitchen arrivals).
    if _verify_model is not None and door_cands:
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
        for j, (dcf, dxy) in enumerate(door_dets):
            if _iou(oxy, dxy) >= DOOR_OBJ_IOU:
                if dcf >= ocf:
                    drop_obj.add(i)
                else:
                    drop_door.add(j)
    obj_dets = [d for i, d in enumerate(obj_dets) if i not in drop_obj]
    door_dets = [d for j, d in enumerate(door_dets) if j not in drop_door]

    boxes = []
    seen = set()
    for cls, conf, (x1, y1, x2, y2) in obj_dets:
        seen.add(cls)
        boxes.append({
            "cls": cls, "conf": round(conf, 2),
            "box": [x1 / w, y1 / h, x2 / w, y2 / h],  # normalized
            "indicator": cls in indicators,
        })
    door_areas = []
    for conf, (x1, y1, x2, y2) in door_dets:
        door_areas.append((x1, y1, x2, y2))
        boxes.append({
            "cls": "door", "conf": round(conf, 2),
            "box": [x1 / w, y1 / h, x2 / w, y2 / h],
            "indicator": False, "door": True,
        })

    # Accumulate indicator evidence across the room-scan: seen in >= INDICATOR_HITS
    # frames = confirmed. During the compass 360 the evidence must additionally be
    # LOCALIZED (one 90-deg window), so scattered false hits can't sum to an arrival.
    _scan_counts.update(seen & indicators)
    if _state["mode"] == "discover" and _state["ref_heading"] is not None:
        confirmed = _confirmed_from_sectors()
    else:
        confirmed = {c for c, n in _scan_counts.items() if n >= INDICATOR_HITS}
    if door_areas:
        bx = max(door_areas, key=lambda d: (d[2] - d[0]) * (d[3] - d[1]))
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

    # --- Doorway transit (ONLY while heading to a door). Phase 1 (discover) is
    # pure scanning: we never react to a transit there. If, while in go_door, we
    # got right up to a door (it filled the view) and then it's gone for a few
    # frames, infer the user walked through it -> rescan the new room. ---
    transit = False
    just_near = False  # near_latch turned on THIS frame -> announce "you're at the door"
    if _state["mode"] == "go_door" and door_confirmed:
        # Track how big the confirmed door got during this approach (scale-free).
        _state["approach_frac"] = max(_state["approach_frac"], cur_frac)
    # The saturated close-range box only counts if a tracked approach already got us
    # near this door — a wall pointed at mid-walk was never a confirmed approach.
    # Two near signals: metric distance, OR the confirmed door having grown to fill
    # half the frame (immune to distance-calibration changes).
    at_door_box = near_box and (
        (_state["last_door_dist"] is not None and _state["last_door_dist"] <= NEAR_DOOR_M)
        or _state["approach_frac"] >= APPROACH_FRAC)
    if _state["mode"] != "go_door":
        _state["near_latch"] = False
        _state["gone"] = 0
        _state["near_age"] = 0
    elif (door_confirmed and cur_frac >= DOOR_FILL_FRAC) or at_door_box:
        just_near = not _state["near_latch"]
        _state["near_latch"] = True
        _state["gone"] = 0
        # New rooms also throw saturated door candidates, which would hold this latch
        # forever AFTER the user walked through. Only a properly confirmed door resets
        # the hold timer; saturated-box frames age it until we infer the transit.
        _state["near_age"] = 0 if door_confirmed else _state["near_age"] + 1
        if _state["near_age"] >= NEAR_HOLD_MAX:
            transit = True
            _state["near_latch"] = False
            _state["gone"] = 0
            _state["near_age"] = 0
            _enter_discover()
    elif _state["near_latch"] and not door_confirmed:
        _state["gone"] += 1
        _state["near_age"] += 1
        if _state["gone"] >= TRANSIT_GONE or _state["near_age"] >= NEAR_HOLD_MAX:
            transit = True
            _state["near_latch"] = False
            _state["gone"] = 0
            _state["near_age"] = 0
            _enter_discover()
    elif door_confirmed:
        _state["gone"] = 0  # door still in view but not close — not a transit
        _state["near_age"] = 0

    result = evaluate_arrival(goal, confirmed)
    matched = result["matched_primary"] + result["matched_secondary"]
    indicator_ok = result["arrived"]  # >=1 primary or >=2 secondary, accumulated

    # --- Spoken guidance --------------------------------------------------------
    # `priority` lines (sector stops, phase changes, arrival, summary) may interrupt;
    # ambient nudges (hold-steady, keep-turning, door countdown) never interrupt.
    guidance = ""
    priority = False
    announce_arrival = False

    if transit:
        # Walked through a doorway -> Pass 1 for the new room (discover set by transit).
        if frame.heading is not None:
            _state["ref_heading"] = frame.heading
            _state["last_heading"] = frame.heading
        _state["skip_scan_prompt"] = True  # instruction is in THIS line; don't repeat it
        guidance = ("You've gone through the doorway. Take two or three steps into the "
                    "room, then slowly turn to your right, all the way around, back to "
                    "where you started, so I can scan this room.")
        priority = True

    elif _state["mode"] == "discover":
        heading = frame.heading
        if heading is None:
            # Fallback (no compass, e.g. a laptop): one slow steady-capture pass.
            if _state["last_heading"] is None:
                _state["last_heading"] = 0.0  # mark started
                if _state["skip_scan_prompt"]:
                    _state["skip_scan_prompt"] = False  # instruction already in the rescan line
                else:
                    guidance = (f"Looking for the {goal}. Let's scan the room — slowly "
                                "pan all the way around, pausing a moment as you go.")
                    priority = True
            elif motion <= STILL_MAX:
                _state["scan_age"] += 1
                objs = _state["sector_objs"].setdefault(0, Counter())
                objs.update(seen & indicators)
                if door_confirmed:
                    objs["door"] += 1
                    _state["door_seen"] = True
                if _state["scan_age"] >= FALLBACK_FRAMES:
                    guidance, priority = _finish_discover(goal, indicator_ok)
        elif _state["ref_heading"] is None:
            # First sensor reading -> set the START direction (our anchor) and begin.
            _state["ref_heading"] = heading
            _state["last_heading"] = heading
            if _state["skip_scan_prompt"]:
                _state["skip_scan_prompt"] = False  # instruction already spoken with the rescan line
            else:
                guidance = (f"Looking for the {goal}. Let's scan the room — slowly turn "
                            "to your right, all the way around, until you are facing "
                            "where you started.")
                priority = True
        else:
            _track_turn(heading)  # advance the full-circle total (non-blurred frame)
            rel = (heading - _state["ref_heading"]) % 360.0
            bucket = int(rel // BUCKET_DEG) % BUCKETS
            _state["covered"].add(bucket)
            objs = _state["sector_objs"].setdefault(bucket, Counter())
            objs.update(seen & indicators)  # sighting COUNTS -> reliability gating later
            # Record TRUE bearings (camera heading + offset within the frame) for
            # everything that matters: doors AND goal indicators.
            for icls, _icf, ixy in obj_dets:
                if icls in indicators:
                    icx = ((ixy[0] + ixy[2]) / 2) / w
                    _state["ind_bearings"].append(
                        (heading + (icx - 0.5) * ASSUMED_HFOV_DEG) % 360.0)
            if door_confirmed:
                _state["door_seen"] = True
                # Count the sighting and its bearing from the SAME frames (ones with an
                # actual box) so the reliability threshold and the direction clusters
                # always agree — a count without a bearing sent us down the
                # no-compass "door nearby" path by mistake.
                if door_cx_frac is not None:
                    objs["door"] += 1
                    bearing = (heading + (door_cx_frac - 0.5) * ASSUMED_HFOV_DEG) % 360.0
                    _state["door_bearings"].append(bearing)
            # Progress context by how far around they've turned (ambient, once each).
            prog = abs(_state["net_rotation"])
            ms = _state["milestones"]
            if prog >= 270 and "75" not in ms:
                ms.add("75"); guidance = "Almost back to where you started, keep turning."
            elif prog >= 180 and "50" not in ms:
                ms.add("50"); guidance = "Halfway around, keep turning."
            elif prog >= 90 and "25" not in ms:
                ms.add("25"); guidance = "Good, keep turning to your right."
            # Done ONLY when a full circle of rotation has accumulated AND the compass
            # confirms they're facing the start direction again. Never on bucket
            # coverage alone — compass noise can fake that early, ending the scan
            # mid-turn with directions computed from a broken premise.
            if abs(_state["net_rotation"]) >= FULL_TURN_DEG:
                if abs(_turn_to(_state["ref_heading"], heading)) <= START_TOL:
                    guidance, priority = _finish_discover(goal, indicator_ok, confirm_start=True)
                else:
                    _state["phase_age"] += 1
                    if "back" not in ms or _state["phase_age"] >= REPROMPT:
                        ms.add("back")
                        _state["phase_age"] = 0
                        guidance = "Almost done — keep turning until you face where you started."

    elif _state["mode"] == "go_indicator":
        # Pass 2a: re-confirm the goal's objects, then arrive. (Doors ignored here.)
        if indicator_ok:
            announce_arrival = True
        else:
            _state["scan_age"] += 1
            if _state["phase_age"] >= REPROMPT:
                guidance = f"Keep the camera there, panning slowly, while I confirm the {goal}."  # ambient
                _state["phase_age"] = 0
            _state["phase_age"] += 1
            if _state["scan_age"] >= ROOM_SCAN_CYCLES:
                # Couldn't re-confirm -> false alarm. The indicator had its chance;
                # if the scan also flagged a door, take it (door bearings survive the
                # confirm phase) — otherwise a full rescan.
                door_clusters = [c for c in _cluster_doors() if c[1] >= DOOR_MIN_SIGHTINGS]
                if door_clusters and frame.heading is not None:
                    _set_door_target(door_clusters)
                    guidance = (f"I couldn't confirm the {goal} here. Let's take the "
                                "door instead — I'll help you turn to face it.")
                else:
                    _enter_discover()
                    _state["skip_scan_prompt"] = True
                    guidance = (f"I couldn't confirm the {goal}. Let's scan the room again — "
                                "slowly turn to your right, all the way around, until you "
                                "are facing where you started.")
                priority = True

    elif _state["mode"] == "face_target":
        # Phase 2 opener (both branches): walk the user through turning until they
        # face the flagged target, THEN run the focused confirmation scan there.
        kind = _state["target_kind"] or "door"
        label = "door" if kind == "door" else goal
        h = frame.heading
        if h is None or _state["target_heading"] is None:
            # No compass -> skip the guided turn, go straight to the confirm phase.
            if kind == "door":
                _enter_go_door()
                guidance = "Turn toward the door, and I'll guide you in."
            else:
                _enter_go_indicator()
                guidance = f"Point the camera where you saw the {goal}, and hold it there."
            priority = True
        else:
            turn = _turn_to(_state["target_heading"], h)
            if abs(turn) <= FACE_TOL:
                if kind == "door":
                    _enter_go_door()
                    guidance = "The door should be right ahead of you now. Let's go to it."
                else:
                    _enter_go_indicator()
                    guidance = (f"You should be facing the {goal} now. Hold the camera "
                                "there and pan slowly while I confirm.")
                priority = True
            else:
                side = "right" if turn > 0 else "left"
                guidance = f"Turn slowly to your {side} to face the {label}."  # ambient, re-evaluated live

    elif _state["mode"] == "arrived":
        pass  # journey complete — arrival is reported via the arrived/phrase fields below

    else:  # go_door — Pass 2b: door-only concern.
        if just_near:
            guidance = "You're right at the door. Reach out, open it, and walk through."
            priority = True
        elif _state["near_latch"] and not door_confirmed:
            # Standing at the door (detector saturated by the panel). Stay quiet —
            # no "no door yet" nudges, no lost-door timer; the transit check is watching.
            _state["scan_age"] = 0
        elif door_confirmed and region:
            _state["scan_age"] = 0  # door in sight -> confirmation holds
            guidance = _door_phrase(region, door_dist)  # ambient (updates as you approach)
        else:
            # Confirmation scan failing: like branch (a), a flag that can't be
            # re-confirmed within a window means a full rescan, not endless nudging.
            _state["scan_age"] += 1
            if _state["scan_age"] >= DOOR_LOST_CYCLES:
                _enter_discover()
                _state["skip_scan_prompt"] = True
                guidance = ("I can't find that door anymore. Let's scan the room again — "
                            "slowly turn to your right, all the way around, until you "
                            "are facing where you started.")
                priority = True
            else:
                if _state["phase_age"] == 0:
                    _state["phase"] += 1
                    guidance = _find_door_phrases()[_state["phase"] % 3]  # ambient
                _state["phase_age"] += 1
                if _state["phase_age"] >= REPROMPT:
                    _state["phase_age"] = 0

    phrase = ""
    if _state["mode"] == "arrived":
        announce_arrival = True
        # On the frame the scan just finished, guidance holds the composed
        # summary + "You've reached the kitchen." line — that IS the arrival phrase.
        phrase = guidance or arrival_phrase(goal, result)
        guidance = ""
    elif announce_arrival:
        phrase = arrival_phrase(goal, result)

    return {
        "goal": goal,
        "mode": _state["mode"],
        "boxes": boxes,
        "arrived": announce_arrival,
        "matched": matched,
        "phrase": phrase,
        "guidance": guidance,
        "priority": priority,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
