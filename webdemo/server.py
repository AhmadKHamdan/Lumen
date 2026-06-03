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
# builds evidence rather than forgetting it after 3 frames.
INDICATOR_HITS = 3
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

# Custom single-class door detector (trained, mAP50 ~0.95). Resolved relative to
# the repo root so it works regardless of the shell's cwd.
DOOR_CONF = 0.45
# Shape sanity check: doors aren't extremely wide. Rejecting very wide boxes cuts
# some wall/furniture false positives without hurting normal door recall.
DOOR_MAX_WH = 1.4  # max width/height ratio to still count as a door

# Distance-from-known-size: a door is ~2 m tall, so its pixel height tells us
# roughly how far away it is. HFOV ~60 deg is typical for a laptop/phone webcam;
# focal length in pixels is derived per-frame from the image width.
DOOR_HEIGHT_M = 2.0
ASSUMED_HFOV_DEG = 60.0
STEP_LENGTH_M = 0.75  # average walking step
HAND_REACH_STEPS = 3  # within this many steps, ask the user to reach out and feel for the door
_door_path = Path(__file__).resolve().parent.parent / "best.pt"
print(f"Loading door model: {_door_path.name} ...")
_door_model = YOLO(str(_door_path))

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
_state = {"goal": None, "mode": "discover", "scan_age": 0,
          "phase": 0, "phase_age": 0, "near_latch": False, "gone": 0,
          "door_seen": False}

REPROMPT = 12          # cycles between spoken re-prompts (~4 s)
ROOM_SCAN_CYCLES = 15  # ~5 s of scanning a room before we give up and seek a door
DOOR_FILL_FRAC = 0.85  # door height (fraction of frame) meaning "you're at the doorway"
TRANSIT_GONE = 3       # cycles with no door after being at one -> user walked through


def _scan_reminders() -> list[str]:
    """Neutral 'keep scanning' nudges — no goal/door talk during discovery.
    Rotated so consecutive reminders differ (the client de-dupes identical lines)."""
    return [
        "Keep scanning the room slowly.",
        "Keep panning slowly, all the way around.",
        "Slowly, a little at a time.",
    ]


def _reset_scan_fields() -> None:
    _state["scan_age"] = 0
    _state["phase"] = 0
    _state["phase_age"] = 0


def _enter_discover() -> None:
    """Pass 1: silent comprehensive scan of a (new) room, collecting flags."""
    _state["mode"] = "discover"
    _state["door_seen"] = False
    _reset_scan_fields()
    _scan_counts.clear()  # fresh room: don't carry indicator evidence across
    _door_hist.clear()


def _enter_go_indicator() -> None:
    """Pass 2a: re-confirm the goal's indicators with a fresh scan, then arrive."""
    _state["mode"] = "go_indicator"
    _reset_scan_fields()
    _scan_counts.clear()  # fresh evidence so the confirm scan is a real re-check
    _door_hist.clear()


def _enter_go_door() -> None:
    """Pass 2b: locate and guide the user to a door (door-only concern)."""
    _state["mode"] = "go_door"
    _reset_scan_fields()
    _state["phase_age"] = 1  # delay the first "no door yet" so it doesn't double up


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


def _door_distance_m(px_height: float, img_w_px: int) -> float | None:
    """Estimate door distance (m) from its pixel height via a pinhole model.
    focal_px = (img_width/2) / tan(HFOV/2); distance = real_height * f / px_height."""
    if px_height <= 0:
        return None
    f_px = (img_w_px / 2) / math.tan(math.radians(ASSUMED_HFOV_DEG / 2))
    return DOOR_HEIGHT_M * f_px / px_height


def _steps_word(n: int) -> str:
    return "step" if n == 1 else "steps"


def _door_phrase(region: str, dist_m: float | None) -> str:
    """Spoken door call-out: bearing + step distance + a tactile hand cue.
    Within HAND_REACH_STEPS we ask the user to reach out now; farther away we tell
    them how many steps to walk before reaching out to feel for the door."""
    if dist_m is not None and dist_m < 1.0:
        return ("There's a door right in front of you. Reach out with your hand to find it."
                if region == "ahead"
                else f"There's a door {region}, right next to you. Reach out with your hand to find it.")
    if dist_m is None:
        return f"There's a door {region}."

    steps = max(1, round(dist_m / STEP_LENGTH_M))
    unit = _steps_word(steps)
    base = (f"There's a door about {steps} {unit} ahead." if region == "ahead"
            else f"There's a door {region}, about {steps} {unit} away.")

    if steps <= HAND_REACH_STEPS:
        return base + " You're close — reach out with your hand to find it."
    remaining = steps - HAND_REACH_STEPS
    return base + f" Walk forward, and after about {remaining} {_steps_word(remaining)} reach out with your hand."


class Frame(BaseModel):
    goal: str
    image: str  # data URL (data:image/jpeg;base64,...)


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
        return {
            "goal": goal, "mode": _state["mode"], "boxes": [], "arrived": False,
            "matched": [], "phrase": "", "priority": False,
            "guidance": "Slow down. Move the phone slowly.",
        }

    primary, secondary = indicators_for(goal)
    indicators = set(primary) | set(secondary)

    res = _model.predict(img, verbose=False, conf=CONF)[0]
    boxes = []
    seen = set()
    for b in res.boxes:
        cls = _names[int(b.cls[0])]
        conf = float(b.conf[0])
        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
        seen.add(cls)
        boxes.append({
            "cls": cls,
            "conf": round(conf, 2),
            "box": [x1 / w, y1 / h, x2 / w, y2 / h],  # normalized
            "indicator": cls in indicators,
        })

    # Accumulate indicator evidence across the whole room-scan (not a rolling
    # window): an indicator seen in >= INDICATOR_HITS frames this room is confirmed.
    _scan_counts.update(seen & indicators)
    confirmed = {c for c, n in _scan_counts.items() if n >= INDICATOR_HITS}

    # --- Door detection (custom model). Draw every plausible door; pick the most
    # prominent (largest = nearest) and confirm it over 2-of-3 frames. ---
    dres = _door_model.predict(img, verbose=False, conf=DOOR_CONF)[0]
    door_areas = []
    for b in dres.boxes:
        conf = float(b.conf[0])
        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
        bw, bh = x2 - x1, y2 - y1
        if bh <= 0 or bw / bh > DOOR_MAX_WH:
            continue  # too wide to be a door (likely a wall/furniture false positive)
        door_areas.append((x1, y1, x2, y2))
        boxes.append({
            "cls": "door",
            "conf": round(conf, 2),
            "box": [x1 / w, y1 / h, x2 / w, y2 / h],
            "indicator": False,
            "door": True,
        })
    if door_areas:
        bx = max(door_areas, key=lambda d: (d[2] - d[0]) * (d[3] - d[1]))
        region = _door_region(((bx[0] + bx[2]) / 2) / w)
        dist_m = _door_distance_m(bx[3] - bx[1], w)
        cur_frac = (bx[3] - bx[1]) / h  # how much of the frame height the door fills
    else:
        region, dist_m, cur_frac = None, None, 0.0
    _door_hist.append((region, dist_m) if region else None)
    confirmed_doors = [d for d in _door_hist if d]
    door_confirmed = len(confirmed_doors) >= MIN_HITS
    # Steady the step count with the median distance over the window (less jitter).
    _dists = sorted(d[1] for d in confirmed_doors if d[1] is not None)
    door_dist = _dists[len(_dists) // 2] if _dists else None

    # --- Doorway transit: if we got right up to a door (it filled the view) and
    # then it's gone for a few frames, infer the user walked through it -> rescan
    # the new room rather than instantly pointing at the next door. ---
    transit = False
    if door_confirmed and cur_frac >= DOOR_FILL_FRAC:
        _state["near_latch"] = True
        _state["gone"] = 0
    elif _state["near_latch"] and not door_confirmed:
        _state["gone"] += 1
        if _state["gone"] >= TRANSIT_GONE:
            transit = True
            _state["near_latch"] = False
            _state["gone"] = 0
            _enter_discover()
    elif door_confirmed:
        _state["gone"] = 0  # door still in view but not close — not a transit

    result = evaluate_arrival(goal, confirmed)
    matched = result["matched_primary"] + result["matched_secondary"]
    indicator_ok = result["arrived"]  # >=1 primary or >=2 secondary, accumulated

    # While discovering, just record whether a door showed up — don't act on it yet.
    if _state["mode"] == "discover" and door_confirmed:
        _state["door_seen"] = True

    # --- Spoken guidance: two passes, one concern at a time -----------------
    # `priority` lines (phase changes, arrival) may interrupt; ambient nudges
    # (scan reminders, slow-down, door countdown) never interrupt on the client.
    guidance = ""
    priority = False
    announce_arrival = False

    if transit:
        # Just walked through a doorway -> Pass 1 for the new room (discover set by transit).
        _state["scan_age"] = 1  # skip the duplicate entry line next cycle
        guidance = ("You've gone through the doorway. Take a step or two in. "
                    "Now let's scan this room slowly.")
        priority = True

    elif _state["mode"] == "discover":
        # Pass 1: ONE instruction, then ONLY neutral scan reminders. No goal/door talk.
        if _state["scan_age"] == 0:
            guidance = ("Okay, let's scan the room. "
                        "Pan your phone slowly, all the way around.")
            priority = True
        elif _state["phase_age"] >= REPROMPT:
            _state["phase"] += 1
            guidance = _scan_reminders()[_state["phase"] % 3]  # ambient
            _state["phase_age"] = 0
        _state["scan_age"] += 1
        _state["phase_age"] += 1
        # End of the sweep -> pick the single next concern from the flags.
        if _state["scan_age"] >= ROOM_SCAN_CYCLES:
            if indicator_ok:
                things = matched[0] if len(matched) == 1 else f"{matched[0]} and {matched[1]}"
                _enter_go_indicator()
                guidance = (f"I can see signs of the {goal} — a {things}. "
                            f"Let me make sure. Keep panning slowly.")
            elif _state["door_seen"]:
                _enter_go_door()
                guidance = (f"I don't see the {goal} in this room, but there is a door. "
                            f"Let's head to it. Keep panning slowly to find the door.")
            else:
                _enter_discover()
                guidance = (f"I couldn't find the {goal} or a door yet. "
                            f"Let's scan the room again, slowly.")
            priority = True

    elif _state["mode"] == "go_indicator":
        # Pass 2a: re-confirm the goal's objects, then arrive. (Doors ignored here.)
        if indicator_ok:
            announce_arrival = True
        else:
            _state["scan_age"] += 1
            if _state["phase_age"] >= REPROMPT:
                guidance = f"Almost there — keep panning slowly to confirm the {goal}."  # ambient
                _state["phase_age"] = 0
            _state["phase_age"] += 1
            if _state["scan_age"] >= ROOM_SCAN_CYCLES:
                # Couldn't re-confirm -> false alarm; start a fresh discovery scan.
                _enter_discover()
                guidance = f"I lost sight of the {goal}. Let's scan the room again, slowly."
                priority = True

    else:  # go_door — Pass 2b: door-only concern.
        if door_confirmed and region:
            guidance = _door_phrase(region, door_dist)  # ambient (updates as you approach)
        else:
            if _state["phase_age"] == 0:
                _state["phase"] += 1
                guidance = _find_door_phrases()[_state["phase"] % 3]  # ambient
            _state["phase_age"] += 1
            if _state["phase_age"] >= REPROMPT:
                _state["phase_age"] = 0

    return {
        "goal": goal,
        "mode": _state["mode"],
        "boxes": boxes,
        "arrived": announce_arrival,
        "matched": matched,
        "phrase": arrival_phrase(goal, result) if announce_arrival else "",
        "guidance": guidance,
        "priority": priority,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
