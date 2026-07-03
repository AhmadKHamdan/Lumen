"""Lumen exploration demo — FastAPI server (thin orchestration layer).

The browser captures the rear camera + compass and speaks via the Web Speech API.
This server runs the vision + decision logic, which lives in the `lumen/` package:

    decode -> motion gate -> perception -> controller -> spoken guidance

All the real logic is in lumen/ (config, models, state, geometry, perception,
obstacles, controller). `detect()` below is just the per-frame pipeline that wires
them together in order.

Run:
    python webdemo/server.py
Then open http://localhost:8000 (or an ngrok https URL on a phone) and tap Start.
"""
from __future__ import annotations

import base64

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pathlib import Path

# Importing the lumen package loads the models (once).
from lumen import controller, geometry, obstacles, perception, state
from lumen.config import MOTION_COACH_FRAMES, MOTION_MAX
from lumen.goals import indicators_for

app = FastAPI()


# --- request/response models -----------------------------------------------------

class Frame(BaseModel):
    goal: str
    image: str  # data URL (data:image/jpeg;base64,...)
    heading: float | None = None  # phone compass heading in degrees, if available


class GoalText(BaseModel):
    text: str  # raw transcript, e.g. "get me to the kitchen"


class Spoke(BaseModel):
    text: str
    mode: str = ""


# --- routes ----------------------------------------------------------------------

@app.get("/")
def index() -> HTMLResponse:
    return HTMLResponse((Path(__file__).parent / "index.html").read_text(encoding="utf-8"))


@app.post("/set_goal")
def set_goal(g: GoalText) -> dict:
    """Resolve a spoken phrase to a canonical room goal (or null if unknown).
    Re-speaking a goal after an arrival restarts the journey."""
    goal = controller.new_goal_request(g.text)
    return {"goal": goal, "heard": g.text}


@app.post("/spoke")
def spoke(s: Spoke) -> dict:
    """Client reports a line it actually spoke -> print a transcript to the terminal
    so the whole demo's narration is easy to copy and share."""
    from datetime import datetime
    print(f"[{datetime.now():%H:%M:%S}] [{s.mode or '-':12}] {s.text}", flush=True)
    return {"ok": True}


def _decode(data_url: str):
    """Decode a base64 data-URL frame to a BGR image (or None on failure)."""
    raw = base64.b64decode(data_url.split(",")[-1])
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


@app.post("/detect")
def detect(frame: Frame) -> dict:
    goal = controller.handle_goal(frame.goal)

    img = _decode(frame.image)
    if img is None:
        return {"error": "decode failed"}
    h, w = img.shape[:2]

    # --- Motion gate: if the phone is panning too fast the frame is blurred and
    # detections would be unreliable, so skip detection and coach the user to slow
    # down. This also avoids wasting a (latency-costly) inference on a bad frame. ---
    small = cv2.cvtColor(cv2.resize(img, (64, 48)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    motion = float(np.abs(small - state.prev_small).mean()) if state.prev_small is not None else 0.0
    state.prev_small = small
    if motion > MOTION_MAX:
        # Keep the turn total advancing even though we skip detection on this blurred
        # frame, so a fast segment doesn't stall the full-circle completion.
        if state._state["mode"] == "discover":
            geometry._track_turn(frame.heading)
        # One blurred frame (autofocus hunt, exposure change) is not the user moving
        # fast — skip it silently, and only COACH after several consecutive ones.
        state._state["fast_frames"] += 1
        coach = state._state["fast_frames"] >= MOTION_COACH_FRAMES
        return {
            "goal": goal, "mode": state._state["mode"], "boxes": [], "arrived": False,
            "matched": [], "phrase": "", "priority": False,
            "guidance": "Slow down. Move the phone slowly." if coach else "",
        }
    state._state["fast_frames"] = 0

    primary, secondary = indicators_for(goal)
    indicators = set(primary) | set(secondary)

    # Perception: one frame -> trustworthy detections.
    obj_dets, obstacle_dets = perception.perceive_objects(img, w, h, indicators)
    obj_dets, door_dets, near_box, vdoor = perception.process_doors(img, w, h, obj_dets)
    boxes, seen = perception.build_boxes(obj_dets, door_dets, indicators, w, h)

    # Interpretation: arrival evidence, door geometry, transit, obstacles.
    confirmed = controller.accumulate_indicator_evidence(seen, indicators)
    dg = perception.door_geometry(door_dets, vdoor, w, h)
    transit, just_near = controller.detect_transit(near_box, dg.door_confirmed,
                                                   dg.cur_frac, motion)
    obst_guidance, obst_priority, obst_blocking = obstacles.evaluate(obstacle_dets, img, w, h)

    # Decision: the state machine produces what to say this frame.
    guidance, priority, announce_arrival, phrase, matched = controller.step(
        goal=goal, heading=frame.heading, motion=motion, w=w, seen=seen,
        indicators=indicators, obj_dets=obj_dets, door_confirmed=dg.door_confirmed,
        door_cx_frac=dg.door_cx_frac, door_corro=dg.corro, region=dg.region,
        door_dist=dg.door_dist, transit=transit, just_near=just_near, confirmed=confirmed,
        obst_guidance=obst_guidance, obst_priority=obst_priority, obst_blocking=obst_blocking)

    return {
        "goal": goal,
        "mode": state._state["mode"],
        "boxes": boxes,
        "arrived": announce_arrival,
        "matched": matched,
        "phrase": phrase,
        "guidance": guidance,
        "priority": priority,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
