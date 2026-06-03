"""M1 live test — semantic room arrival.

Proves Milestone M1 end to end with a real camera: name a room goal, point the
camera at its indicator objects, and hear "we've reached the {room}". Uses the
base COCO YOLOv8n (no door model needed — M1 is about recognising a room from
its objects, not doors).

Pipeline per frame:
    webcam -> COCO YOLOv8n -> 2-of-3 temporal confirmation
           -> navigation.goal_map.evaluate_arrival -> spoken announcement

Run from repo root:
    python scripts/m1_arrival_demo.py                 # goal defaults to "kitchen"
    python scripts/m1_arrival_demo.py --goal bathroom

Keys in the video window:  r = re-arm (announce again)   q = quit
"""
from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from collections import Counter, deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import cv2  # noqa: E402

from navigation import (  # noqa: E402
    resolve_goal,
    indicators_for,
    evaluate_arrival,
    arrival_phrase,
    is_known_goal,
)

# Match the navigation engine's constants.
CONF_THRESHOLD = 0.5
WINDOW = 3          # temporal window
MIN_HITS = 2        # 2-of-3 confirmation


class SpeakingTTS:
    """Speak on a worker thread; print as a fallback if pyttsx3 is missing."""

    def __init__(self) -> None:
        self._q: "queue.Queue[str]" = queue.Queue()
        try:
            import pyttsx3  # noqa: F401
            self._enabled = True
            threading.Thread(target=self._run, daemon=True).start()
        except Exception as e:
            print(f"[TTS] pyttsx3 unavailable ({e}); printing only.")
            self._enabled = False

    def _run(self) -> None:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 175)
        while True:
            text = self._q.get()
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"[TTS] speak failed: {e}")

    def say(self, text: str) -> None:
        print(f"\n  🔊 {text}")
        if self._enabled:
            self._q.put(text)


def _grab_usable(cap):
    """Warm up a capture; return the first non-black frame it delivers, else None.
    Waits up to ~1.5 s (cameras can need a moment to start, or to be released by
    another app), but fast-fails after a few failed reads so a dead/hanging backend
    (MSMF can take ~10 s per failed grab) doesn't stall the scan."""
    fails = 0
    for _ in range(20):
        ok, frame = cap.read()
        if not ok:
            fails += 1
            if fails >= 3:
                return None  # backend/index genuinely can't deliver
            time.sleep(0.05)
            continue
        if frame is not None and float(frame.mean()) > 5.0:
            return frame  # got real video — accept immediately
        time.sleep(0.05)  # opened but still black; give it a moment
    return None


def _open_camera(index: int = 0):
    """Open the webcam. Many laptop cams (e.g. this HP 5MP) are MediaFoundation
    (MIPI) devices: they need the MSMF backend plus a short init delay before
    frames are ready, while legacy DSHOW returns black on them. Try MSMF first,
    then DSHOW as a fallback for other machines/webcams."""
    # MIPI/MF cams (HP 5MP) intermittently fail to START streaming, but stream
    # fine once open — so retry the open patiently. MSMF first (DSHOW returns
    # black on these cams).
    plan = [("MSMF", cv2.CAP_MSMF, 1.5, 10), ("DSHOW", cv2.CAP_DSHOW, 0.3, 3)]
    for name, be, init_delay, attempts in plan:
        for attempt in range(attempts):
            cap = cv2.VideoCapture(index, be)
            if cap.isOpened():
                time.sleep(init_delay)  # let the camera pipeline spin up
                if _grab_usable(cap) is not None:
                    print(f"[camera] connected via {name} (attempt {attempt + 1}).")
                    return cap
            cap.release()
            print(f"[camera] {name} attempt {attempt + 1}/{attempts} not ready, retrying...")
            time.sleep(3.0)  # MF device needs a few seconds to settle/release
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--goal", default="kitchen", help="room goal, e.g. kitchen / bathroom / bedroom")
    args = ap.parse_args()

    goal = resolve_goal(args.goal)
    if goal is None:
        print(f"Unknown goal {args.goal!r}. Known goals include: kitchen, bathroom, "
              f"bedroom, living room, office, dining room.")
        return
    primary, secondary = indicators_for(goal)
    indicator_set = set(primary) | set(secondary)
    print(f"Goal: {goal!r}")
    print(f"  primary indicators:   {primary}")
    print(f"  secondary indicators: {secondary}")
    print("Arrival = >=1 primary OR >=2 secondary, confirmed over 2 of 3 frames.\n")

    print("Loading YOLOv8n (COCO)...")
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    names = model.names

    cap = _open_camera()
    if cap is None:
        print("ERROR: could not get live frames from the webcam.\n"
              "  - Close any other app/window using the camera (old demo windows, Zoom, Teams, browser).\n"
              "  - Check the privacy shutter / that the lens isn't covered.\n"
              "  - Windows Settings > Privacy & security > Camera > allow desktop apps.")
        return

    tts = SpeakingTTS()
    tts.say(f"Looking for the {goal}.")

    history: deque = deque(maxlen=WINDOW)
    announced = False

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        fh, fw = frame.shape[:2]

        res = model.predict(frame, verbose=False, conf=CONF_THRESHOLD)[0]
        this_frame = set()
        for box in res.boxes:
            cls = names[int(box.cls[0])]
            conf = float(box.conf[0])
            this_frame.add(cls)
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            is_ind = cls in indicator_set
            color = (0, 200, 0) if is_ind else (140, 140, 140)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2 if is_ind else 1)
            cv2.putText(frame, f"{cls} {conf:.2f}", (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        history.append(this_frame)
        counts = Counter()
        for s in history:
            counts.update(s)
        confirmed = {c for c, n in counts.items() if n >= MIN_HITS}

        result = evaluate_arrival(goal, confirmed)
        if result["arrived"] and not announced:
            tts.say(arrival_phrase(goal, result))
            announced = True

        # Overlay status.
        matched = result["matched_primary"] + result["matched_secondary"]
        banner = f"goal={goal} | confirmed indicators: {matched if matched else '-'}"
        cv2.putText(frame, banner, (10, fh - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if announced:
            cv2.putText(frame, f"ARRIVED: {goal}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2)

        cv2.imshow("M1 — semantic arrival (r=re-arm, q=quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            announced = False
            history.clear()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
