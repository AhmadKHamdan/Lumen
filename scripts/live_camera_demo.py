"""Live webcam → YOLOv8n → Sprint 4 NavigationTaskManager → spoken audio.

This is the *real* navigation logic (backend/navigation) driven by a *real*
camera, supplying the one piece that doesn't exist in the monorepo yet:
Ahmad's Sprint 3 detection + spatial-reasoning output.

Pipeline per frame (~5 FPS):

    webcam frame (OpenCV)
        -> YOLOv8n detection (ultralytics)
        -> spatial reasoning  (region: left/center/right, distance: near/medium/far)
        -> NavigationTaskManager.on_detections([...])   <-- your real Sprint 4 code
        -> spoken audio (pyttsx3, offline Windows SAPI)

You drive the *voice* side two ways, both standing in for Omar's Whisper layer:
  - SPEAK: focus the video window, press 't' to start recording, speak, press
    't' again to stop — Whisper transcribes it and feeds the manager.
  - TYPE:  type into the terminal instead (handy fallback).
The first utterance is the destination; every one after is a user response
(a landmark, "skip", "stop", "done", "yes"/"no", "I'm here", ...).

Run from the repo root:

    python scripts/live_camera_demo.py
    python scripts/live_camera_demo.py --no-voice   # disable mic, type only

Keys (focus the OpenCV video window):
    t   start / stop voice recording (push-to-talk)
    q   quit

Type in the TERMINAL (not the video window):
    <destination>   first line, e.g.  kitchen
    <landmark>      e.g.  the chair  /  the chair and the table
    skip / stop / done / yes / no / I'm here   normal navigation replies
"""
from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path

# Make the `navigation` package importable from backend/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import cv2  # noqa: E402

from navigation import NavigationTaskManager  # noqa: E402


# ---------------------------------------------------------------------------
# Spoken TTS — runs pyttsx3 in a dedicated thread so synthesis never blocks
# the camera loop. Falls back to print-only if pyttsx3 isn't available.
# ---------------------------------------------------------------------------

class SpeakingTTS:
    """Implements the manager's TTSService protocol: synthesize(text)->bytes.

    We don't actually return audio bytes (the manager's _speak passes them to
    send_tts, which is a no-op here); instead we speak directly on a worker
    thread. Returning b"" keeps the manager happy.
    """

    def __init__(self) -> None:
        self._q: "queue.Queue[str]" = queue.Queue()
        self._engine = None
        try:
            import pyttsx3  # noqa: F401
            self._enabled = True
            threading.Thread(target=self._run, daemon=True).start()
        except Exception as e:  # pyttsx3 missing or no SAPI voice
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

    def synthesize(self, text: str) -> bytes:
        print(f"\n  🔊 [SPEAK] {text}\n" + "user> ", end="", flush=True)
        if self._enabled:
            self._q.put(text)
        return b""


def _send_tts_noop(_audio: bytes) -> None:
    pass


class PrintFSM:
    """Stands in for Omar's FSM — just prints the terminal events."""

    def handle_event(self, event: str, payload=None) -> None:
        print(f"\n  🟦 [FSM] event={event} payload={payload}\n" + "user> ",
              end="", flush=True)


# ---------------------------------------------------------------------------
# Spatial reasoning — the heuristic that stands in for Ahmad's Sprint 3 module.
# Converts a pixel-space bbox into the region / distance_category strings the
# detector consumes.
# ---------------------------------------------------------------------------

# YOLOv8 default inference resolution. obstacle_map.bbox_area_ratio() assumes
# this frame size for its small-object size gate, so we scale bboxes into this
# reference space before handing them to the navigation layer.
REF_W = REF_H = 640.0


def spatial_reasoning(x1, y1, x2, y2, frame_w, frame_h):
    """Return (region, distance_category) from a pixel-space bbox."""
    cx = (x1 + x2) / 2.0
    # Region: horizontal thirds of the frame (egocentric left/center/right).
    if cx < frame_w / 3.0:
        region = "left"
    elif cx > 2.0 * frame_w / 3.0:
        region = "right"
    else:
        region = "center"
    # Distance: how much of the frame the object fills. Bigger = closer.
    area_frac = ((x2 - x1) * (y2 - y1)) / (frame_w * frame_h)
    if area_frac >= 0.18:
        distance = "near"
    elif area_frac >= 0.05:
        distance = "medium"
    else:
        distance = "far"
    return region, distance


# ---------------------------------------------------------------------------
# Keyboard input thread — stands in for Omar's Whisper transcription. Pushes
# typed lines onto a queue the main loop drains (manager is single-threaded).
# ---------------------------------------------------------------------------

def _input_loop(q: "queue.Queue[str]") -> None:
    while True:
        try:
            line = input()
        except EOFError:
            q.put("__quit__")
            return
        q.put(line)


# ---------------------------------------------------------------------------
# Microphone push-to-talk — stands in for Omar's Whisper STT. Records mic audio
# straight into a NumPy array (no ffmpeg needed) and transcribes with Whisper.
# Transcription runs on a worker thread; the recognised text is pushed onto the
# same queue the typed input uses, so the main loop treats voice and keyboard
# identically.
# ---------------------------------------------------------------------------

class VoiceInput:
    SAMPLE_RATE = 16000  # Whisper expects 16 kHz mono float32.

    def __init__(self, out_q: "queue.Queue[str]", model_name: str = "base.en") -> None:
        import numpy as np
        import sounddevice as sd
        import whisper

        self._np = np
        self._sd = sd
        self._out_q = out_q
        print("Loading Whisper model (first run downloads ~140 MB)...")
        self._model = whisper.load_model(model_name)
        self.recording = False
        self.transcribing = False
        self._frames: list = []
        self._stream = None

    def _callback(self, indata, frames, time_info, status) -> None:
        if self.recording:
            self._frames.append(indata.copy())

    def toggle(self) -> None:
        """Press 't' once to start, again to stop-and-transcribe."""
        if not self.recording:
            self._frames = []
            self.recording = True
            self._stream = self._sd.InputStream(
                samplerate=self.SAMPLE_RATE, channels=1,
                dtype="float32", callback=self._callback,
            )
            self._stream.start()
            print("\n  🎤 [REC] listening... press 't' again to stop.")
        else:
            self.recording = False
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
                self._stream = None
            if not self._frames:
                print("user> ", end="", flush=True)
                return
            audio = self._np.concatenate(self._frames, axis=0).flatten()
            threading.Thread(target=self._transcribe, args=(audio,), daemon=True).start()

    def _transcribe(self, audio) -> None:
        self.transcribing = True
        try:
            result = self._model.transcribe(audio, language="en", fp16=False)
            text = (result.get("text") or "").strip()
        except Exception as e:
            print(f"\n  [STT] transcription failed: {e}")
            text = ""
        finally:
            self.transcribing = False
        if text:
            print(f"\n  🎤 [HEARD] {text}\n" + "user> ", end="", flush=True)
            self._out_q.put(text)
        else:
            print("\n  🎤 [HEARD nothing]\n" + "user> ", end="", flush=True)


def main() -> None:
    use_voice = "--no-voice" not in sys.argv

    print("Loading YOLOv8n model (first run downloads ~6 MB)...")
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    names = model.names  # {id: class_name}

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: could not open webcam (index 0).")
        return

    tts = SpeakingTTS()
    mgr = NavigationTaskManager(tts_service=tts, send_tts=_send_tts_noop, fsm=PrintFSM())

    input_q: "queue.Queue[str]" = queue.Queue()
    threading.Thread(target=_input_loop, args=(input_q,), daemon=True).start()

    voice = None
    if use_voice:
        try:
            voice = VoiceInput(input_q)
        except Exception as e:
            print(f"[VOICE] disabled ({e}); use the keyboard instead.")
            voice = None

    print("\n" + "=" * 64)
    print("LIVE NAVIGATION DEMO — real camera + YOLO + your Sprint 4 nav logic")
    print("=" * 64)
    if voice is not None:
        print("SPEAK: focus the video window, press 't' to record, speak, 't' to stop.")
    print("TYPE:  type into this terminal as a fallback.")
    print("First utterance = DESTINATION (e.g. 'kitchen'); then landmarks")
    print("('the chair'), or skip/stop/done/yes/no. Press 'q' in the window to quit.\n")
    print("user> ", end="", flush=True)

    started = False
    last_infer = 0.0
    infer_interval = 0.2  # ~5 FPS, matching the project's assumed frame rate.

    while True:
        # ---- drain keyboard input (voice stand-in) ----
        try:
            while True:
                line = input_q.get_nowait()
                if line == "__quit__":
                    raise KeyboardInterrupt
                line = line.strip()
                if not line:
                    print("user> ", end="", flush=True)
                    continue
                if not started:
                    mgr.on_navigation_command(line)
                    started = True
                else:
                    mgr.on_user_response(line)
        except queue.Empty:
            pass

        # ---- camera frame ----
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue
        fh, fw = frame.shape[:2]

        now = time.time()
        detections = []
        if now - last_infer >= infer_interval:
            last_infer = now
            results = model.predict(frame, verbose=False, conf=0.4)[0]
            sx, sy = REF_W / fw, REF_H / fh
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                cls_name = names[cls_id]
                region, distance = spatial_reasoning(x1, y1, x2, y2, fw, fh)
                detections.append({
                    "class_name": cls_name,
                    "confidence": conf,
                    # Scale bbox into the 640x640 reference frame so the
                    # navigation layer's small-object size gate is correct.
                    "bbox": [x1 * sx, y1 * sy, x2 * sx, y2 * sy],
                    "region": region,
                    "distance_category": distance,
                })
                # Draw overlay on the live frame.
                color = (0, 200, 0) if region == "center" else (200, 160, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"{cls_name} {region}/{distance} {conf:.2f}",
                            (int(x1), max(15, int(y1) - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # ---- feed the real navigation logic ----
            if mgr.is_frame_processing_enabled():
                mgr.on_detections(detections)
            mgr.check_timeout()

        # Status banner on the frame.
        if mgr.is_active():
            st = mgr.get_status()
            banner = f"dest={st.get('destination')} wp={st.get('current_waypoint')}"
        else:
            banner = "no active task — say/type a destination"
        cv2.putText(frame, banner, (10, fh - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Voice indicator (top-left).
        if voice is not None:
            if voice.recording:
                cv2.putText(frame, "( ) REC", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.circle(frame, (22, 22), 8, (0, 0, 255), -1)
            elif voice.transcribing:
                cv2.putText(frame, "transcribing...", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2)
            else:
                cv2.putText(frame, "press 't' to talk", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        cv2.imshow("Lumen — live navigation (t=talk, q=quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("t") and voice is not None and not voice.transcribing:
            voice.toggle()

    cap.release()
    cv2.destroyAllWindows()
    print("\nDemo ended.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
