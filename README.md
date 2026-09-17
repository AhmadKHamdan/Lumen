# Lumen

**A task-oriented assistive system that helps blind users find objects in their environment, navigate to landmarks, and reach for objects, using only a smartphone.**

[![Python](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/server-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![YOLOv8n](https://img.shields.io/badge/vision-YOLOv8n-orange.svg)](https://docs.ultralytics.com/)
[![MediaPipe](https://img.shields.io/badge/hands-MediaPipe-4285F4.svg)](https://developers.google.com/mediapipe)
[![Whisper](https://img.shields.io/badge/STT-faster--whisper-7c3aed.svg)](https://github.com/SYSTRAN/faster-whisper)

> Birzeit University - ENCS5300 Graduation Project - 2026.

---

## What is Lumen?

A blind user opens a web page on their phone, points the camera at the room, and says **"find my cup."** Lumen answers in voice: *"Looking for your cup."* As they pan the camera, Lumen tracks the cup, calls out direction and distance ("to your left, a few steps away" -> "straight ahead, close by" -> "right in front of you, within arm's reach - raise your hand up in front of the camera"), watches their hand enter the frame, guides it ("move your hand to the right... almost there"), and announces grasp when the fingertip lands on the cup.

No special hardware. No app install. Just a phone browser, a WebSocket, and a Python backend doing the vision and speech work.

## Highlights

- **Voice in, voice out.** Push-to-talk recording, server-side faster-whisper for STT, gTTS for synthesis. No screen interaction required.
- **Three task families.** Find objects in a room, navigate to another room by exploring door-by-door, and reach for an object that's within arm's reach.
- **YOLOv8n + MediaPipe Hands.** Object detection and 21-landmark hand pose, both running on CPU, ~5 FPS end-to-end.
- **Per-class distance estimation.** "A laptop occupying 40% of frame width is near; the same image of a cup at 18% is also near" - same camera, correct guidance for each.
- **Auto-grasp completion.** When the user's fingertip enters the target's bounding box, the task ends automatically and announces success.
- **iOS-friendly audio.** Persistent primed `<audio>` element + Web Audio AudioContext unlock so TTS actually plays on iPhone Safari and Chrome.
- **Phone-deployable in minutes.** Same-origin frontend + Cloudflare quick tunnel = real HTTPS URL the phone can hit, no certificates to manage.

## Demo

Two phone recordings, straight from the browser session:

| Video | What happens |
| --- | --- |
| [**Find & reach** (42 s)](docs/demo/lumen-find-and-reach.mp4) | "Find my bottle" → Lumen locates the bottle on the desk, calls out direction and distance, invites the hand into the frame, guides it ("move your hand to the right… almost there") and announces the grasp. |
| [**Navigation** (1 min 34 s)](docs/demo/lumen-navigation.mp4) | "Take me to the kitchen" → guided 360° scan of the bedroom, door call-out with step count, a chair in the walking lane flagged by the obstacle watchdog, doorway transit, and arrival in the kitchen confirmed by the fridge. |

<p align="center">
  <img src="docs/demo/stills/find-bottle.jpg" width="19%" alt="Object Allocation: the bottle is detected on the desk">
  <img src="docs/demo/stills/reach-guidance.jpg" width="19%" alt="Reach Guidance: the hand is guided onto the bottle">
  <img src="docs/demo/stills/nav-door-and-chair.jpg" width="19%" alt="Navigation: door found, chair in the walking lane">
  <img src="docs/demo/stills/nav-at-the-door.jpg" width="19%" alt="Navigation: at the door, about to cross">
</p>

The spoken lines in the navigation video are exactly what the code says; [`tts_export/demo_script.md`](tts_export/demo_script.md) lists them in order and [`tts_export/make_demo_audio.py`](tts_export/make_demo_audio.py) regenerates the clips with the app's own gTTS voice. The final report is in [`docs/Lumen_Final_Report.pdf`](docs/Lumen_Final_Report.pdf) (LaTeX source under `report/`) and the final presentation in [`docs/Lumen_Final_Presentation.pdf`](docs/Lumen_Final_Presentation.pdf).

---

## Quickstart

### Requirements

- Python 3.13 (3.11+ works, 3.13 is what's verified).
- A modern browser (tested: Chrome on Android, Safari on iOS, Chrome on desktop).
- Optional but recommended for phone testing: [`cloudflared`](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/).

### Run the backend

```bash
cd backend
python -m venv .venv

# Activate
source .venv/Scripts/activate          # Git Bash on Windows
source .venv/bin/activate              # macOS / Linux
.\.venv\Scripts\Activate.ps1           # Windows PowerShell

# GPU box? Install the CUDA build of PyTorch FIRST - on Windows/Linux, plain
# `pip install torch` gives the CPU-only wheel, and the whole navigation stack
# (YOLOv8m + door models + SegFormer) would silently run on CPU. Pick the
# right command for your CUDA version at https://pytorch.org/get-started/
# e.g.:  pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

For demos, set `LUMEN_NAV_PRELOAD=1` before starting uvicorn: the navigation
models load (and, first time, download) in the background at startup, and the
fixed guidance phrases are pre-synthesized into the TTS cache - so the first
"navigate to the kitchen" answers immediately instead of sitting in silence.

```bash
LUMEN_NAV_PRELOAD=1 uvicorn main:app --host 127.0.0.1 --port 8000   # Git Bash / macOS / Linux
$env:LUMEN_NAV_PRELOAD="1"; uvicorn main:app --host 127.0.0.1 --port 8000   # PowerShell
```

First request triggers a few one-time downloads:

- Whisper `base` weights (~150 MB, from HuggingFace, cached in `~/.cache/huggingface/`).
- YOLOv8n weights (`yolov8n.pt`, ~6 MB, from the ultralytics CDN).
- MediaPipe Hands models ship inside the pip wheel - no download.
- First *navigation* task additionally pulls YOLOv8m (~50 MB, ultralytics CDN) and
  SegFormer-B0 (~15 MB, HuggingFace). The custom door detector + verifier weights
  are committed in the repo (`best.pt`, `door_training/runs/...`) - no download.

### Use it

Open <http://localhost:8000> in any browser. The frontend is served by FastAPI itself, so the page and the WebSocket share a single origin. Press **Start**, hold **PTT**, say "find my cup."

### Phone-test it (Cloudflare Tunnel)

Browsers require HTTPS for camera + microphone access except on `localhost`. Easiest path to a real HTTPS URL:

```bash
# In a second terminal (keep uvicorn running)
cloudflared tunnel --url http://localhost:8000
```

Cloudflare prints a `https://<random>.trycloudflare.com` URL. Open it on your phone. That's it - no cert provisioning, no router config.

---

## Supported commands

| User says | Result |
| --- | --- |
| "find my **cup**" / "where is my **bottle**" / "look for the **laptop**" | Start an Object Allocation task. |
| "navigate to the **kitchen**" / "take me to the **bathroom**" | Start a Navigation task. |
| "**got it**" / "found it" / "thanks" / "done" | Mark the current task complete and return to listening. |
| "**stop**" / "cancel" / "never mind" / "forget it" | Abort the current task. Session stays alive for the next command. |

Object Allocation supports a curated set of COCO classes (cup, bottle, chair, couch, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, sink, refrigerator, book, clock, vase, scissors). Common mishearings ("phone" -> "cell phone", "fridge" -> "refrigerator", "fone" -> "cell phone") are normalised before fuzzy matching.

---

## Architecture

```
+---------------------+                +-----------------------------------+
|  Browser (phone)    |    WebSocket   |  FastAPI backend                  |
|                     |  <----------> |  /ws (single connection per user) |
|  - getUserMedia     |                |                                   |
|  - MediaRecorder    |   JSON +       |  Router -> Session -> FSM         |
|  - <audio> element  |   binary tags  |                                   |
|  - Push-to-talk UI  |                |  Services:                        |
|  - compass heading  |                |    faster-whisper  (STT)          |
+---------------------+                |    YOLOv8n         (detection)    |
                                       |    MediaPipe Hands (hand pose)    |
                                       |    spatial_reasoning + guidance   |
                                       |    navigation (door model +       |
                                       |      exploration + obstacles)     |
                                       |    gTTS            (synthesis)    |
                                       +-----------------------------------+
```

A single `/ws` connection carries JSON control messages and three tagged binary frames:

- `0x01` JPEG camera frame (client -> server, 5 FPS).
- `0x02` WebM/Opus PTT audio blob (client -> server, on PTT release).
- `0x03` MP3 TTS clip (server -> client).

See [`docs/protocol.md`](docs/protocol.md) for the frozen wire contract.

The backend is a single uvicorn process. Each connected user gets a `Session` that owns its own FSM, latest decoded frame, task context, and detection-loop asyncio task. There is no shared task state across users.

### Task FSM

```
                  user_start
                ----------------->
   Idle                              ListeningForCommand
   ^                                    |
   |              cleanup_done          | command_recognized
   |              <-------              v
   ReturningToIdle <-----+    ObjectAllocationActive  /  NavigationActive
            ^             \              |
            |              \   user_stop |                task_abort
            |               \            v               ------------>
            +---- task_complete <--- (active states) ----> back to LISTENING
```

The FSM is authoritative for task state - the client doesn't track its own state, it just renders whatever `fsm_state` the server pushes. Every transition is driven by an explicit event (user gesture, recognised command, completion, or cancellation); there are no autonomous transitions except `cleanup_done` (one tick after entering `ReturningToIdle`) and `task_complete` on grasp.

---

## Task families

### Object Allocation - find a thing in the room

Pipeline per frame, 5 FPS:

1. **Detect** the requested COCO class with YOLOv8n. Drop boxes below 0.35 confidence.
2. **Temporal filter** - the target must appear in 3 of the last 5 frames before we trust it. Kills single-frame flickers.
3. **Spatial reasoning** - classify the target's region (left / center / right, split at 0.35 / 0.65 of frame width) and distance (near / medium / far) using `max(width_frac, height_frac)` against a per-COCO-class threshold. The per-class threshold means a laptop at 40% width is "near", a cup at 18% is also "near", and a phone at 12% is also "near" - same numeric area, three different right answers.
4. **Speak** a throttled, region-aware phrase. "Found your cup, to your left" first time it's seen; "Your cup is straight ahead, a few steps away" when bucket changes; same phrase silenced until either bucket changes or 6 s elapse.

Edge cases handled:

- Never detected within 30 s: periodic scan prompt ("I don't see your cup yet, slowly turn around").
- Seen, then lost from view for a full 5-frame window: one-shot "I lost sight of your cup, it was to your left" announcement.
- Never detected within 60 s: auto-abort with "I couldn't find your cup."
- Multiple instances visible at once: guide to the most head-on one (closest centre x to frame centre).

### Reach Guidance - hand-relative cues

When Object Allocation has the target at `near` distance AND MediaPipe detects a hand in frame, the loop switches to hand-relative cues:

- `approach` state with a named direction: "Move your hand to the right / left", "Raise your hand up", "Lower your hand."
- `almost` state when the fingertip is within 10% of the frame from the target's centroid: "Almost there. Push your hand straight forward."
- `touching` state when the fingertip enters the target's bounding box: "Your hand is on the cup. Grasp it." This is the **only autonomous success-exit** in the system - it fires `task_complete` automatically. Every other path requires the user to say "got it."

MediaPipe is only invoked when the tracker says we're in (or just left) reach distance - it stays idle the rest of the time, saving CPU.

### Navigation - goal-directed exploration

The user names only a destination ("navigate to the kitchen") and Lumen guides them there room-by-room, asking the user for direction only when perception is genuinely ambiguous. Per room:

1. **Scan** - a compass-tracked guided 360° turn. Detections are bucketed into 30° sectors; door and indicator sightings are recorded with true bearings (camera heading + position in frame).
2. **Arrive?** - rooms are recognised by their objects: ≥1 primary indicator (fridge/oven ⇒ kitchen, toilet ⇒ bathroom, bed ⇒ bedroom, ...) or ≥2 secondary ones, temporally accumulated so a single flicker never declares arrival. A directed re-confirmation ("turn that way, let's make sure") always precedes the announcement.
3. **Choose a door** - doors come from a three-layer funnel: a custom-trained single-class door detector (mAP50 ~0.95), geometric gates (aspect ratio, edge density against blank walls, frame-fill), and a 4-class DoorDetect verifier as a semantic second opinion (kills curtains and door↔fridge confusions).
4. **Go** - the spoken call-out gives bearing + step count (pinhole distance from the door's pixel height), then a path check. While walking, an **obstacle watchdog** fuses YOLO named objects in the walking lane with SegFormer floor segmentation (class-agnostic - catches clutter COCO has no word for) and preempts door guidance.
5. **Cross** - doorway transit is detected from the door box saturating the frame, then disappearing, *plus* sustained camera motion (the camera is the odometer - detection flicker can never fake "you're through"). Then the next room's scan starts automatically.

Runs as a per-session async loop (same pattern as Object Allocation) at ~3 Hz; a motion gate skips blurred frames while the phone pans and coaches the user to slow down. Without a compass (laptop, permission denied) the scan falls back to a single steady pass. See [`docs/Exploration_Navigation_Design.md`](docs/Exploration_Navigation_Design.md) for the design rationale and [`docs/Door_Model_Training_Results.md`](docs/Door_Model_Training_Results.md) for the door-model training.

---

## Project structure

```
Lumen/
├── backend/                       # FastAPI server (single process)
│   ├── main.py                    # /health, /ws, mounts ../frontend
│   ├── requirements.txt
│   ├── api/
│   │   ├── session.py             # one Session per WS, owns FSM + state
│   │   ├── router.py              # JSON + binary dispatch
│   │   ├── frame_handler.py       # JPEG -> numpy
│   │   └── audio_handler.py       # PTT blob -> STT -> parse -> FSM -> TTS
│   ├── fsm/
│   │   └── task_fsm.py            # 5 states, explicit transitions only
│   ├── services/
│   │   ├── speech_service.py      # faster-whisper + PyAV (no ffmpeg required)
│   │   ├── command_parser.py      # rapidfuzz intent extraction
│   │   ├── tts_service.py         # gTTS + bounded LRU cache
│   │   ├── yolo_service.py        # YOLOv8n via ultralytics, pure parser
│   │   ├── hand_service.py        # MediaPipe Hands, pure landmark->pose helper
│   │   ├── spatial_reasoning.py   # per-class distance + region classifier
│   │   ├── guidance_generator.py  # Object Allocation phrase templates
│   │   ├── reach_guidance.py      # fingertip-vs-target spatial logic + phrases
│   │   ├── object_allocation.py   # GuidanceTracker + 5 Hz async loop
│   │   └── navigation/            # goal-directed exploration (Sprint 4)
│   │       ├── engine.py          # per-session ~3 Hz async loop
│   │       ├── controller.py      # discover/face/confirm/go_door state machine
│   │       ├── perception.py      # 3-layer door funnel + COCO indicators
│   │       ├── obstacles.py       # walking-lane watchdog (YOLO + floor seg)
│   │       ├── geometry.py        # compass/bearing math
│   │       ├── goals.py           # room -> indicator tables, goal resolution
│   │       ├── models.py          # lazy YOLOv8m + door models + SegFormer
│   │       ├── state.py           # per-session NavState
│   │       └── config.py          # every tuning constant, in one place
│   ├── tests/                     # 290+ pytest cases
│   └── captured_audio/            # raw PTT WebM blobs for debugging (gitignored)
├── frontend/                      # Vanilla HTML + JS, no build step
│   ├── index.html
│   ├── styles.css
│   └── js/
│       ├── app.js                 # button wiring + audio prime on Start gesture
│       ├── ws_client.js
│       ├── media.js               # getUserMedia + MediaRecorder + 5 FPS loop
│       ├── audio_queue.js         # persistent primed <audio> for iOS
│       ├── heading.js             # compass heading for the navigation scan
│       └── wakelock.js
├── door_training/                 # door-model dataset prep + trained verifier
├── best.pt                        # custom single-class door detector (committed)
├── docs/
│   ├── demo/                      # the two demo videos + stills used above
│   ├── Lumen_Final_Report.pdf
│   ├── Lumen_Final_Presentation.pdf
│   ├── protocol.md                # frozen WS contract
│   ├── Exploration_Navigation_Design.md    # navigation design + rationale
│   ├── Door_Model_Training_Results.md      # door detector training report
│   └── ...                        # sprint plan + intro PDFs
├── report/                        # final report, LaTeX source (XeLaTeX + biber, see report/README.md)
├── tts_export/                    # demo-video narration script + gTTS clips
├── LICENSE
└── README.md
```

---

## Development

### Running tests

```bash
cd backend
python -m pytest tests/ -q
```

290+ tests covering: FSM transitions, command parsing (50+ realistic transcriptions including mishearings and synonyms), spatial bucketing (per-class distance), guidance phrase rendering, reach-guidance state machine, hand-pose landmark conversion, YOLO detection parsing, the Object Allocation `GuidanceTracker` end-to-end with scripted frames and a fake clock, and the navigation exploration controller end-to-end (guided 360 scan → door choice → approach → doorway transit → next room → semantic arrival) plus its compass/bearing math and room-indicator arrival rules.

The tests deliberately avoid loading the actual heavy ML models (YOLO weights, MediaPipe runtime) - they exercise the pure decision logic with mocks, so the suite runs in under a second.

### Tech stack

| Layer | Choice | Why |
| --- | --- | --- |
| Server | FastAPI + uvicorn | Async WebSockets, fast iteration, type hints. |
| STT | faster-whisper (`base`) | CTranslate2 backend; 4x faster CPU than openai-whisper; no torch dependency; Python 3.13 wheels. |
| Audio decode | PyAV | Bundles FFmpeg shared libs in the wheel - no system `ffmpeg.exe` on PATH required. |
| Object detection | YOLOv8n via ultralytics | Smallest of the family (~6 MB), CPU-friendly, COCO-pretrained matches our noun list. |
| Hand pose | MediaPipe Hands | 21 landmarks, CPU realtime, models bundled in wheel. |
| Navigation detection | YOLOv8m + custom door YOLOv8s x2 | v8m for room indicators at angle/distance (GPU); a trained single-class door detector + 4-class verifier (weights committed). |
| Obstacle floor check | SegFormer-B0 (ADE20K) | Labels every pixel; the walking lane must stay mostly floor. Optional - degrades to YOLO-only. |
| TTS | gTTS | Free; we cache MP3s; latency masked by parallel detection. |
| Command parsing | rapidfuzz | Tolerant of Whisper mishearings, structural matching of prefix + noun. |
| Frontend | Vanilla HTML/JS, no build | One less moving part. Frontend served same-origin by the backend. |

### Tunneling decisions

Cloudflare Tunnel quick tunnels are the default recommendation in this README because they're free, no-account, and need no DNS. For an always-on host, the codebase is ready for Azure App Service (B1), a small Linux VM behind Caddy, Azure Container Apps, or Hugging Face Spaces - same backend, same frontend, same WebSocket URL derivation.

---

## Roadmap

- [x] Sprint 1 - voice round-trip working end-to-end (FSM + WS + STT + TTS).
- [x] Sprint 2 - phone deployment over HTTPS (Cloudflare Tunnel + iOS audio unlock).
- [x] Sprint 3 - Object Allocation with YOLOv8n.
- [x] Sprint 4 - Navigation via goal-directed exploration (custom door model + room recognition + obstacle watchdog).
- [x] Sprint 5 - Reach Guidance with MediaPipe Hands.
- [x] Sprint 6 - Demo-feedback rounds (chatter control, hand-invite handoff, grouped scene descriptions), demo videos, final presentation and report.

See [`docs/Lumen_Implementation_Plan.pdf`](docs/Lumen_Implementation_Plan.pdf) for the full plan.

---

## Author

**Ahmad Hamdan** - Birzeit University, Electrical & Computer Engineering.

Developed as a team graduation project at Birzeit University's Department of Electrical and Computer Engineering. Architecture, vision pipeline, and reach-guidance work by the author; the exploration-navigation pipeline (door model training, perception funnel, exploration controller) by team-mates, integrated into the backend architecture jointly.

## License

Academic project - released under the [MIT License](LICENSE).
