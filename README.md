# Lumen

**A task-oriented assistive system that helps blind users find objects in their environment, navigate to landmarks, and reach for objects, using only a smartphone.**

[![Python](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/server-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![YOLOv8n](https://img.shields.io/badge/vision-YOLOv8n-orange.svg)](https://docs.ultralytics.com/)
[![MediaPipe](https://img.shields.io/badge/hands-MediaPipe-4285F4.svg)](https://developers.google.com/mediapipe)
[![Whisper](https://img.shields.io/badge/STT-faster--whisper-7c3aed.svg)](https://github.com/SYSTRAN/faster-whisper)

> Birzeit University - ENCS5200 Graduation Project - 2026.

---

## What is Lumen?

A blind user opens a web page on their phone, points the camera at the room, and says **"find my cup."** Lumen answers in voice: *"Looking for your cup."* As they pan the camera, Lumen tracks the cup, calls out direction and distance ("to your left, a few steps away" -> "straight ahead, close by" -> "right in front of you, reach forward"), watches their other hand enter the frame, guides it ("move your hand to the right... almost there"), and announces grasp when the fingertip lands on the cup.

No special hardware. No app install. Just a phone browser, a WebSocket, and a Python backend doing the vision and speech work.

## Highlights

- **Voice in, voice out.** Push-to-talk recording, server-side faster-whisper for STT, gTTS for synthesis. No screen interaction required.
- **Three task families.** Find objects in a room, navigate toward a landmark, and reach for an object that's within arm's reach.
- **YOLOv8n + MediaPipe Hands.** Object detection and 21-landmark hand pose, both running on CPU, ~5 FPS end-to-end.
- **Per-class distance estimation.** "A laptop occupying 40% of frame width is near; the same image of a cup at 18% is also near" - same camera, correct guidance for each.
- **Auto-grasp completion.** When the user's fingertip enters the target's bounding box, the task ends automatically and announces success.
- **iOS-friendly audio.** Persistent primed `<audio>` element + Web Audio AudioContext unlock so TTS actually plays on iPhone Safari and Chrome.
- **Phone-deployable in minutes.** Same-origin frontend + Cloudflare quick tunnel = real HTTPS URL the phone can hit, no certificates to manage.

## Demo

> Screenshots and a demo video will go here once recorded.
>
> ```
> [ phone screenshot: search in progress ]   [ phone screenshot: reach guidance ]
> ```

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

pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

First request triggers two one-time downloads:

- Whisper `base` weights (~150 MB, from HuggingFace, cached in `~/.cache/huggingface/`).
- YOLOv8n weights (`yolov8n.pt`, ~6 MB, from the ultralytics CDN).
- MediaPipe Hands models ship inside the pip wheel - no download.

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
+---------------------+                |    faster-whisper  (STT)          |
                                       |    YOLOv8n         (detection)    |
                                       |    MediaPipe Hands (hand pose)    |
                                       |    spatial_reasoning + guidance   |
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
- `almost` state when the fingertip is within 10% of the frame from the target's centroid: "Almost there. Reach forward."
- `touching` state when the fingertip enters the target's bounding box: "Your hand is on the cup. Grasp it." This is the **only autonomous success-exit** in the system - it fires `task_complete` automatically. Every other path requires the user to say "got it."

MediaPipe is only invoked when the tracker says we're in (or just left) reach distance - it stays idle the rest of the time, saving CPU.

### Navigation - guide to a landmark

Sister task to Object Allocation, currently being developed by another team member. Uses YOLOv8n's furniture classes as proxy landmarks, with the same FSM scaffolding (`NavigationActive` state, the same cancel / completion verbs).

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
│   │   └── navigation.py          # landmark waypoints (in development)
│   ├── tests/                     # 200+ pytest cases
│   └── captured_audio/            # raw PTT WebM blobs for debugging (gitignored)
├── frontend/                      # Vanilla HTML + JS, no build step
│   ├── index.html
│   ├── styles.css
│   └── js/
│       ├── app.js                 # button wiring + audio prime on Start gesture
│       ├── ws_client.js
│       ├── media.js               # getUserMedia + MediaRecorder + 5 FPS loop
│       ├── audio_queue.js         # persistent primed <audio> for iOS
│       └── wakelock.js
├── docs/
│   ├── protocol.md                # frozen WS contract
│   └── ...                        # sprint plan + intro PDFs
└── README.md
```

---

## Development

### Running tests

```bash
cd backend
python -m pytest tests/ -q
```

200+ tests covering: FSM transitions, command parsing (50+ realistic transcriptions including mishearings and synonyms), spatial bucketing (per-class distance), guidance phrase rendering, reach-guidance state machine, hand-pose landmark conversion, YOLO detection parsing, and the Object Allocation `GuidanceTracker` end-to-end with scripted frames and a fake clock.

The tests deliberately avoid loading the actual heavy ML models (YOLO weights, MediaPipe runtime) - they exercise the pure decision logic with mocks, so the suite runs in under a second.

### Tech stack

| Layer | Choice | Why |
| --- | --- | --- |
| Server | FastAPI + uvicorn | Async WebSockets, fast iteration, type hints. |
| STT | faster-whisper (`base`) | CTranslate2 backend; 4x faster CPU than openai-whisper; no torch dependency; Python 3.13 wheels. |
| Audio decode | PyAV | Bundles FFmpeg shared libs in the wheel - no system `ffmpeg.exe` on PATH required. |
| Object detection | YOLOv8n via ultralytics | Smallest of the family (~6 MB), CPU-friendly, COCO-pretrained matches our noun list. |
| Hand pose | MediaPipe Hands | 21 landmarks, CPU realtime, models bundled in wheel. |
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
- [ ] Sprint 4 - Navigation with landmark waypoints (in development).
- [x] Sprint 5 - Reach Guidance with MediaPipe Hands.
- [ ] Sprint 6 - Blindfolded user trials, performance polish, final report.

See [`docs/Lumen_Implementation_Plan.pdf`](docs/Lumen_Implementation_Plan.pdf) for the full plan.

---

## Author

**Ahmad Hamdan** - 1210241 - Birzeit University, ENCS5200.

Developed as a team graduation project at Birzeit University's Department of Electrical and Computer Engineering. Architecture, vision pipeline, and reach-guidance work by the author; navigation pipeline by a team-mate.

## License

Academic project - released under the MIT License. See `LICENSE` (to be added) if you want to build on it.
