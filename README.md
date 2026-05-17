# Lumen

A task-oriented assistive system for blind people.
Birzeit University · ENCS5200 Graduation Project · 2026.

> **Sprint 1 status:** in progress. End-to-end vertical slice — open the page,
> press Start, hold push-to-talk, say "find my cup", hear "looking for your cup"
> come back as audio. No object detection or navigation logic yet (Sprints 2–4).

## Team

- Diaa Badaha · 1210478 — Frontend, Audio Pipeline, Integration Owner
- Ahmad Hamdan · 1210241 — Vision Pipeline & Object Allocation Task
- Omar Husein · 1212738 — Voice Pipeline, FSM, Navigation Task

## Reference Documents

- [`docs/Intro.pdf`](docs/Intro.pdf) — original design report (Feb 4, 2026). Hardware section is superseded.
- [`docs/Lumen_Implementation_Plan.pdf`](docs/Lumen_Implementation_Plan.pdf) — six-sprint build plan. **Authoritative for implementation.**
- [`docs/protocol.md`](docs/protocol.md) — frozen WebSocket message contract.

## Architecture

A smartphone running Chrome opens a URL, grants camera/mic permissions, and acts as a
thin transport for media + audio. All inference (Whisper STT, YOLOv8n object detection,
gTTS) and the 5-state FSM live on a Python/FastAPI backend. See `docs/protocol.md` for
the message contract.

## Repo Layout

```
Lumen/
├── frontend/          # Single-page browser app (HTML + JS, no build step)
│   ├── index.html
│   ├── styles.css
│   └── js/            # ws_client, media, audio_queue, wakelock, app
├── backend/           # FastAPI server
│   ├── main.py
│   ├── api/           # session, router, frame_handler, audio_handler
│   ├── services/      # tts_service, speech_service, command_parser, object_allocation, navigation
│   ├── fsm/           # task_fsm
│   ├── tests/         # pytest suite
│   ├── captured_audio/  # raw PTT WebM blobs (gitignored)
│   └── requirements.txt
├── docs/
├── scripts/
└── README.md
```

## Prerequisites

- **Python 3.10+** with pip.
- **ffmpeg** on PATH (Whisper uses it to decode WebM/Opus → PCM).
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`
  - Windows: download from <https://ffmpeg.org/download.html> or `winget install ffmpeg`
- **Chrome** (Android, iOS, or desktop) for the frontend.

## Run Locally

### Backend

> ⚠️ **Don't put the project inside OneDrive / iCloud / Dropbox.** Sync clients
> lock files while pip and git try to write them, breaking venv creation and
> commits. Put the repo somewhere like `C:\Users\<you>\Projects\Lumen` or
> `~/Projects/Lumen`.

> ⚠️ **Use Python 3.11.** Python 3.13 has rough edges with `openai-whisper`'s
> torch dependency. We've verified Sprint 1 on Python 3.11.

```bash
cd backend

# Create the venv with Python 3.11 specifically
python3.11 -m venv .venv          # macOS/Linux
"C:/Users/Asus/AppData/Local/Programs/Python/Python311/python.exe" -m venv .venv  # Windows

# Activate
source .venv/Scripts/activate     # Git Bash on Windows
source .venv/bin/activate         # macOS/Linux
.venv\Scripts\Activate.ps1        # Windows PowerShell
```

Then install dependencies. **Do this in two steps because of the openai-whisper
build gotcha:**

```bash
# 1. Pin setuptools<80 in the venv first (openai-whisper's setup.py imports
#    pkg_resources, which setuptools 80+ removed)
python -m pip install --upgrade pip
python -m pip install "setuptools<80" wheel

# 2. Build openai-whisper with --no-build-isolation so it uses the
#    setuptools<80 we just installed (instead of pip pulling latest)
python -m pip install --no-build-isolation openai-whisper==20240930

# 3. Install the rest (these all ship wheels and install cleanly)
python -m pip install -r requirements.txt

# 4. Run the server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

> 💡 **If `which python` shows the global Python after activation** (Git Bash
> quirk), call the venv Python explicitly: `./.venv/Scripts/python.exe -m pip ...`
> and `./.venv/Scripts/python.exe -m uvicorn main:app --reload`.

The first request that triggers Whisper will download the `whisper-base` model
(~140 MB) into `~/.cache/whisper/`. This happens once.

### Frontend

The frontend is plain HTML+JS — no build step. Serve `frontend/` with any static
file server. The simplest option:

```bash
cd frontend
python -m http.server 8080
```

Then open <http://localhost:8080> in Chrome.

> **Why localhost?** Sprint 1 doesn't ship HTTPS, and Chrome only allows
> `getUserMedia` (camera + mic) on `http://localhost` or on `https://`.
> Phone testing requires HTTPS — that's added in Sprint 2 via `mkcert` + `ngrok`.

The frontend connects to `ws://localhost:8000/ws` by default. If you serve the
backend on a different host/port, edit `frontend/js/app.js`.

## Run Tests

```bash
cd backend
pytest
```

Should cover FSM transition rules and the command parser's intent extraction.

## Supported Browsers (v1)

- Chrome on Android — primary target.
- Chrome on iOS — secondary.
- Chrome on desktop — for development only.

Safari and Firefox are out of scope for v1.

## Known Limitations (v1)

- Push-to-talk only (no voice activity detection).
- English only (no Arabic).
- Foreground browser tab required during active tasks — switching apps or locking
  the phone ends the task.
- COCO-pretrained YOLOv8n only — no door/hallway/stairs detection.
- One concurrent user per server.
- No reconnection state recovery — WebSocket drop falls back to Idle.
- No HTTPS in Sprint 1 → no real-phone testing yet (added in Sprint 2).

## Sprint Plan

See [`docs/Lumen_Implementation_Plan.pdf`](docs/Lumen_Implementation_Plan.pdf) for the
full six-sprint plan. Quick summary:

1. **Sprint 1** (current): vertical slice — voice round-trip working end-to-end.
2. **Sprint 2**: HTTPS via mkcert+ngrok; phone testing.
3. **Sprint 3**: Object Allocation task with YOLOv8n (two weeks).
4. **Sprint 4**: Navigation task with landmark waypoints.
5. **Sprint 5**: Polish, internal blindfolded trials, report rewrite.
