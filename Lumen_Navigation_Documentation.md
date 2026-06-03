# Lumen — Navigation Task Documentation

> **For Ahmad and Omar:** This document describes what I built for the Navigation task (Sprint 4) and how it fits with what you're building. It's meant to be read once before integration so you know what to expect, what I need from you, and where things live.

---

## Table of Contents

1. [What I built and why](#1-what-i-built-and-why)
2. [How the pieces connect](#2-how-the-pieces-connect)
3. [What I need from each of you](#3-what-i-need-from-each-of-you)
4. [Walk through a full navigation session](#4-walk-through-a-full-navigation-session)
5. [Part 1 — Dialog scaffolding](#5-part-1--dialog-scaffolding)
6. [Part 2 — Landmark mapping & detection](#6-part-2--landmark-mapping--detection)
7. [Part 3 — Obstacle awareness](#7-part-3--obstacle-awareness)
8. [Part 4 — Guidance generation](#8-part-4--guidance-generation)
9. [Part 5 — Integration, completion & end-to-end](#9-part-5--integration-completion--end-to-end)
10. [Every decision I made and why](#10-every-decision-i-made-and-why)
11. [The code files](#11-the-code-files)
12. [Tests](#12-tests)
13. [How to run it yourself](#13-how-to-run-it-yourself)
14. [Known limitations and Sprint 5 ideas](#14-known-limitations-and-sprint-5-ideas)

---

## 1. What I built and why

The Navigation task lets a blind user walk from one indoor location to another using voice and landmark detection. The user says something like *"navigate to the kitchen"* and then describes landmarks along the way (*"through the doorway"*). The system uses the camera to spot those landmarks, tells the user where each one is relative to them as they walk, warns about obstacles, and announces when each landmark is reached. When all the landmarks are done, it asks "What's next?" and the user can either add more landmarks or say "done" to finish cleanly.

I broke this into five parts so I could ship and test each piece independently:

| Part | What it does | Status |
|---|---|---|
| 1 | The conversation: ask for landmarks, store them, handle cancel and timeout | Done |
| 2 | Translate the spoken landmark into something YOLO can detect; decide "reached" | Done |
| 3 | Warn about obstacles in the user's path | Done |
| 4 | Speak guidance ("chair on your right, getting closer") | Done |
| 5 | Task completion, "I'm here" override, FSM events, final integration | Done |

Total: **294 automated tests passing**, all in under a second. Everything runs without YOLO, gTTS, or a WebSocket — I used dependency injection so I could build and test the navigation logic independently while you two were working on Sprints 1–3.

---

## 2. How the pieces connect

### Where the navigation code lives

```
backend/navigation/
├── __init__.py         What other code can import from us
├── context.py          Data: Waypoint, NavigationContext, all the state
├── parser.py           Text helpers: clean transcriptions, detect "stop"/"I'm here"/"done"/"yes"/"no"
├── landmark_map.py     "doorway" → which YOLO classes? (Part 2)
├── obstacle_map.py     Which classes count as obstacles? (Part 3)
├── detector.py         The per-frame engine: obstacles, reached, guidance
└── manager.py          The orchestrator everything else talks to
```

### One class to talk to

You only need to know about one class: **`NavigationTaskManager`**. It's the entry point. The FSM calls it when navigation starts, the WebSocket calls it when the user speaks, the frame handler calls it when detections come in.

Easiest way to construct it:

```python
from navigation import create_navigation_session

nav = create_navigation_session(
    tts_service = <Omar's gTTS service>,
    fsm         = <Omar's FSM>,
    send_tts    = lambda mp3: ws.send_bytes(mp3),
)
```

That helper just wraps the regular constructor. The point is: you can each work on your own pieces and plug them together at the end with one line.

### How a frame moves through the system

```
Ahmad's frame handler captures a frame, runs YOLO, builds detection dicts
              ↓
nav.on_detections([dict1, dict2, ...])    ← one call per frame, ~5 FPS
              ↓
Manager normalizes dict shapes (accepts class/class_name/label, conf/confidence/score, etc.)
              ↓
LandmarkDetector.on_frame_detections():
              ↓
   1. Obstacle check  — fires first; if it warns, suppress everything else
              ↓
   2. Filter landmark detections, update 3-frame rolling buffer
              ↓
   3. Disambiguation? — if "seat" matched both chair AND couch, ask user
              ↓
   4. Guidance check  — speak only if region or distance changed
              ↓
   5. Reached check   — temporal + center + near → "You've reached the X"
              ↓
   6. Auto-advance to next waypoint, or "What's next?" if queue is empty (Part 5)
```

---

## 3. What I need from each of you

### From Omar (FSM, voice, command parser)

**Three hooks need to fire at the right times.** I've already designed the manager around them.

1. **When the FSM enters `NavigationActive`,** call:
   ```python
   nav.on_navigation_command(parsed_intent["destination"])
   ```
   where `parsed_intent["destination"]` is whatever your command parser extracted ("kitchen", "bathroom", etc.). If a task is already running and the user says "navigate to bathroom", just call this again — I detect that and cleanly restart with the new destination.

2. **For every transcription that comes back from Whisper while we're in `NavigationActive`,** call:
   ```python
   nav.on_user_response(transcription_text)
   ```
   I route it internally — could be a waypoint, "stop", "skip", "I'm here", "yes"/"no" (during arrival confirmation), "done" (during what's-next), a disambiguation answer, etc.

3. **In the session loop (e.g. once per second),** call:
   ```python
   nav.check_timeout()
   ```
   This handles the 30-second waypoint-prompt timer, the 60-second detection timer, the 10-second disambiguation timer, the 30-second "what's next" timer, and the 30-second arrival-confirmation timer. Fire-and-forget.

**Two FSM events you'll receive from me:**
```python
fsm.handle_event("task_completed", {"task": "navigation", "summary": {...}})
fsm.handle_event("task_cancelled", {"task": "navigation"})
```
- `task_completed` fires when the user says "done" after "What's next?", or when the what's-next timeout elapses.
- `task_cancelled` fires on voice cancel, on waypoint-prompt timeout with no answer, or on the give-up after 3 unintelligible inputs.

Both should transition the FSM from `NavigationActive` → `ReturningToIdle`. If your `handle_event` raises an exception, I'll log it and continue gracefully.

### From Ahmad (YOLO, spatial reasoning, temporal consistency)

**One method to call, with one specific dict shape.** When a frame is processed and you have detection results, hand them to me as a list of dicts:

```python
[
  {
    "class_name": "chair",                          # the COCO class name
    "confidence": 0.87,                             # 0 to 1
    "bbox": [120, 200, 380, 540],                   # x1,y1,x2,y2
    "region": "center",                             # from spatial reasoning
    "distance_category": "near",                    # from spatial reasoning
  },
  ...
]
```

Then call:
```python
if nav.is_frame_processing_enabled():
    nav.on_detections(processed_detections)
```

**I tolerate a few key-name variations** (the adapter in Part 5):
- `class` or `label` or `name` work in place of `class_name`
- `conf` or `score` work in place of `confidence`
- `box` works in place of `bbox`
- `distance` works in place of `distance_category` (if it's a string)

**`is_frame_processing_enabled()` is the gate.** It's `False` until the user has given me their first waypoint, and `False` after the task ends or while we're in the "What's next?" dialog. Skipping detection when the gate is closed saves you YOLO compute.

**About `region` and `distance_category`:** Your spatial-reasoning module attaches these. If they're missing in the early days, I won't crash — I log a warning once and the reached/guidance checks just fail safely.

**Frame rate:** I assume **5 FPS** (one frame every 200 ms). That matches the report and the implementation plan.

---

## 4. Walk through a full navigation session

```
USER (voice):  "navigate to the kitchen"
               [FSM enters NavigationActive]
               [FSM calls nav.on_navigation_command("kitchen")]
SYSTEM (TTS):  "Where should I take you first?"

USER (voice):  "through the doorway and the table"
               [I clean and resolve both to ["door"] and ["dining table"]]
               [I store two Waypoints; doorway active, table pending]
SYSTEM (TTS):  "Looking for doorway."

(Ahmad's frame handler starts feeding me detections every 200 ms)

Frame 12: door detected on the right, "far" — but only one frame so far
Frame 13: door detected on the right, "far" — 2-of-3 satisfied
SYSTEM (TTS):  "Door on your right, far away."

Frame 30: door detected on the right, "medium" — distance changed
SYSTEM (TTS):  "Door on your right, getting closer."

Frame 38: a person walks into the path, "center", "near"
SYSTEM (TTS):  "Person ahead."     ← obstacle preempts; reached still tracks silently

Frame 39-49: person still there, on cooldown — silent
Frame 50: person has cleared

Frame 55: door now centered, "near"
SYSTEM (TTS):  "You've reached the doorway."
SYSTEM (TTS):  "Looking for table."           ← auto-advance

(continues with table guidance and reached...)

SYSTEM (TTS):  "You've reached the table."
SYSTEM (TTS):  "What's next? Say a landmark, 'done' to finish, or 'stop' to cancel."

USER (voice):  "done"
SYSTEM (TTS):  "Navigation complete."
               [FSM event: task_completed with summary {destination: 'kitchen',
                waypoints_reached: 2, waypoints_skipped: 0, duration_seconds: 87.3}]
```

---

## 5. Part 1 — Dialog scaffolding

### Goal

Build the conversation skeleton: enter navigation, ask the user for a landmark, store what they say, handle cancellation and timeouts, exit cleanly.

### What's in this part

**The data structures (`context.py`):**

- A `Waypoint` is one landmark the user wants to pass through. It has the raw text they said, the cleaned form, a status (PENDING / ACTIVE / REACHED / SKIPPED), and a timestamp.
- A `NavigationContext` is the full state of one in-flight task. It holds the destination, the list of waypoints (in order, never discarded), which one is currently active, and dialog sub-state flags.
- The context lives as an attribute on the session — one navigation task = one context.

**The text helpers (`parser.py`):**

- `normalize_phrase("Go Through The Doorway!")` returns `"doorway"`. Strips fillers and articles.
- `parse_waypoint("doorway then turn left")` returns two candidates — multi-waypoint in one utterance.
- `is_cancellation_phrase` recognizes "stop", "cancel", "never mind", "abort".
- `is_arrival_phrase`, `is_yes`, `is_no`, `is_done_phrase` — used in Parts 2 and 5.

**The orchestrator (`manager.py`):**

Public methods in Part 1:
- `on_navigation_command(destination)` — FSM entry hook
- `on_user_response(transcription)` — routes by sub-state
- `check_timeout()` — fires 30-second prompt timeout
- `cancel()` — universal exit, emits `task_cancelled` FSM event

### Phrases the user hears in Part 1

| Trigger | What the user hears |
|---|---|
| Task starts | "Where should I take you first?" |
| Empty / unintelligible answer | "I didn't catch that. What's the landmark?" |
| Waypoint accepted | "Looking for {waypoint}." |
| User says "stop" | "Navigation cancelled." |
| 30s with no answer | "Cancelling navigation. No landmark received." |
| 3 empty answers in a row | "I'm having trouble understanding. Cancelling navigation." |

---

## 6. Part 2 — Landmark mapping & detection

### Goal

Connect a stored waypoint to YOLO's vocabulary, consume detection results frame by frame, and decide the exact moment the user has reached the landmark.

### The bridge: spoken words to YOLO classes

YOLO has 80 fixed class names. The user says "the fridge" or "the seat" — not YOLO class names. The 78-entry **alias dictionary** in `landmark_map.py` translates.

Three-step resolution:
1. **Exact match.** "chair" is already a key → `["chair"]`.
2. **Substring match.** "blue chair" → finds "chair" inside → `["chair"]`.
3. **Fuzzy match** (rapidfuzz, threshold 80). "doorwya" → "doorway" → `["door"]`.

If all three fail, return empty list → re-prompt the user.

**One alias can map to multiple classes.** "Seat" returns `["chair", "couch", "bench"]`. The detector handles this case by asking the user.

**Architectural classes are a known gap.** COCO doesn't have "wall", "hallway", "stairs". "doorway" → "door" is best-effort. Reserved for a Sprint 5 fine-tune.

### The per-frame engine: `LandmarkDetector`

Constructed when navigation starts looking. Registers callbacks:
- `on_reached(waypoint)` — temporal consistency + center + near.
- `on_disambiguation_needed(choices)` — multiple alias classes both present.
- `on_detection_timeout()` — 60 seconds with no matching detection.

The detector is a **pure observer**. It doesn't speak; it emits events; the manager decides what to say.

### Temporal consistency: 2-of-3

A 3-frame rolling buffer. A landmark is "confirmed" if it appeared in at least 2 of the last 3 frames. Same rule gates reached AND guidance.

### The disambiguation dialog

When user says "seat" and we see BOTH a chair AND a couch confirmed:

```
SYSTEM:  "I see a chair and a couch. Which would you like me to find?"
USER:    "chair"
SYSTEM:  "Looking for the chair."
```

While waiting:
- Reached check paused; obstacles keep running; guidance paused.
- 10-second timeout → fall back to most-observed class.
- Wrong answer → re-prompt once, then fall back.
- "stop" always cancels.

Once picked, the waypoint is **locked** to that class.

### What "reached" means

ALL three:
1. Temporal consistency satisfied.
2. Current frame `region == "center"`.
3. Current frame `distance_category == "near"`.

On reached: speak "You've reached the {waypoint}", set status to REACHED, advance to next or trigger "What's next?".

### Detection timeout (60 s of no match)

```
SYSTEM:  "I haven't spotted the {waypoint} yet. Say 'skip' to move on,
          'stop' to cancel, or describe a different landmark."
```

### Phrases the user hears in Part 2

| Trigger | What the user hears |
|---|---|
| Waypoint unmappable | "I don't know that landmark. What's the landmark?" |
| Some waypoints unmappable | "I didn't recognize: {dropped}." (before "Looking for...") |
| Landmark reached | "You've reached the {waypoint}." |
| Auto-advance | "Looking for {next}." |
| 60s no detection | (detection-timeout text) |
| User skip + queue empty | "What's the next landmark?" |
| Disambiguation needed | "I see a {a} and a {b}. Which would you like me to find?" |
| User picks one | "Looking for the {locked class}." |

---

## 7. Part 3 — Obstacle awareness

### Honest framing first

The report's known limitations: *"perception latency unsuitable for time-critical obstacle avoidance"*. Part 3 is **advisory, not collision-avoidance**. Calm phrasing. No "STOP."

### What counts as an obstacle

`obstacle_map.py` covers ~50 COCO classes:
- **Person** — moving hazard, distinct phrasing, highest priority.
- **Large furniture/appliances** — chair, couch, bed, fridge, oven, etc.
- **Medium objects** — backpack, suitcase, bowl.
- **Small objects** — cup, bottle, book, fork. **Only warn if bbox ≥ 8% of frame.**
- **Pets** — cat, dog.

**The active landmark is never an obstacle.**

### When warnings fire

All of these:
1. Class is an obstacle (not the active landmark).
2. `region == "center"`.
3. `distance_category == "near"`.
4. Confidence ≥ 0.6.
5. For small objects: bbox area ≥ 8% of frame.

Multiple obstacles: **person > pet > furniture**. Ties broken by bbox area.

### Throttling

**3-second per-class cooldown.** A different class bypasses immediately. When an obstacle clears, its cooldown resets.

### Constrained escalation

| Level | Furniture | Person |
|---|---|---|
| 0 | "Chair ahead." | "Person ahead." |
| 1 | "Chair still ahead." | "Person still ahead." |
| 2 | "Chair still in path, please be careful." | "Person still in path, please be careful." |

Level 2 caps. No shouting, no "STOP."

### Preemption

Obstacle check runs FIRST every frame. If anything fires, reached and guidance announcements are suppressed for that frame. State still updates internally — we just don't speak.

---

## 8. Part 4 — Guidance generation

### Goal

Speak useful direction cues during the walk. Only when something meaningfully changes.

### The cognitive-load rule

The implementation plan: *"speak only when spatial region or distance category changes."* That's the entire rule.

### What we track

```python
last_spoken_region    # "left", "center", "right", or None
last_spoken_distance  # "far", "medium", "near", or None
last_spoken_class     # which COCO class we last named
```

Cleared on `advance()` so each new waypoint starts fresh.

### When guidance fires

After obstacle check (so obstacles preempt) and after disambiguation gate:

1. Temporal consistency must be satisfied.
2. Region and distance must be present.
3. Compare to last spoken state:
   - First time → "first cue" (full description)
   - Class name changed → fresh full cue
   - Region changed → fresh full cue
   - Distance changed only → "getting closer" cue
   - Both changed → fresh full cue
   - Nothing changed → silent

### Phrasing

Directions: `left` → "on your left", `right` → "on your right", `center` → "ahead".

Distances: `far` → "far away", `medium` → "approaching", `near` → "right in front of you".

Smoothed combinations:
- Off-center: *"Chair on your right, far away."*
- Center+far: *"Chair far ahead."*
- Center+near: *"Chair right in front of you."*
- Off-center to near: *"Chair on your right, right in front of you."* (course-correction moment)
- Distance changing to medium: *"Chair on your right, getting closer."*

### Lost sight

> *"Lost sight of the chair. Looking for it."*

Spoken once when landmark disappears after being visible. Resets last-spoken state.

---

## 9. Part 5 — Integration, completion & end-to-end

### Goal

Close the loop: how does the task end? What if the user says "I'm here" before the detector agrees? What event does the FSM see?

### Task completion

When the last queued waypoint is reached:

```
SYSTEM:  "What's next? Say a landmark, 'done' to finish, or 'stop' to cancel."
```

Options:
- **"done"** / "finished" / "all done" → "Navigation complete." → `task_completed` FSM event → tear down.
- **A new landmark** → appended, navigation continues.
- **"stop"** → standard cancellation.
- **30s silence** → "Navigation finished." → auto-complete.

### Completion payload

```python
{
  "task": "navigation",
  "summary": {
    "destination": "kitchen",
    "waypoints_total": 2,
    "waypoints_reached": 2,
    "waypoints_skipped": 0,
    "duration_seconds": 87.3,
  }
}
```

Exactly what Sprint 5 evaluation needs.

### "I'm here" override

User can claim arrival even if the detector hasn't fired reached.

```
USER:    "I'm here"
SYSTEM:  "I see the chair on your right, far away. Are you at the chair?
          Say 'yes' to confirm or 'no' to keep going."
USER:    "yes"
SYSTEM:  "You've reached the chair."
```

- If yes → mark reached, advance.
- If no → "OK, continuing to look." Navigation continues.
- Ambiguous answer → default to yes.
- 30-second silence → default to yes.
- "I'm here" before any waypoint → "There's no destination yet. Where should I take you first?"

### Re-issued "navigate to" mid-task

User says "navigate to bathroom" mid-task → manager detects existing context, tears it down silently, starts fresh with new destination.

### Async-aware TTS

`_speak` auto-detects coroutine return from `synthesize()`. Sync and async TTS both work.

### Dict-key adapter

Accepts variations: `class`/`label`/`name` for `class_name`, `conf`/`score` for `confidence`, etc.

### Wiring helper

```python
from navigation import create_navigation_session

nav = create_navigation_session(
    tts_service=omar_tts_service,
    fsm=omar_fsm,
    send_tts=lambda mp3: ws.send_bytes(mp3),
)
```

### Status query

```python
status = nav.get_status()
# {"active": True, "destination": "kitchen", "current_waypoint": "chair",
#  "current_waypoint_status": "active", "frame_processing_enabled": True, ...}
```

Safe when no task is active (returns `{"active": False}`).

### Resilience

- TTS failure → log, continue.
- FSM `handle_event` raises → log, continue.
- Malformed detection dict → silently dropped.
- Manager is **not thread-safe**; partners call from a single session loop.

### Phrases the user hears in Part 5

| Trigger | What the user hears |
|---|---|
| Last waypoint reached, queue empty | "What's next? Say a landmark, 'done' to finish, or 'stop' to cancel." |
| User says "done" | "Navigation complete." |
| 30s silence after "What's next?" | "Navigation finished." |
| "I'm here" with visible detection | "I see the chair on your right, far away. Are you at the chair? Say 'yes' to confirm or 'no' to keep going." |
| "I'm here" with nothing visible | "I don't see the chair yet. Are you sure you've reached it? Say 'yes' to confirm or 'no' to keep going." |
| User says no after I'm-here | "OK, continuing to look." |
| "I'm here" before any waypoint | "There's no destination yet. Where should I take you first?" |

---

## 10. Every decision I made and why

### Part 1 — Dialog scaffolding

| Decision | What we do | Why |
|---|---|---|
| Code organization | Separate `navigation/` package with multiple files | The task is big enough to grow; one giant file would be unmaintainable |
| Orchestrator class | One `NavigationTaskManager` class as the single entry point | Easy for partners to call, single owner of state |
| Waypoint representation | `@dataclass Waypoint` with raw text, normalized text, status, timestamp | Structured object beats a dict — autocomplete, type checking, no typos |
| Waypoint container | A plain Python list, with an index pointing at the current waypoint | We need to keep completed waypoints around for evaluation metrics |
| Text normalization | Store both raw and normalized text on the waypoint | Raw text for logs; normalized text for matching |
| Parser extension | New `parse_waypoint()` function instead of extending Omar's command parser | Single responsibility |
| First prompt | "Where should I take you first?" instead of the report's placeholder | More natural and conversational |
| Multi-waypoint utterance | Split on "then", "and", commas | User-collaboration design encourages this; fewer dialog turns |
| FSM transition timing | Transition to NavigationActive FIRST, then ask for the landmark | Cleaner FSM — landmark-collection is a sub-phase |
| Frame processing during waypoint collection | Camera/YOLO stays off until first waypoint stored | Task-scoped perception principle |
| Voice cancellation | "stop", "cancel", "never mind", "abort" all work | Assistive UX can't assume button reachability |
| Unrecognized waypoint handling | Re-prompt twice, then cancel | Bounded retries prevent infinite loops |
| Waypoint-prompt timeout | 30 seconds | Matches the OA detection-timeout pattern |
| Audible confirmation | Always speak "Looking for X" after a waypoint is accepted | A blind user needs auditory acknowledgment |

### Part 2 — Landmark mapping & detection

| Decision | What we do | Why |
|---|---|---|
| Mapping table storage | Hardcoded Python dict in `landmark_map.py` | Simple, version-controlled, no I/O at startup |
| Mapping direction | One alias → list of COCO classes (1-to-many) | Users describe loosely ("seat") |
| Match strategy | Exact → substring → fuzzy (rapidfuzz, threshold 80) | Robust to Whisper typos and natural phrasing |
| When nothing matches | Reject, re-prompt with "I don't know that landmark" | Silent failures are the worst UX |
| Initial coverage | Furniture + appliances + common indoor objects + best-effort architectural | Covers what users actually describe |
| Architectural gap | "doorway" → "door" (best-effort); "wall"/"hallway"/"stairs" → empty | COCO doesn't have these; planned Sprint 5 fine-tune |
| Resolution timing | Eager — at waypoint-add time | Unmappable waypoints caught immediately |
| Multi-waypoint with one unmappable | Keep mappable ones, announce dropped | Honors user effort; transparent |
| Detection input shape | Dicts with class_name, confidence, bbox, region, distance_category | Reuses Ahmad's spatial-reasoning output |
| Frame rate | 5 FPS (200 ms) | Matches the report and the plan |
| Confidence threshold | 0.6 | Stricter than YOLO's default; false positives confuse blind users |
| Multi-class alias both present | Ask the user (disambiguation dialog) | User autonomy |
| Disambiguation: lock choice | Lock to chosen class for rest of waypoint | Avoid re-asking |
| Disambiguation: state while waiting | Pause reached and guidance; obstacles run | Safety doesn't pause |
| Disambiguation timeout | 10 seconds, then auto-pick most-observed | Don't trap the user |
| Temporal consistency | 3-frame window, 2-of-3 hits | Single-frame YOLO results flicker |
| Reached criterion | Temporal-confirmed AND region=center AND distance=near | All three needed |
| Reached re-firing | One event per waypoint | Avoids duplicate announcements |
| Detection timeout | 60 seconds with no match → recovery dialog | Navigation distances are bigger than OA |
| Skip command | Mark SKIPPED, advance; if no next, ask for one | Most useful recovery path |
| Missing region/distance | Log warning once, treat as unknown, fail safe | Lets Ahmad and me develop independently |

### Part 3 — Obstacle awareness

| Decision | What we do | Why |
|---|---|---|
| Obstacle classification | Broad set (~50 classes) + size gate for small objects | Safety intent + no false alarms about distant cups |
| Active landmark exempt | Never warn about the thing user is walking toward | Self-defeating to warn about the goal |
| Person handling | Distinct phrasing, highest priority | People are dynamic hazards |
| Path region | Only `region == "center"` | User walks forward, side objects aren't in the way |
| Distance gate | Only `distance_category == "near"` warns | Matches advisory-not-realtime constraint |
| Multiple obstacles in path | Warn about highest priority + nearest only | One clear warning beats a list |
| Temporal consistency on obstacles | Yes — same 3-frame, 2-of-3 rule as landmarks | One consistency mechanism |
| Preemption | Obstacle suppresses reached announcement AND guidance for that frame | Per the implementation plan |
| Reached state during obstacle | Continues tracking silently | Landmark may still legitimately be reached |
| Obstacles during disambiguation | Still fire | Safety doesn't pause |
| Obstacles during waypoint collection | Off (camera off) | Preserves task-scoped perception principle |
| Warning phrasing | Class-specific: "Person ahead", "Chair ahead", "TV ahead" | Informative |
| Throttling | Per-class cooldown 3 seconds | Repeating every 200ms is unusable |
| Different class bypasses cooldown | Yes — new hazard type is new information | "Person ahead" after "Chair ahead" matters |
| Priority tie-break | Person > pet > furniture; then by bbox area | Person most urgent; static furniture least |
| Constrained escalation | 3 levels: mild → "still ahead" → "still in path, please be careful" | Slightly more urgent if user isn't reacting |
| Escalation reset | When obstacle clears, reset level + cooldown | New episode = fresh level 0 |
| Confidence threshold | Same 0.6 as landmarks | One threshold to reason about |
| Advisory framing | Calm phrasing, no "STOP" | Over-promising safety is harmful |

### Part 4 — Guidance generation

| Decision | What we do | Why |
|---|---|---|
| When to speak | Only when region OR distance changed | Plan: "speak only when spatial region or distance category changes" |
| Temporal consistency required | Yes — same 2-of-3 as reached | Trust over speed |
| During obstacle warning | Suppressed for that frame | Obstacle preemption applies to guidance too |
| During disambiguation | Paused completely | Guiding to one of two ambiguous landmarks is misleading |
| Direction phrasing | "on your left" / "on your right" / "ahead" | Natural egocentric directions |
| Distance phrasing | "far away" / "approaching" / "right in front of you" | Qualitative |
| Class name in cues | Include the class name; re-say on class change | Working memory; class-change forces fresh cue |
| Phrasing smoothing | "Chair far ahead." not "Chair ahead, far away." | Avoids awkward double-words |
| Distance-only change | "Getting closer" for medium; "right in front of you" for near | Progress indicator |
| Last-spoken state location | Fields on NavigationContext | Task-scoped; cleared on advance |
| Logic location | Inside `LandmarkDetector` | Per-frame decisions in one place |
| Phrasing location | Inside manager `_on_guidance` | Detector stays a pure observer |
| Near but not centered | Guidance fires (course-correction moment) | Most useful guidance scenario |
| Lost sight | "Lost sight of the chair. Looking for it." (once) | Then reset spoken state for fresh return cue |

### Part 5 — Integration, completion & end-to-end

| Decision | What we do | Why |
|---|---|---|
| Final waypoint reached | Speak "You've reached X" then "What's next?" dialog | Lets users add more landmarks iteratively or finish cleanly |
| "Done" recognition | Multiple phrasings: "done", "finished", "all done", "that's all" | Natural language; no memorization |
| "What's next?" timeout | 30 seconds → "Navigation finished." → complete | Silence = done |
| FSM completion event | `task_completed` with structured summary payload | Distinct from `task_cancelled` for evaluation |
| Summary payload | destination, waypoints_total, waypoints_reached, waypoints_skipped, duration_seconds | What Sprint 5 evaluation needs |
| Mid-task waypoint reached | Only terminal events go to FSM | Keeps FSM contract minimal |
| "I'm here" override | Accept user override, but ask for confirmation with current observation | User autonomy + transparency |
| Arrival confirmation answer | Yes → mark reached; No → keep going; Ambiguous → default yes | Charitable interpretation |
| Arrival confirmation timeout | 30 seconds → default to yes | Consistent with charitable interpretation |
| "I'm here" before any waypoint | Re-prompt the waypoint question | Nothing to arrive at |
| Re-issued "navigate to" mid-task | Cancel current silently, start fresh | Matches voice-assistant expectations |
| Async TTS support | Auto-detect coroutine return from `synthesize()` | Supports both sync and async TTS |
| Detection dict adapter | Accept variations on key names | Defensive integration |
| Wiring helper | `create_navigation_session(...)` one-liner | Single import for partners |
| Logging | Standard Python `logging` module | Doesn't fight with Omar's logging setup |
| End-to-end smoke test | One scripted scenario in CI | Catches regressions in data flow |
| Demo commands | Added `/here`, `/done`, `/status` | Verify new flows interactively |
| Thread safety | Not thread-safe; documented assumption | Python GIL + asyncio naturally serializes |
| FSM event dispatch failure | Log and continue | Graceful degradation |
| Persistent storage | Out of scope — structured logs suffice | Navigation shouldn't own analytics |
| Status query API | `nav.get_status()` returns a snapshot dict | Useful for UI/debugging |

---

## 11. The code files

| File | Approx lines | What's in it |
|---|---|---|
| `navigation/__init__.py` | 170 | Public API |
| `navigation/context.py` | 270 | Data classes, status enum, all state, summary() |
| `navigation/parser.py` | 200 | Pure text helpers: normalize, split, cancel/arrival/yes/no/done |
| `navigation/landmark_map.py` | 252 | Alias → COCO resolver, 78-entry table |
| `navigation/obstacle_map.py` | 145 | Obstacle classification, size gate, priority |
| `navigation/detector.py` | 580 | Per-frame engine: obstacles, temporal consistency, disambiguation, guidance, reached, timeouts |
| `navigation/manager.py` | 870 | Orchestrator: dialog, callbacks, phrasings, completion, override |
| `tests/` | — | 294 tests across 13 files |
| `scripts/nav_demo.py` | 380 | Interactive demo with all Parts 1-5 commands |
| `scripts/regen_snapshots.py` | 40 | Regenerates landmark-map snapshot |

---

## 12. Tests

294 automated tests, all passing in ~0.3 seconds.

| Test file | What it covers |
|---|---|
| `test_navigation_context.py` | Waypoint dataclass, context queue helpers |
| `test_navigation_parser.py` | Text normalization, multi-waypoint splitting, cancel/arrival |
| `test_navigation_parser_part5.py` | Yes/no/done phrase detection |
| `test_navigation_manager.py` | Part 1 dialog: prompt, re-prompt, cancel, timeout |
| `test_navigation_landmark_map.py` | Exact/substring/fuzzy resolution; snapshot test |
| `test_navigation_detector.py` | Part 2 detector: filtering, temporal, reached, disambiguation, timeouts |
| `test_navigation_manager_part2.py` | Part 2 integration: eager resolution, partial recognition, full disambiguation |
| `test_navigation_obstacle_map.py` | Obstacle class membership, size gate, priority |
| `test_navigation_obstacles.py` | Detector obstacle logic: filter, cooldown, escalation, preemption |
| `test_navigation_manager_part3.py` | Part 3 phrasings: person/furniture, escalation, preemption |
| `test_navigation_guidance.py` | Part 4 detector: change detection, lost sight, preemption, disambig pause |
| `test_navigation_manager_part4.py` | Part 4 spoken phrasings |
| `test_navigation_manager_part5.py` | Part 5: completion, arrival override, status, dict adapter, FSM resilience |
| `test_navigation_e2e.py` | End-to-end: full sessions with reached + obstacle + override + cancel |

---

## 13. How to run it yourself

### Install

```bash
unzip lumen_navigation_final.zip -d lumen
cd lumen
pip install -r requirements.txt
```

### Run the tests

```bash
cd backend
python -m pytest tests/ -v
```

Expected: `294 passed`.

### Interactive demo

```bash
python scripts/nav_demo.py
```

Commands:

- `/state` and `/status` — inspect current state
- `/detect <cls> [region] [dist] [conf]` — simulate one detection (use underscore for multi-word like `dining_table`)
- `/detect_two <a> <b>` — two classes at once (for disambiguation)
- `/detect_off` — frame with no detections
- `/obstacle <cls> [region] [dist]` — simulate an obstacle
- `/here` — simulate "I'm here"
- `/done` — simulate "done"
- `/skip` — simulate "skip"
- `/timeout` — fast-forward the active timer
- `/reset` — cancel and restart
- `/quit` — exit

### Full session you can try

```
User says 'navigate to ___':  kitchen
user>  the chair and the table
user>  /detect chair right far
user>  /detect chair right far
user>  /detect chair right medium
user>  /detect chair center near
user>  /detect dining_table center far
user>  /detect dining_table center far
user>  /detect dining_table center near
user>  done
```

Expected TTS sequence: "Looking for chair" → "Chair on your right, far away" → "Chair on your right, getting closer" → "Chair right in front of you" → "You've reached the chair" → "Looking for table" → "Dining table far ahead" → "Dining table right in front of you" → "You've reached the table" → "What's next?" → "Navigation complete." Plus the FSM `task_completed` event with the full summary.

### "I'm here" override

```
User says 'navigate to ___':  kitchen
user>  the chair
user>  /here
user>  yes
user>  done
```

System asks for confirmation since the detector hasn't seen the chair; user confirms; navigation completes.

---

## 14. Known limitations and Sprint 5 ideas

1. **COCO has no architectural classes.** "doorway"→"door" is best-effort; "wall"/"hallway"/"stairs" map to nothing. Reserved for a Sprint 5 fine-tune on a custom indoor dataset.

2. **Turning instructions are deferred.** "turn left" is currently treated as an unrecognized landmark. Sprint 5 could add a directional-phrase vocabulary if real-user feedback shows demand.

3. **Obstacle warnings are advisory, not collision-avoidance.** The report itself acknowledges perception latency makes real-time safety impossible. Phrasing is calm; no "STOP."

4. **Disambiguation adds a dialog turn mid-walk.** Deliberate trade-off in favor of user autonomy.

5. **No empirical evaluation yet.** Reached-criterion thresholds, temporal window, obstacle cooldowns, guidance phrasings — all reasoned, none measured. Sprint 5 evaluation may tune them.

6. **Manager is not thread-safe.** Documented assumption; partners run a single session loop.

7. **No persistent session storage.** Structured INFO logs include everything Sprint 5 evaluation needs.

---

*Lumen Navigation task — Parts 1, 2, 3, 4, 5 complete. 294 tests passing. Ready for full-stack integration.*
