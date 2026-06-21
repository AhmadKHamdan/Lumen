# Lumen — Goal-Directed Exploration: Implementation Progress

> **What this is.** A status + how-it-works report on the **exploration prototype** —
> the system where the user says only a destination ("get me to the kitchen") and
> Lumen guides them there room-by-room. It is the implementation of the
> [Exploration Navigation Design](Exploration_Navigation_Design.md) (the M0–M5
> roadmap). Read this to understand what is built, how it works, and what is left.
>
> **Audience:** a teammate who knows the project but hasn't been in the build sessions.
>
> **Companion docs:** [Exploration_Navigation_Design.md](Exploration_Navigation_Design.md)
> (the agreed design) and [Door_Model_Training_Results.md](Door_Model_Training_Results.md)
> (the door detector).

---

## Table of contents

1. [Status at a glance](#1-status-at-a-glance)
2. [How this relates to the original design](#2-how-this-relates-to-the-original-design)
3. [What a full session looks like today](#3-what-a-full-session-looks-like-today)
4. [Architecture — the two-phase state machine](#4-architecture--the-two-phase-state-machine)
5. [How we built each piece (and why)](#5-how-we-built-each-piece-and-why)
6. [The reliability philosophy](#6-the-reliability-philosophy)
7. [What's left](#7-whats-left)
8. [Tuning constants reference](#8-tuning-constants-reference)
9. [How to run it](#9-how-to-run-it)
10. [Known limitations & risks](#10-known-limitations--risks)
11. [File map](#11-file-map)

---

## 1. Status at a glance

| Milestone (from the design doc) | Delivers | Status |
|---|---|---|
| **M0 — Door model** | A detector that sees doors | ✅ **Done.** `best.pt`, single-class, mAP@50 **0.946**, recall 0.89. A second 4-class model (door/handle/cabinet/fridge-door) is also trained and used as a *verifier*. |
| **M1 — Semantic arrival** | Point at a fridge → "we've reached the kitchen" | ✅ **Done.** Goal→indicator map + primary/secondary arrival rule. |
| **M2 — Single-door exploration** | One door per room: scan → walk through → scan → arrive, end to end | ✅ **Done & live-tested.** Full loop runs two rooms deep on a phone. |
| **M3 — Multi-door + human hints** | "I see two doors — which way?" → "left" → go | 🟡 **Partial.** When several doors are seen, one is auto-chosen (most-sighted, tie-break toward straight-ahead). Asking the user, parsing direction hints, and not re-entering the door you came from are **not yet built**. |
| **M4 — Memory + backtracking + IMU** | Dead-end backtracking, breadcrumbs | 🟡 **Partial-by-accident.** The phone orientation sensor (an M4 prerequisite) was pulled forward into M2 and is fully wired. Backtracking/breadcrumbs/room-cap are not built. |
| **M5 — Stretch** | Door open/closed, complex layouts | ⬜ Not started. |
| **Obstacle awareness** (was design "Part 3") | Warn about things in the walking path | ✅ **Done — and upgraded.** Two-layer watchdog: YOLO named-object corridor + monocular-depth tripwire that catches *unnamed* clutter (clothes piles, boxes). Added this session as a critical safety task. |

**Bottom line:** M0, M1, M2 and the obstacle layer are complete and demonstrated on a
phone. The remaining work is M3 (multi-door decision-making) and M4 (robust memory),
plus the polish items in §7.

---

## 2. How this relates to the original design

The [design doc](Exploration_Navigation_Design.md) proposed building exploration *on top
of* an earlier **waypoint-mode** system — injecting each chosen door into a waypoint
queue and reusing its reached/guidance logic. **In practice we did not do that.** The
exploration prototype is a **self-contained real-time controller** (the `webdemo/lumen/`
package) that implements its own scanning, perception, distance, and guidance from scratch.

**Why the divergence.** The waypoint manager assumed someone upstream already produced
clean per-frame detections tagged with region/distance. The exploration prototype had to
*be* that upstream — running the models, fighting false positives, doing compass geometry.
Bolting that onto the waypoint manager would have meant gutting it; a focused controller
that owns the whole pixels-to-speech path was simpler and let us iterate fast against real
phone footage.

The one piece worth keeping from the old system — the goal→room-indicator map — has been
**vendored into the package** as [`webdemo/lumen/goals.py`](webdemo/lumen/goals.py). The
waypoint code itself has been removed, so **`webdemo/` is now the entire system with no
external dependencies beyond pip packages.**

---

## 3. What a full session looks like today

This is a real run, lightly narrated. The phone shows only the clean camera feed; the
audience hears the voice.

```
USER:  "get me to the kitchen"
LUMEN: "Looking for the kitchen. Let's scan the room — slowly turn to your right,
        all the way around, until you are facing where you started."
        (user turns slowly; Lumen is silent except brief progress nudges:)
LUMEN: "Good, keep turning to your right."  ...  "Halfway around, keep turning."
        (compass confirms a full circle back to the start direction)
LUMEN: "You're back where you started — scan complete. I found a door behind you,
        to the right. Turn toward it and point your camera at it, so I can guide
        you in precisely."
        (user rotates; Lumen walks them through the turn)
LUMEN: "The door should be right ahead of you now. Let's go to it."
LUMEN: "The door is on your right, about 4 steps away. Walk forward, and after
        about 1 step reach out with your hand."
        (a chair sits in the path)
LUMEN: "Stop. There's a chair in your way. Step to your left."   ← obstacle overrides
LUMEN: "Okay, the way ahead is clear."
LUMEN: "You're right at the door. Reach out, open it, and walk through."
        (user walks through; door leaves the frame)
LUMEN: "You've gone through the doorway. Take two or three steps into the room,
        then slowly turn to your right, all the way around, back to where you
        started, so I can scan this room."
        (new room scan; a fridge and oven are confirmed)
LUMEN: "You're back where you started — scan complete. I found a refrigerator on
        your right and an oven ahead. You've reached the kitchen."   ← task done
```

Every line above is produced by the state machine in §4.

---

## 4. Architecture — the two-phase state machine

The whole controller lives in [`webdemo/server.py`](webdemo/server.py). The browser
([`webdemo/index.html`](webdemo/index.html)) just captures the rear camera, reads the
compass, sends one frame (~3/sec) to `POST /detect`, and speaks whatever text comes
back. All intelligence is server-side.

The design's two-phase principle — **Phase 1: silent scan that sets flags; Phase 2: act
on the flags by priority** — is implemented as five modes:

```
                       ┌──────────────────────────────────────────────┐
        "kitchen"      │  discover  (Phase 1: silent guided 360 scan)  │
   ───────────────────▶│  Collect, per direction: door bearings,       │
                       │  indicator sightings. Speak only turn/slow     │
                       │  nudges. End ONLY when the compass confirms a   │
                       │  full circle back to the start heading.        │
                       └───────────────┬──────────────────────────────┘
                                       │ scan complete → pick by priority
              ┌────────────────────────┼─────────────────────────────┐
              │ strong indicator         │ weak indicator      │ door only
              ▼                          ▼                     ▼
        ┌───────────┐          ┌──────────────────┐    ┌──────────────────┐
        │  arrived  │          │   face_target    │    │   face_target    │
        │ declare,  │          │ turn to face the │    │ turn to face the │
        │ task ends │          │ sighting, then…  │    │ door, then…      │
        └───────────┘          ▼                       ▼
                          ┌──────────────┐        ┌──────────────┐
                          │ go_indicator │        │   go_door    │
                          │ confirm there│        │ guide in:    │
                          │ → arrived,   │        │ steps + hand │
                          │ or fall to   │        │ cue; obstacle│
                          │ door / rescan│        │ watchdog ON  │
                          └──────────────┘        └──────┬───────┘
                                                          │ walked through doorway
                                                          ▼  → reset to discover (new room)
```

**Priority is absolute: indicators always beat doors.** If the scan confirms the goal's
objects, we arrive — even if a door was also seen. We never walk you out of the room
you were trying to reach.

**The five modes:**

- **`discover`** — Phase 1. One guided 360° turn. Sets flags + bearings, speaks nothing
  but turn coaching. (Details in §5.1.)
- **`face_target`** — Phase 2 opener. Actively rotates the user to face the chosen
  target (door *or* indicator), using the live compass.
- **`go_door`** — walk the user to a door (distance + hand-reach cue), watch for
  obstacles, detect the walk-through.
- **`go_indicator`** — focused re-confirmation of a *weak* indicator sighting before
  declaring arrival.
- **`arrived`** — terminal; the arrival line is spoken and the journey ends.

---

## 5. How we built each piece (and why)

Most of the engineering effort this session went **not** into the happy path but into
making perception trustworthy enough to point a blind person at. Each subsystem below
exists because a specific real-world failure forced it.

### 5.1 The compass-anchored 360° scan (`discover`)

**What.** On entering a room, Lumen asks for one slow full turn to the right. The phone
compass (`DeviceOrientation`) gives an absolute heading each frame. The **first heading
is locked as the anchor** ("ahead" = where you started). Every detection is filed into
one of **12 × 30° sectors**, and every door/indicator gets a **true bearing** =
`camera_heading + (object's horizontal position in frame − 0.5) × field-of-view`.

**Why each design choice:**

- **Start direction as the fixed anchor.** All directions in the summary ("a door behind
  you, to the right") are relative to where the scan *began*, so they stay meaningful.
- **Completion requires BOTH a full ~350° of accumulated rotation AND a compass-confirmed
  return to within 25° of the start heading.** An earlier version ended on "all 12
  sectors seen", which compass noise could fake mid-turn — ending the scan while the user
  faced the wrong way and computing every direction from a broken premise. Now the scan
  cannot end until you are physically back where you started, which also makes the spoken
  directions true *at the moment they're heard*.
- **Bearings, not just sectors.** Filing a door by camera-heading alone ignored where the
  door sat in the frame — a door at the right edge got mislabeled. Adding the in-frame
  offset fixed "said left, was right".
- **Clustering.** One physical door spans several frames/sectors as you pan. Sightings
  within 30° are merged into one cluster, so a door is announced **once** with one precise
  heading (and a door directly behind you, which straddles the ±180° seam, merges
  correctly instead of becoming "two doors behind you").
- **Motion gate.** If the phone pans too fast the frame blurs; we skip detection on that
  frame and coach "slow down" — but still advance the rotation total so a fast sweep
  doesn't stall completion.

### 5.2 The door perception stack — the centerpiece

Doors are the single most safety-critical object (a false door points a blind user at a
wall or a curtain). A plain confidence threshold is **not** enough — walls, curtains, and
fridges all fire as high-confidence doors. So every door candidate passes through a
**three-layer funnel**, and we tightened it failure-by-failure against real phone footage:

**Layer 1 — geometry gates** (cheap, kill obvious non-doors):

| Gate | Rejects | The failure that motivated it |
|---|---|---|
| width/height ≤ 1.4 | wide wall spans | — |
| frame-fill ≤ 0.80 (during scan) | whole-scene "doors" | A blank wall boxed edge-to-edge scored 0.84. A real door always leaves wall/floor margin during a scan. |
| edge-density (Canny) | smooth blank walls | Walls have almost no internal edges; doors have frames/seams/hinges. |

A subtlety we hit: **edge-density is not a hard reject.** A plain door in dim light is as
smooth as a wall (both ~0.01). So a smooth candidate isn't dropped — it's *demoted* to
require semantic proof in Layer 2. (Confidence can't break the tie either: walls hit 0.9.)

**Layer 2 — semantic verification** (the 4-class DoorDetect model as a second opinion):

- A **strong** candidate (conf ≥ 0.60, textured) passes on its own.
- A **weak or smooth** candidate passes **only if corroborated** — the verifier also sees
  a `door` there, or a `handle` inside the box. *This is what killed the curtain:* a lace
  curtain is edge-rich, tall, and door-sized (beats every geometry gate) but has no handle
  and isn't a door to the verifier.
- If the verifier sees a `refrigerator door` on the candidate, it's a fridge, not a door.

**Layer 3 — fridge arbitration** (the most dangerous failure): a real door at an angle is
often called `refrigerator` by COCO — and a fake fridge is a *kitchen indicator*, which
can cause a **false arrival**. So a COCO "refrigerator" that overlaps a door candidate is
kept **only if** the verifier confirms a `refrigerator door` there; otherwise it's dropped
as "it's the door, not a fridge".

**Reliability gates on top of all that:** a door must be sighted in **≥3 frames** to be
named or chosen; an object mention needs **≥3 sightings per direction**. A one- or
two-frame flicker is never spoken or navigated to.

> One honest caveat carried forward: in some rooms the user's **walls fool all three
> models at once** (door model + COCO + verifier). The gates contain it, but the durable
> fix is a hard-negative fine-tune — see §7.

### 5.3 Distance + the hand-reach cue (`go_door`)

Door distance comes from a **pinhole model**: a door is ~2 m tall, so its pixel height
gives range (`distance = real_height × focal_px / pixel_height`). The spoken output
converts to steps (~0.75 m each) and adds a tactile cue: within 3 steps, "reach out with
your hand"; farther, "walk N−3 steps, then reach out."

Two corrections we made:
- **Focal axis.** Focal length was derived from image *width*, but a camera's quoted FOV
  belongs to its *longer* axis — phones stream portrait, so every distance was ~33% short.
  Fixed to use the long side.
- **Tighter box.** The primary door model's boxes run loose at range (wall above/below),
  and a fat box reads as "near". When the verifier also boxed the door, we measure on the
  shorter of the two boxes. There's also a one-shot calibration constant `DIST_CAL` (a
  single ground-truth measurement scales the whole chain); currently 1.0 and judged "quite
  good" in testing.

### 5.4 Doorway transit (the loop that makes it multi-room)

Detecting "the user just walked through the door" is unsolved by a single signal, so we
fuse several: while in `go_door`, if a tracked approach got the door close (metric
distance **or** the confirmed door grew to fill ~half the frame) **and then** a saturated
door-shaped box fills the view, we latch "at the door"; when the door then disappears for
a few frames, we infer the walk-through and reset to a fresh `discover` for the new room.

This was the trickiest piece. Two failures we fixed:
- At arm's length a door is a smooth panel → the edge gate rejected it → the "at the door"
  latch never armed. Fixed with a distance-independent "the door grew to half-frame" signal.
- New rooms throw their *own* full-frame door candidates, which held the latch open forever
  after you'd already walked through. Fixed with a hold cap (~7 s) that forces the transit
  inference rather than getting stuck in `go_door`.

### 5.5 Indicator arrival & directed confirmation

- **Strong evidence → instant arrival.** If the 360 scan confirms the goal's objects
  (≥`INDICATOR_HITS` sightings **localized to one 90° window** — see below), arrival is
  declared right at scan end. No second ceremony; the scan *is* the confirmation.
- **Weak evidence → directed confirmation.** If a fridge was glimpsed but below the bar,
  Lumen does for indicators exactly what it does for doors: "That might be the kitchen —
  turn toward it and point the camera there," then re-checks. Confirmed → arrive; not
  confirmed → fall to the door branch (the indicator had its chance) or rescan. This
  replaced a confusing "…but nothing I can act on" dead-end.
- **Localized evidence.** Arrival used to sum sightings globally, so two wall-flickers here
  and two there could fake a "refrigerator". Now a class only counts if `INDICATOR_HITS`
  sightings land within one 90° window — a real fridge passes trivially (you sweep past it,
  it fires every frame in one spot); scattered noise can't.

### 5.6 The obstacle watchdog (added this session as a critical task)

Armed **only while walking to a door** (`go_door`), because that's the only phase the user
moves. Two layers, fused, with their own top speech priority (an obstacle interrupts even
door guidance):

- **Phase A — YOLO corridor.** During `go_door` we also request COCO obstacle classes
  (person, chair, couch, table, etc.). Anything whose box sits low (near the floor) and
  overlaps the central "walking lane" triggers "Stop. There's a chair in your way. Step to
  your left."
- **Phase B — monocular depth tripwire (Depth Anything V2 Small).** COCO can't see the
  clutter that actually litters floors — clothes piles, boxes, bins, a standing fan (all
  present in test rooms, none in COCO). The depth model returns per-pixel relative depth;
  we define a floor "lane" just ahead of the feet and flag any patch that reads *much
  nearer than the side-floor at the same height* — **class-agnostic**, names nothing, just
  "something is close, step aside." It self-calibrates each frame (bottom strip = near
  anchor, top strip = far anchor) so the model's absolute scale never matters, and it's
  skipped when the scene is too flat to judge (facing a near wall).

YOLO names the obstacle when it can ("a chair"); depth covers everything else ("something").
Both are debounced (warn after 2 consecutive frames, re-warn every ~3 s while blocked, one
"path is clear" when it ends). The depth model is **optional** — if `torch`/`transformers`
aren't installed it degrades to YOLO-only.

### 5.7 Speech discipline

A blind user can't see a garbled transcript, so overlapping prompts are a real bug, not
cosmetic. Rules enforced:
- **Two tiers:** *priority* lines (phase changes, arrival, obstacles) interrupt; *ambient*
  nudges (keep-turning, distance updates) never interrupt and de-dupe.
- **Never two priority lines back-to-back** — announcement + instruction are merged into
  one utterance (a `skip_scan_prompt` flag stops the next phase from repeating an
  instruction already spoken).
- The goal echo ("Looking for the kitchen") is folded into the first scan instruction, not
  spoken separately, so it can't be cut off.

---

## 6. The reliability philosophy

One principle drove most decisions and is worth internalizing before you touch this code:

> **A false positive is worse than a false negative.** Missing a real door costs a re-scan
> (annoying). Calling a wall a door points a blind person at it (dangerous). So every gate
> biases toward rejection, and we'd rather say "let's scan again" than guess.

Consequences you'll see throughout:
- Layered, independent checks instead of one tunable threshold.
- Evidence must be **repeated** (multi-frame) and **localized** (one direction) before it's
  spoken or acted on.
- A second model (the verifier) gets a veto on the first model's confident mistakes.
- Confidence alone is never trusted to separate look-alikes (wall vs door, door vs fridge).

`DOOR_DEBUG = True` prints every decision to the server terminal (`[door] … KEEP/reject`,
`[obj] …`, `[depth] …`, `[dist] …`). It's intentionally left on — it's how we diagnose
every failure — and has zero effect on behavior. The phone UI is separately cleaned for
demos (`SHOW_DEBUG_UI = false` in `index.html` hides all boxes and text).

---

## 7. What's left

### Immediate polish
- **One clean verification run** of the two newest changes (weak-indicator directed
  confirmation in §5.5; the obstacle layer in §5.6) to confirm no regressions.
- **`DIST_CAL` calibration** — currently 1.0 and "quite good"; one ground-truth step-count
  measurement would lock it exactly. The `[dist]` debug line gives the input.
- **`DOOR_DEBUG`** — flip to `False` for the final demo once detection is settled (kept on
  deliberately for now).

### M3 — multi-door + human hints (the real next milestone)
Currently, when a scan finds several doors, Lumen **auto-picks** the most-sighted one
(tie-break toward straight-ahead). To finish M3:
- **Ask when genuinely ambiguous:** "I see two doors — one ahead, one on your left. Which
  way is the kitchen?"
- **Direction-hint parsing:** let the user say "the kitchen is to the left" (at the start
  or at a fork) and bias door choice toward that bearing. The compass machinery to convert
  a spoken direction to a heading already exists.
- **Come-from-door exclusion (most important, and the thing that blocks real multi-room
  journeys):** after a transit, the door you just walked through is behind you and *will*
  be re-found by the new room's scan. Without memory, Lumen can ping-pong between two rooms
  forever. The compass gives this nearly for free — the entry door sits ~180° from your
  entry heading; deprioritize it.
- **Tried-door memory:** don't re-pick a door that led to a goal-less room.

### M4 — robust memory & backtracking
Breadcrumb stack of entry headings, dead-end backtracking, a room-visit cap with a
graceful "do you want to stop, or name the landmarks yourself?" hand-off. The IMU/compass
prerequisite is already done.

### M5 — stretch
Door open/closed state, complex layouts, traversal polish.

### Durable perception fix (recommended before a high-stakes demo)
A **hard-negative fine-tune**: 30–50 photos of the test environment's walls, curtains, and
wardrobes (empty labels = "nothing here"), added to the dataset, ~20 epochs on the GPU for
both door models. This teaches the models that these surfaces are *not* doors, instead of
fencing around the weakness with gates — and would let us *relax* some gates. ~1 hour of
work; high value because M3/M4 add more rooms (more walls) per journey.

### Engineering hygiene
- The exploration controller has **no automated tests** (it's been validated live). The
  pure-geometry helpers (clustering, bearing math, direction mapping) are easily unit-
  testable and worth covering.
- If we ever want a single engine, reconcile the exploration controller with the waypoint
  manager (§2).

---

## 8. Tuning constants reference

All in [`webdemo/lumen/config.py`](webdemo/lumen/config.py) — the one place to tune
behavior. The values that most affect it:

| Constant | Value | Controls |
|---|---|---|
| `INDICATOR_HITS` | 4 | sightings (in one 90° window) to confirm arrival |
| `MOTION_MAX` | 60 | blur threshold for the "slow down" gate |
| `DOOR_CONF` | 0.45 | door model confidence floor |
| `DOOR_MAX_WH` | 1.4 | max door width/height (rejects wide walls) |
| `DOOR_EDGE_MIN` | 0.030 | edge density below which a door needs verifier proof |
| `DOOR_MAX_FRAME_FRAC` | 0.80 | frame-fill above which a "door"/"indicator" is a wall latch |
| `DOOR_STRONG_CONF` | 0.60 | confidence above which a door passes without corroboration |
| `VERIFY_CONF` / `VERIFY_IOU` | 0.30 / 0.40 | verifier sensitivity / overlap to count as "same object" |
| `DOOR_MIN_SIGHTINGS` | 3 | frames before a door is named/chosen |
| `OBJ_MIN_SIGHTINGS` | 3 | sightings before an object is mentioned |
| `BUCKETS` / `DOOR_CLUSTER_DEG` | 12 / 30° | scan sectors / sighting-merge width |
| `FULL_TURN_DEG` / `START_TOL` | 350 / 25 | rotation + heading tolerance to finish a scan |
| `FACE_TOL` | 25 | "facing the target" tolerance |
| `DOOR_HEIGHT_M` / `ASSUMED_HFOV_DEG` / `STEP_LENGTH_M` | 2.0 / 60 / 0.75 | distance model inputs |
| `DIST_CAL` | 1.0 | one-shot distance calibration multiplier |
| `HAND_REACH_STEPS` | 3 | within this many steps → "reach out now" |
| `NEAR_DOOR_M` / `APPROACH_FRAC` / `NEAR_HOLD_MAX` | 3.5 / 0.5 / 20 | doorway-transit detection |
| `CORRIDOR_X` / `OBST_*` | (0.30,0.70) / … | obstacle walking-lane + debounce |
| `DEPTH_*` | … | depth tripwire lane, anchors, intrusion thresholds |
| `DOOR_DEBUG` | True | print every detection decision to the terminal |

---

## 9. How to run it

**Dependencies** — all in `requirements.txt`:

```bash
pip install -r requirements.txt
# fastapi, uvicorn, opencv-python, numpy, ultralytics, pillow (+ optional torch,
# transformers, which enable the Phase-B depth obstacle layer — without them the
# obstacle watchdog runs YOLO-only and nothing else changes).
```

YOLO weights (`yolov8m.pt`) download on first run; `best.pt` (door) and the 4-class
verifier are already in the repo.

**Start the server:**
```bash
python webdemo/server.py        # serves on http://127.0.0.1:8000
```

**Test on a phone** (needed for the camera, mic, and compass; HTTPS required):
```bash
ngrok http 8000                 # open the https URL on the phone, tap Start, hold Talk
```

On a laptop without a compass it still works — it falls back to a single steady-pan scan
(no directional summary). The full experience (360 scan, directions, transit) needs the
phone's orientation sensor.

---

## 10. Known limitations & risks

1. **No localization.** A phone camera can't map a building, so deep multi-room journeys
   will drift until M4's breadcrumbs land. Mitigated by the (coming) come-from-door
   exclusion and the always-available user hints.
2. **Walls can fool all three models in some rooms.** Contained by the gates today;
   §7's fine-tune is the durable fix.
3. **Door traversal is advisory, not collision-avoidance.** Lumen complements a white
   cane (which covers the ~1 m ground zone); its value is the 2–4 m pre-warning. We do not
   claim real-time safety.
4. **Obstacle depth is relative, not metric.** It reliably says "something close, step
   aside" but not an exact distance; that's the right trade for a browser prototype.
5. **No automated tests on the controller** — validated live only (§7).
6. **Single-user state.** The server holds one global `_state`; it serves one phone at a
   time. Fine for the demo, not multi-tenant.

---

## 11. File map

| File | Role |
|---|---|
| [`webdemo/server.py`](webdemo/server.py) | **Thin FastAPI layer**: routes + the per-frame `detect()` pipeline that wires the `lumen/` package together (decode → motion gate → perception → controller → response). ~150 lines. |
| [`webdemo/lumen/config.py`](webdemo/lumen/config.py) | Every tuning constant, grouped by concern. The one place to adjust behavior. |
| [`webdemo/lumen/goals.py`](webdemo/lumen/goals.py) | Goal→room-indicator map + the arrival rule (vendored; the destination knowledge). |
| [`webdemo/lumen/models.py`](webdemo/lumen/models.py) | Loads + warms up the four models (COCO / door / verifier / depth). |
| [`webdemo/lumen/state.py`](webdemo/lumen/state.py) | The FSM `_state` + per-room evidence buffers + the `enter_*` phase helpers. |
| [`webdemo/lumen/geometry.py`](webdemo/lumen/geometry.py) | Compass/bearing math: start-anchored directions, sighting clustering, turns (pure, unit-testable). |
| [`webdemo/lumen/perception.py`](webdemo/lumen/perception.py) | One frame → trustworthy detections: COCO objects, the 3-layer door stack, overlay boxes, door geometry. |
| [`webdemo/lumen/obstacles.py`](webdemo/lumen/obstacles.py) | The walking-lane watchdog: YOLO corridor (Phase A) + depth tripwire (Phase B). |
| [`webdemo/lumen/controller.py`](webdemo/lumen/controller.py) | The two-phase state machine + scan summary/arrival + transit + everything Lumen says. |
| [`webdemo/index.html`](webdemo/index.html) | Phone front-end: camera capture, compass, speech, demo-clean UI (`SHOW_DEBUG_UI`). |
| [`best.pt`](best.pt) | Single-class door detector (M0, mAP@50 0.946). |
| `door_training/runs/detect/door_yolov8s_4cls/weights/best.pt` | 4-class verifier (door/handle/cabinet/fridge-door). |
| [`door_training/`](door_training/) | Dataset conversion + training scripts for both door models. |
| [`Exploration_Navigation_Design.md`](Exploration_Navigation_Design.md) | The agreed design (M0–M5 roadmap). |
| [`Door_Model_Training_Results.md`](Door_Model_Training_Results.md) | Door detector training write-up. |

---

*Exploration prototype: M0–M2 complete and live-tested, obstacle awareness complete,
M3 partially built (auto door-choice; hints/ask/come-from-exclusion pending), M4–M5 ahead.*
