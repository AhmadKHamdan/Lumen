# Lumen — Goal-Directed Exploration Navigation (Design Doc)

> **What this is.** The design for Lumen's next navigation mode: the user names
> only a *destination* ("get me to the kitchen") and Lumen guides them there
> room-by-room — autonomously when it can, asking the user for direction only when
> it genuinely can't decide. This replaces the need to name every landmark.
>
> **Sequence:** this doc → train the door model → implement.

---

## TL;DR — the agreed decision

- **Human-guided semantic exploration.** Lumen explores toward a goal on its own and
  uses the user as a *direction oracle only when perception is ambiguous*.
- **Autonomous by default, human as tiebreaker.** Lumen asks a question only when it
  sees multiple doors and has no other signal. One door → just go. Goal already
  visible → arrive. No needless questions.
- **Arrival is detected by recognizing the room**, not by the user saying "done":
  fridge/oven/sink ⇒ kitchen.
- **Not fully autonomous mapping** — a plain phone camera can't localize, so we
  deliberately *don't* try to build a building map. The human's coarse direction
  hints fill that gap.
- **~70% reuse.** All low-level perception/guidance/obstacle/TTS code stays; only a
  new high-level *controller* is added.

---

## 1. Why not fully autonomous

A phone camera (no depth, no LiDAR, no SLAM/odometry) **cannot know where it is** in a
building. Without localization, true autonomous wayfinding loops and gets lost — it
can walk you back through the door you just used because it has no map. Robust
autonomous indoor nav needs hardware/maps we don't have.

**The fix is the human.** A blind user usually has *some* sense of direction (memory,
having been told). We use that as the missing "compass" — but only when needed.

---

## 2. The core loop

```
USER: "get me to the kitchen"
      │
      ▼
RESOLVE GOAL:  "kitchen" → indicators {refrigerator, oven, microwave, sink}
      │
      ▼
┌─────────────────── per room ───────────────────┐
│ 1. SCAN     "Slowly turn around so I can look." │
│             → detect goal-indicators + doors    │
│ 2. GOAL?    indicators present?                 │
│               YES → "We've reached the kitchen" → DONE
│ 3. CHOOSE   how many candidate doors?           │
│               0  → recovery (ask user for help) │
│               1  → take it (no question)        │
│               >1 → use direction hint, else ASK │
│ 4. GO       guide user to the chosen door       │  ← reuses existing logic
│ 5. CROSS    "Go through the doorway."           │
│ 6. mark visited, loop                           │
└─────────────────────────────────────────────────┘
```

---

## 3. The decision rule (autonomous vs. ask)

Per room, evaluate **in this order** — first match wins:

| # | Condition | Action | Question asked? |
|---|---|---|---|
| 1 | Goal indicators detected | "We've reached the {goal}." → DONE | No |
| 2 | Exactly one unvisited door | Go to it | No |
| 3 | User volunteered a direction ("it's on the left") | Pick the door in that direction | No |
| 4 | Multiple doors, no hint | "I see two doors — which way is the {goal}?" | **Yes** |
| 5 | No doors visible | "I don't see a way out — can you point me to a door?" | **Yes** |

This is the whole point: **the question in rows 4–5 is the exception, not the rule.**
And if Lumen *does* ask but gets no usable answer ("I don't know"), it falls back to
the **nearest unvisited door** (by bbox size) rather than stalling — see §9.3.

---

## 4. New components to build

### 4.1 Goal → indicator map (`goal_map.py`)
Analogous to the existing `landmark_map.py`. Maps a room goal to the COCO/door
classes that indicate it.

```python
GOAL_INDICATORS = {
    "kitchen":     {"primary": ["refrigerator", "oven", "microwave"],
                    "secondary": ["sink", "dining table"]},
    "bathroom":    {"primary": ["toilet"],
                    "secondary": ["sink"]},
    "bedroom":     {"primary": ["bed"], "secondary": []},
    "living room": {"primary": ["couch", "tv"], "secondary": ["potted plant"]},
    "office":      {"primary": ["laptop", "keyboard"], "secondary": ["tv", "mouse"]},
}
```
**Arrival rule:** declare the room when **≥1 primary** indicator is temporally
confirmed (2-of-3 frames), OR **≥2 secondary**. This avoids false positives — e.g. a
lone `sink` shouldn't trigger "kitchen" (bathrooms have sinks too); a *fridge* should.

### 4.2 Exploration controller (`exploration.py`)
A state machine — `SCAN → GOAL_TEST → CHOOSE_DOOR → GO_TO_DOOR → TRAVERSE → (loop)` —
that **drives the existing waypoint machinery as its primitive.** When it chooses a
door, it injects that door as a temporary waypoint into the current
`NavigationTaskManager` and lets the existing reached/guidance/obstacle logic walk the
user there. This is the key reuse: the controller decides *which* door; the existing
code handles *getting to* it.

### 4.3 Direction-hint parser (extend `parser.py`)
`parse_direction("the kitchen is on the left")` → `"left"`. Vocabulary:
left / right / ahead / straight / forward / behind / back. Maps to the same
`region` space the detector already uses.

### 4.4 Scan/sweep behavior
A prompt ("slowly turn around") + a short collection window (e.g. 3–5 s) that
aggregates detections across the pan into a per-room snapshot: which indicators seen,
which doors and their directions.

### 4.5 Coarse memory (no SLAM)
- `rooms_visited` counter, with a **cap** so it can't wander forever. On hitting the
  cap Lumen does **not** quit silently — it lets the user choose (see §9.5).
- Track the **direction we entered from** so the entry door is *deprioritized* — not
  banned (see §4.6 backtracking). Heuristic, not a real map — acceptable because the
  human hints correct mistakes.

---

### 4.6 Door buckets & backtracking
Exploration is **depth-first search with backtracking** over an unknown room graph.
Each door seen in a room is one of:

| Bucket | Meaning | Priority |
|---|---|---|
| **Unvisited** | not yet gone through | prefer (explore forward) |
| **Entry** | the door we came in through | backtrack — use when stuck |
| **Exhausted** | went through it, dead-end branch | avoid unless forced |

**Decision order per room:** (1) goal here → arrive; (2) user hint → follow it,
*including "go back"*; (3) nearest unvisited door → go; (4) dead end (no unvisited) →
backtrack through the entry door, mark branch exhausted; (5) all exhausted / room cap
→ ask the user (stop or waypoint mode).

One level of backtracking is reliable (entry door = "behind me"). Multi-level
backtracking needs a **breadcrumb stack of entry headings** — which is why the
**phone orientation sensor** (M4) matters; without it, deep backtracking drifts. The
user can force a backtrack at any time ("go back").

## 5. What we reuse unchanged

| Existing piece | Role in exploration mode |
|---|---|
| Door model (from training plan) | Detect doors + their direction |
| COCO YOLOv8n | Detect goal indicators (fridge, oven, bed, couch…) |
| Spatial reasoning (region / distance) | Which door is left/right/ahead, how near |
| `LandmarkDetector` temporal consistency | Stable door + indicator confirmation |
| "Reached" logic | "You've reached the door" |
| Guidance phrasing | "Door on your right, getting closer" |
| Obstacle preemption + escalation | "Person ahead" still preempts everything |
| TTS + FSM events + per-waypoint perception gate | Unchanged |

Only the **high-level controller** is new. The sensing and the talking are done.

---

## 6. Example sessions

**A) One door — no questions (your example):**
```
USER: "get me to the kitchen"
SYS:  "Looking for the way to the kitchen. Slowly turn around."
      (scan: one door ahead, no kitchen indicators)
SYS:  "There's a door ahead. Heading to it."
SYS:  "Door right in front of you."
SYS:  "Go through the doorway, and let me know once you're through."
USER: "okay, I'm through"     (or Lumen notices the door is no longer in view)
SYS:  "Slowly turn around so I can look."
      (new room scan: refrigerator + oven detected)
SYS:  "I can see a fridge and an oven — we've reached the kitchen."
```

**B) Two doors — one question:**
```
SYS:  "I see two doors — one on your left and one ahead.
       Do you know which way the kitchen is?"
USER: "left"
SYS:  "Heading to the door on your left."
      ... (guides, traverses, scans next room) ...
```

---

## 7. Relationship to the current waypoint mode

Keep both:
- **Goal-only command** ("get me to the kitchen") → **exploration mode** (this doc).
- **Explicit landmark list** ("…through the door, past the table") → existing
  **waypoint mode**.

The command parser routes to one or the other based on whether the user named
intermediate landmarks. Both share the same low-level engine.

---

## 8. Scope, risks & safety (honest)

- **Door traversal is the riskiest part.** Aligning/handling/thresholds for a blind
  user is hard and latency-bound. Keep it minimal ("go through the doorway") and rely
  on the cane. Not collision-avoidance.
- **No localization → it can get confused.** Mitigated by: the room cap, "don't go
  back the way I came" heuristic, the user's direction hints, and an always-available
  "stop."
- **Room mis-ID** (shared objects like sinks) → mitigated by the primary/secondary
  indicator rule.
- **"When am I through the door / in the next room?"** is an unsolved-by-perception
  moment — see open questions.
- **Demoable scope:** a known 2–4 room space, doors openable, user able to give a
  hint when asked. A fully robust building-scale version is research-grade and out of
  scope for the project.

---

## 9. Resolved decisions (from review)

1. **Room-entry detection — combined (a + b).** When guiding through a door, Lumen
   prompts the user to confirm *and* watches the camera: it ends the traversal step
   with "…and let me know once you're through," while also detecting the door
   leaving the frame for a few seconds. Whichever fires first starts the next room's
   scan; the user's spoken confirmation always overrides.
2. **Scan trigger — automatic on every room entry.** A blind user won't know to turn
   unsolicited, so on entering each room Lumen briefly prompts a slow turn ("slowly
   turn around so I can look") and aggregates detections across the sweep into a
   per-room snapshot (doors + their directions, goal indicators).
3. **Door distance — bbox size.** Bounding-box size is the distance proxy (bigger =
   nearer) — the same basis as the existing near/medium/far. It is the **fallback
   door-picker**: if there are multiple doors and the user gives no usable hint,
   Lumen heads to the nearest unvisited door instead of stalling. It does **not**
   replace asking (rule §3.4); it's only the tiebreaker when asking yields nothing.
4. **Hints are per-room.** A direction hint ("it's on the left") applies to the
   current room only. Lumen will not pester the user for a hint at every room — it
   asks only when a room is genuinely ambiguous (multiple doors, no hint).
5. **Give-up — the user decides.** On hitting the room cap, Lumen does not quit
   silently: "I'm having trouble finding the {goal}. Do you want to stop, or name the
   landmarks yourself?" → cancel, or hand back to the existing waypoint mode.

---

## 10. Milestone roadmap (ordered by complexity)

Each milestone is independently demoable; complexity rises with each step, and the
hard, localization-dependent work is deliberately last.

| Milestone | Delivers (demoable) | Complexity | Depends on | Frontend? |
|---|---|---|---|---|
| **M0 — Door model** | Detector that sees doors | Med | datasets | No |
| **M1 — Semantic arrival** | Point at a fridge → "We've reached the kitchen" (one room, no movement) | **Low** | nothing (COCO) | No |
| **M2 — Single-door exploration** | One door per room: through → scan → arrive. One-door, no-questions path end to end | Med | M0 + M1 | No |
| **M3 — Multi-door + human hints** | "I see two doors — which way?" → "left" → go. Scan, describe by direction, within-room tracker | Med–High | M2 | No |
| **M4 — Memory + backtracking + phone IMU** | Dead-end backtrack, breadcrumbs, orientation sensor | **High** | M3 + sensor wiring | **Yes** |
| **M5 — Stretch** | Door open/closed, complex layouts, traversal polish | High | M4 | maybe |

**Key points:**
- **M1 is independent of the door model** — build it in parallel with M0 training.
- **M1–M3 are backend-only and decide locally** (per current scan) — no global map,
  relatively safe and incremental.
- **The complexity cliff is M3 → M4.** All no-SLAM fragility (backtracking,
  breadcrumbs, IMU, frontend changes) lives in M4. By then M1–M3 already work for the
  common cases, so M4 is *robustness*, not make-or-break.
- **Recommended path:** M1 (now, in parallel with training) → M2 → M3 → evaluate
  before committing to M4.

---

*Decision locked: human-guided semantic exploration — autonomous by default, user as
the direction tiebreaker, arrival by room recognition. Train the door model, then
build the controller on top of the perception engine that already exists.*
