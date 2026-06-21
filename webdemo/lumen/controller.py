"""The decision layer: the two-phase exploration state machine and everything Lumen
says. Perception tells it *what is in the frame*; the controller decides *what to do*
and *what to speak*.

Modes (in `_state["mode"]`):
  discover     — Phase 1: silent guided 360 scan, collecting door/indicator bearings.
  face_target  — Phase 2 opener: rotate the user to face the chosen door OR indicator.
  go_indicator — confirm a weak indicator sighting, then arrive (or fall back).
  go_door      — guide the user to a door (distance + hand cue), watch for obstacles.
  arrived      — terminal; the arrival line is spoken and the journey ends.

Priority is absolute: a confirmed indicator always beats a door.
"""
from __future__ import annotations

from collections import Counter

from .config import *  # noqa: F401,F403 — all tuning constants, referenced unqualified
from .goals import arrival_phrase, evaluate_arrival, resolve_goal
from .geometry import (_cluster_bearings, _cluster_doors, _direction, _signed_from_ref,
                       _track_turn, _turn_to)
from .state import (_enter_discover, _enter_face_target, _enter_go_door,
                    _enter_go_indicator, _scan_counts, _state)


# --- spoken phrasing -------------------------------------------------------------

def _steps_word(n: int) -> str:
    return "step" if n == 1 else "steps"


def _door_phrase(region: str, dist_m: float | None) -> str:
    """Spoken door call-out: bearing + step distance + a tactile hand cue.
    Within HAND_REACH_STEPS we ask the user to reach out now; farther away we tell
    them how many steps to walk before reaching out to feel for the door."""
    # Definite phrasing on purpose: by the time this speaks, the door has been
    # confirmed — "The door is", never "There's a door" (which sounds like a guess).
    if dist_m is not None and dist_m < 1.0:
        return ("The door is right in front of you. Reach out with your hand to find it."
                if region == "ahead"
                else f"The door is {region}, right next to you. Reach out with your hand to find it.")
    if dist_m is None:
        return f"The door is {region}."

    steps = max(1, round(dist_m / STEP_LENGTH_M))
    unit = _steps_word(steps)
    base = (f"The door is about {steps} {unit} ahead." if region == "ahead"
            else f"The door is {region}, about {steps} {unit} away.")

    if steps <= HAND_REACH_STEPS:
        return base + " You're close — reach out with your hand to find it."
    remaining = steps - HAND_REACH_STEPS
    return base + f" Walk forward, and after about {remaining} {_steps_word(remaining)} reach out with your hand."


def _find_door_phrases() -> list[str]:
    """Nudges while hunting for a door (rotated so re-prompts aren't identical)."""
    return [
        "I don't see a door yet. Keep scanning the room slowly.",
        "Still looking for a door. Keep moving the camera around the room.",
        "No door yet — keep scanning the walls slowly.",
    ]


# --- scan evidence + summary -----------------------------------------------------

def _confirmed_from_sectors() -> set:
    """Arrival evidence for the 360 scan, LOCALIZED: a real fridge racks up its
    sightings in one spot (a few adjacent sectors), while detector noise scatters
    around the room. A class is confirmed only when some 90-degree window (a sector
    plus its two neighbours) holds INDICATOR_HITS sightings — scattered one-off
    flickers can never add up to an arrival."""
    per_class: dict[str, dict[int, int]] = {}
    for bucket, counts in _state["sector_objs"].items():
        for c, n in counts.items():
            if c != "door":
                per_class.setdefault(c, {})[bucket] = n
    confirmed = set()
    for c, by_bucket in per_class.items():
        for b in by_bucket:
            window = (by_bucket.get((b - 1) % BUCKETS, 0) + by_bucket.get(b, 0)
                      + by_bucket.get((b + 1) % BUCKETS, 0))
            if window >= INDICATOR_HITS:
                confirmed.add(c)
                break
    return confirmed


def _scan_summary(goal: str) -> tuple[str, int]:
    """Spoken spatial map of what the 360 scan found, by direction. Each physical
    door is named once (clustered), and a repeated object in one direction once.
    Returns (text, item_count) so the caller can avoid re-announcing a sole finding."""
    items = []  # doors first, then indicators
    door_dirs = []
    for mean_signed, n in _cluster_doors():
        if n < DOOR_MIN_SIGHTINGS:
            continue  # a flicker, not a door
        where = _direction(mean_signed)
        if where not in door_dirs:
            door_dirs.append(where)
    # "A door nearby" is ONLY for genuinely compass-less scans (no bearings possible).
    # With a compass, doors either have a reliable direction or aren't mentioned.
    if (_state["ref_heading"] is None and not door_dirs
            and sum(c.get("door", 0) for c in _state["sector_objs"].values())
            >= DOOR_MIN_SIGHTINGS):
        items.append("a door nearby")
    items += [f"a door {w}" for w in door_dirs]

    # Object mentions: total sightings per (class, direction); below the minimum it's
    # detector noise and we keep quiet about it.
    dir_counts: dict = {}
    for bucket, counts in _state["sector_objs"].items():
        where = _direction(_signed_from_ref(bucket))
        for c, n in counts.items():
            if c != "door":
                dir_counts[(c, where)] = dir_counts.get((c, where), 0) + n
    items += [f"a {c} {where}" for (c, where), n in dir_counts.items()
              if n >= OBJ_MIN_SIGHTINGS]

    if not items:
        return f"I scanned the whole room but didn't find the {goal} or a door.", 0
    listing = items[0] if len(items) == 1 else ", ".join(items[:-1]) + f", and {items[-1]}"
    return f"Scan complete. I found {listing}.", len(items)


def _set_door_target(clusters: list[tuple[float, int]]) -> str:
    """Pick the most-sighted door (tie-break: nearest straight-ahead), make it the
    face_target, and return its spoken direction (relative to the start anchor)."""
    mean_signed, _n = max(clusters, key=lambda c: (c[1], -abs(c[0])))
    ref = _state["ref_heading"] or 0.0
    _state["target_heading"] = (ref + mean_signed) % 360.0
    _state["target_kind"] = "door"
    _enter_face_target()
    return _direction(mean_signed)


def _finish_discover(goal: str, indicator_ok: bool, confirm_start: bool = False) -> tuple:
    """End of the 360 scan: confirm the user is back at the start, speak the spatial
    summary, then pick the next phase. Returns (guidance, priority)."""
    summary, n_items = _scan_summary(goal)  # built before any _enter_* clears scan state
    if confirm_start:
        # Spoken ONLY when the compass verified the return to the start direction —
        # so every direction in the summary is true of where the user faces RIGHT NOW.
        summary = "You're back where you started — scan complete. " + summary.removeprefix("Scan complete. ")
    if indicator_ok:
        # Branch (a) — ALWAYS beats doors. The 360 scan itself accumulated the evidence
        # (INDICATOR_HITS frames of the goal's objects) — that IS the confirmation.
        # Declare arrival right here, with the directions, and the journey is done.
        _state["mode"] = "arrived"
        if n_items:
            return summary + f" You've reached the {goal}.", True
        lead = ("You're back where you started — scan complete. " if confirm_start
                else "Scan complete. ")
        return lead + f"You've reached the {goal}.", True

    # Branch (a-weak) — indicators were sighted but below the arrival bar. Same
    # directed confirmation we give doors: turn the user toward the sighting and
    # re-check there. Indicator priority holds: this runs BEFORE the door branch.
    ind_clusters = [c for c in _cluster_bearings(_state["ind_bearings"])
                    if c[1] >= OBJ_MIN_SIGHTINGS]
    if ind_clusters:
        mean_signed, _n = max(ind_clusters, key=lambda c: c[1])  # densest sighting area
        ref = _state["ref_heading"] or 0.0
        _state["target_heading"] = (ref + mean_signed) % 360.0
        _state["target_kind"] = "indicator"
        _enter_face_target()
        return (summary + f" That might be the {goal} — let's make sure. Turn toward "
                "it and point the camera there."), True

    # Branch (b) — doors: turn toward the chosen door, then re-confirm it before
    # guiding in. Flickers below DOOR_MIN_SIGHTINGS are noise, never a target.
    clusters = [c for c in _cluster_doors() if c[1] >= DOOR_MIN_SIGHTINGS]
    if clusters:
        where = _set_door_target(clusters)
        if n_items == 1:
            # The summary already named exactly this door — don't announce it twice.
            return (summary + " Turn toward it and point your camera at it, so I can "
                    "guide you in precisely."), True
        return (summary + f" Let's go to the door {where}. Turn that way and point your "
                "camera at it, so I can guide you in precisely."), True

    # No-compass fallback ONLY: doors were confirmed but bearings are impossible.
    # On a compass run, a door without a reliable direction cluster is a flicker —
    # fall through to the rescan instead of vaguely pointing at "the door".
    if (_state["ref_heading"] is None
            and sum(c.get("door", 0) for c in _state["sector_objs"].values()) >= DOOR_MIN_SIGHTINGS):
        _enter_go_door()
        return summary + " Point your camera at the door, and I'll guide you in.", True

    # Nothing useful found. ONE merged utterance (announcement + instruction) — two
    # back-to-back priority lines would cut each other off.
    _enter_discover()
    _state["skip_scan_prompt"] = True
    lead = "You're back where you started. " if confirm_start else ""
    if n_items == 0:
        body = "I couldn't find anything useful in this room. "
    else:  # something was sighted (e.g. a lone fridge glimpse) but no flag was earned
        body = summary.removeprefix("Scan complete. ").rstrip(".") + " — but nothing I can act on yet. "
    return (lead + body + "Let's scan one more time — slowly turn to your right, all "
            "the way around, until you are facing where you started."), True


# --- per-frame entry points the server calls ------------------------------------

def handle_goal(text: str) -> str:
    """Resolve the spoken goal; on a change, reset and start a fresh discovery scan."""
    goal = resolve_goal(text) or "kitchen"
    if goal != _state["goal"]:
        _state["goal"] = goal
        _state["near_latch"] = False
        _state["gone"] = 0
        _enter_discover()
    return goal


def accumulate_indicator_evidence(seen: set, indicators: set) -> set:
    """Update the per-room indicator counts and return the currently-confirmed set.
    During the compass 360 the evidence must be LOCALIZED (one 90-deg window), so
    scattered false hits can't sum to an arrival."""
    _scan_counts.update(seen & indicators)
    if _state["mode"] == "discover" and _state["ref_heading"] is not None:
        return _confirmed_from_sectors()
    return {c for c, n in _scan_counts.items() if n >= INDICATOR_HITS}


def detect_transit(near_box: bool, door_confirmed: bool, cur_frac: float) -> tuple:
    """Decide whether the user just walked through a doorway (only meaningful in
    go_door). Mutates the near-latch state and, on a confirmed transit, resets to a
    fresh discover scan. Returns (transit, just_near)."""
    transit = False
    just_near = False  # near_latch turned on THIS frame -> announce "you're at the door"
    if _state["mode"] == "go_door" and door_confirmed:
        # Track how big the confirmed door got during this approach (scale-free).
        _state["approach_frac"] = max(_state["approach_frac"], cur_frac)
    # The saturated close-range box only counts if a tracked approach already got us
    # near this door — a wall pointed at mid-walk was never a confirmed approach.
    # Two near signals: metric distance, OR the confirmed door having grown to fill
    # half the frame (immune to distance-calibration changes).
    at_door_box = near_box and (
        (_state["last_door_dist"] is not None and _state["last_door_dist"] <= NEAR_DOOR_M)
        or _state["approach_frac"] >= APPROACH_FRAC)
    if _state["mode"] != "go_door":
        _state["near_latch"] = False
        _state["gone"] = 0
        _state["near_age"] = 0
    elif (door_confirmed and cur_frac >= DOOR_FILL_FRAC) or at_door_box:
        just_near = not _state["near_latch"]
        _state["near_latch"] = True
        _state["gone"] = 0
        # New rooms also throw saturated door candidates, which would hold this latch
        # forever AFTER the user walked through. Only a properly confirmed door resets
        # the hold timer; saturated-box frames age it until we infer the transit.
        _state["near_age"] = 0 if door_confirmed else _state["near_age"] + 1
        if _state["near_age"] >= NEAR_HOLD_MAX:
            transit = True
            _state["near_latch"] = False
            _state["gone"] = 0
            _state["near_age"] = 0
            _enter_discover()
    elif _state["near_latch"] and not door_confirmed:
        _state["gone"] += 1
        _state["near_age"] += 1
        if _state["gone"] >= TRANSIT_GONE or _state["near_age"] >= NEAR_HOLD_MAX:
            transit = True
            _state["near_latch"] = False
            _state["gone"] = 0
            _state["near_age"] = 0
            _enter_discover()
    elif door_confirmed:
        _state["gone"] = 0  # door still in view but not close — not a transit
        _state["near_age"] = 0
    return transit, just_near


def step(*, goal, heading, motion, w, seen, indicators, obj_dets, door_confirmed,
         door_cx_frac, region, door_dist, transit, just_near, confirmed,
         obst_guidance, obst_priority, obst_blocking) -> tuple:
    """Run the state machine for one frame. Returns
    (guidance, priority, announce_arrival, phrase, matched)."""
    result = evaluate_arrival(goal, confirmed)
    matched = result["matched_primary"] + result["matched_secondary"]
    indicator_ok = result["arrived"]  # >=1 primary or >=2 secondary, accumulated

    # `priority` lines (sector stops, phase changes, arrival, summary) may interrupt;
    # ambient nudges (hold-steady, keep-turning, door countdown) never interrupt.
    guidance = ""
    priority = False
    announce_arrival = False

    if transit:
        # Walked through a doorway -> Pass 1 for the new room (discover set by transit).
        if heading is not None:
            _state["ref_heading"] = heading
            _state["last_heading"] = heading
        _state["skip_scan_prompt"] = True  # instruction is in THIS line; don't repeat it
        guidance = ("You've gone through the doorway. Take two or three steps into the "
                    "room, then slowly turn to your right, all the way around, back to "
                    "where you started, so I can scan this room.")
        priority = True

    elif _state["mode"] == "discover":
        if heading is None:
            # Fallback (no compass, e.g. a laptop): one slow steady-capture pass.
            if _state["last_heading"] is None:
                _state["last_heading"] = 0.0  # mark started
                if _state["skip_scan_prompt"]:
                    _state["skip_scan_prompt"] = False  # instruction already in the rescan line
                else:
                    guidance = (f"Looking for the {goal}. Let's scan the room — slowly "
                                "pan all the way around, pausing a moment as you go.")
                    priority = True
            elif motion <= STILL_MAX:
                _state["scan_age"] += 1
                objs = _state["sector_objs"].setdefault(0, Counter())
                objs.update(seen & indicators)
                if door_confirmed:
                    objs["door"] += 1
                    _state["door_seen"] = True
                if _state["scan_age"] >= FALLBACK_FRAMES:
                    guidance, priority = _finish_discover(goal, indicator_ok)
        elif _state["ref_heading"] is None:
            # First sensor reading -> set the START direction (our anchor) and begin.
            _state["ref_heading"] = heading
            _state["last_heading"] = heading
            if _state["skip_scan_prompt"]:
                _state["skip_scan_prompt"] = False  # instruction already spoken with the rescan line
            else:
                guidance = (f"Looking for the {goal}. Let's scan the room — slowly turn "
                            "to your right, all the way around, until you are facing "
                            "where you started.")
                priority = True
        else:
            _track_turn(heading)  # advance the full-circle total (non-blurred frame)
            rel = (heading - _state["ref_heading"]) % 360.0
            bucket = int(rel // BUCKET_DEG) % BUCKETS
            _state["covered"].add(bucket)
            objs = _state["sector_objs"].setdefault(bucket, Counter())
            objs.update(seen & indicators)  # sighting COUNTS -> reliability gating later
            # Record TRUE bearings (camera heading + offset within the frame) for
            # everything that matters: doors AND goal indicators.
            for icls, _icf, ixy in obj_dets:
                if icls in indicators:
                    icx = ((ixy[0] + ixy[2]) / 2) / w
                    _state["ind_bearings"].append(
                        (heading + (icx - 0.5) * ASSUMED_HFOV_DEG) % 360.0)
            if door_confirmed:
                _state["door_seen"] = True
                # Count the sighting and its bearing from the SAME frames (ones with an
                # actual box) so the reliability threshold and the direction clusters
                # always agree — a count without a bearing sent us down the
                # no-compass "door nearby" path by mistake.
                if door_cx_frac is not None:
                    objs["door"] += 1
                    bearing = (heading + (door_cx_frac - 0.5) * ASSUMED_HFOV_DEG) % 360.0
                    _state["door_bearings"].append(bearing)
            # Progress context by how far around they've turned (ambient, once each).
            prog = abs(_state["net_rotation"])
            ms = _state["milestones"]
            if prog >= 270 and "75" not in ms:
                ms.add("75"); guidance = "Almost back to where you started, keep turning."
            elif prog >= 180 and "50" not in ms:
                ms.add("50"); guidance = "Halfway around, keep turning."
            elif prog >= 90 and "25" not in ms:
                ms.add("25"); guidance = "Good, keep turning to your right."
            # Done ONLY when a full circle of rotation has accumulated AND the compass
            # confirms they're facing the start direction again. Never on bucket
            # coverage alone — compass noise can fake that early, ending the scan
            # mid-turn with directions computed from a broken premise.
            if abs(_state["net_rotation"]) >= FULL_TURN_DEG:
                if abs(_turn_to(_state["ref_heading"], heading)) <= START_TOL:
                    guidance, priority = _finish_discover(goal, indicator_ok, confirm_start=True)
                else:
                    _state["phase_age"] += 1
                    if "back" not in ms or _state["phase_age"] >= REPROMPT:
                        ms.add("back")
                        _state["phase_age"] = 0
                        guidance = "Almost done — keep turning until you face where you started."

    elif _state["mode"] == "go_indicator":
        # Pass 2a: re-confirm the goal's objects, then arrive. (Doors ignored here.)
        if indicator_ok:
            announce_arrival = True
        else:
            _state["scan_age"] += 1
            if _state["phase_age"] >= REPROMPT:
                guidance = f"Keep the camera there, panning slowly, while I confirm the {goal}."  # ambient
                _state["phase_age"] = 0
            _state["phase_age"] += 1
            if _state["scan_age"] >= ROOM_SCAN_CYCLES:
                # Couldn't re-confirm -> false alarm. The indicator had its chance;
                # if the scan also flagged a door, take it (door bearings survive the
                # confirm phase) — otherwise a full rescan.
                door_clusters = [c for c in _cluster_doors() if c[1] >= DOOR_MIN_SIGHTINGS]
                if door_clusters and heading is not None:
                    _set_door_target(door_clusters)
                    guidance = (f"I couldn't confirm the {goal} here. Let's take the "
                                "door instead — I'll help you turn to face it.")
                else:
                    _enter_discover()
                    _state["skip_scan_prompt"] = True
                    guidance = (f"I couldn't confirm the {goal}. Let's scan the room again — "
                                "slowly turn to your right, all the way around, until you "
                                "are facing where you started.")
                priority = True

    elif _state["mode"] == "face_target":
        # Phase 2 opener (both branches): walk the user through turning until they
        # face the flagged target, THEN run the focused confirmation scan there.
        kind = _state["target_kind"] or "door"
        label = "door" if kind == "door" else goal
        h = heading
        if h is None or _state["target_heading"] is None:
            # No compass -> skip the guided turn, go straight to the confirm phase.
            if kind == "door":
                _enter_go_door()
                guidance = "Turn toward the door, and I'll guide you in."
            else:
                _enter_go_indicator()
                guidance = f"Point the camera where you saw the {goal}, and hold it there."
            priority = True
        else:
            turn = _turn_to(_state["target_heading"], h)
            if abs(turn) <= FACE_TOL:
                if kind == "door":
                    _enter_go_door()
                    guidance = "The door should be right ahead of you now. Let's go to it."
                else:
                    _enter_go_indicator()
                    guidance = (f"You should be facing the {goal} now. Hold the camera "
                                "there and pan slowly while I confirm.")
                priority = True
            else:
                side = "right" if turn > 0 else "left"
                guidance = f"Turn slowly to your {side} to face the {label}."  # ambient, re-evaluated live

    elif _state["mode"] == "arrived":
        pass  # journey complete — arrival is reported via the arrived/phrase fields below

    else:  # go_door — Pass 2b: door-only concern.
        if obst_blocking or obst_guidance:
            # Safety first: an obstacle in the walking lane overrides door guidance
            # (and a "path is clear" line gets spoken before door directions resume).
            guidance, priority = obst_guidance, obst_priority
        elif just_near:
            guidance = "You're right at the door. Reach out, open it, and walk through."
            priority = True
        elif _state["near_latch"] and not door_confirmed:
            # Standing at the door (detector saturated by the panel). Stay quiet —
            # no "no door yet" nudges, no lost-door timer; the transit check is watching.
            _state["scan_age"] = 0
        elif door_confirmed and region:
            _state["scan_age"] = 0  # door in sight -> confirmation holds
            guidance = _door_phrase(region, door_dist)  # ambient (updates as you approach)
        else:
            # Confirmation scan failing: like branch (a), a flag that can't be
            # re-confirmed within a window means a full rescan, not endless nudging.
            _state["scan_age"] += 1
            if _state["scan_age"] >= DOOR_LOST_CYCLES:
                _enter_discover()
                _state["skip_scan_prompt"] = True
                guidance = ("I can't find that door anymore. Let's scan the room again — "
                            "slowly turn to your right, all the way around, until you "
                            "are facing where you started.")
                priority = True
            else:
                if _state["phase_age"] == 0:
                    _state["phase"] += 1
                    guidance = _find_door_phrases()[_state["phase"] % 3]  # ambient
                _state["phase_age"] += 1
                if _state["phase_age"] >= REPROMPT:
                    _state["phase_age"] = 0

    phrase = ""
    if _state["mode"] == "arrived":
        announce_arrival = True
        # On the frame the scan just finished, guidance holds the composed
        # summary + "You've reached the kitchen." line — that IS the arrival phrase.
        phrase = guidance or arrival_phrase(goal, result)
        guidance = ""
    elif announce_arrival:
        phrase = arrival_phrase(goal, result)

    return guidance, priority, announce_arrival, phrase, matched
