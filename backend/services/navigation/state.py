"""All mutable navigation state, one instance per Session.

The webdemo prototype kept this as module-level globals (single user). Integrated
into the backend, every connected user gets their own :class:`NavState`, created by
the navigation engine when the FSM enters ``NavigationActive`` and thrown away when
the task ends. The other navigation modules receive it as their first ``st``
argument and index it like the original dict (``st["mode"]``), so their decision
logic is unchanged from the field-tested prototype.

Contents:
  - the phase dict (``st[...]``) — the exploration FSM's single source of truth
  - ``scan_counts`` — indicator-class hit counts for the CURRENT room
  - ``door_hist``  — rolling (region, distance) of the strongest door, per frame
  - ``prev_small`` — last downscaled grayscale frame, for motion gating

The ``enter_*`` methods are the only sanctioned way to switch phase — they reset
the fields that phase depends on.

Two-pass, single-concern exploration so prompts never override each other:
  discover     -> Pass 1: ONE instruction, then scan SILENTLY for a full sweep,
                  accumulating two flags: indicator_ok and door_seen.
  go_indicator -> Pass 2a (chosen when indicator_ok): re-confirm the goal's
                  objects, then announce arrival. Beats doors.
  face_target  -> Pass 2 opener: rotate the user to face the chosen door/indicator.
  go_door      -> Pass 2b (chosen when only door_seen): locate + guide to a door.
  arrived      -> terminal; the arrival line is spoken and the journey ends.
Walking through a near door resets us to discover for the new room.

Key fields (see the prototype's docs for the full story):
  mode       : current phase
  scan_age   : usable (non-blurred) frames spent in the current scan
  phase_age  : paces spoken re-prompts (seconds of observed-frame time)
  near_latch : we got right up to a door (it filled the view)
  gone       : seconds (of frames) with no door since near_latch -> walked through
  rooms_visited: doorway transits this JOURNEY (never reset per room) - drives the
                 ROOM_CAP check-in so exploration can't loop silently forever
  door_seen  : a door was confirmed at some point during the current discover
  ref_heading: compass heading (deg) the user faced when this scan began = "ahead"
  last_heading / net_rotation: track cumulative turn to know when 360 is done
  covered    : set of 30-deg sector indices the turn has passed through
  milestones : which progress lines were already spoken
  sector_objs: {sector -> Counter of detected classes (incl 'door')} for the summary
  door_bearings: absolute compass headings (deg) where a door was seen, each
                 corrected by the door's horizontal position in the frame. Clustered
                 at scan-end so ONE physical door = one target (not one per sector).
"""
from __future__ import annotations

from collections import Counter, deque
from typing import Any, Optional

import numpy as np

from .config import WINDOW


def _initial_fields() -> dict[str, Any]:
    return {"goal": None, "rooms_visited": 0, "mode": "discover", "scan_age": 0,
            "phase": 0, "phase_age": 0, "near_latch": False, "gone": 0,
            "door_seen": False,
            "ref_heading": None, "last_heading": None, "net_rotation": 0.0,
            "covered": set(), "sector_objs": {}, "milestones": set(),
            "door_bearings": [], "ind_bearings": [],
            "target_heading": None, "target_kind": None,
            "skip_scan_prompt": False, "last_door_dist": None,
            "approach_frac": 0.0, "near_age": 0, "close_gone": 0.0,
            "obst_hits": 0, "obst_clear": 0, "obst_cool": 0, "obst_active": False,
            "fast_frames": 0, "door_announced": False, "reorient": False,
            "confirm_settle": 0,
            "obst_hold": 0, "path_checked": True, "walk_frames": 0, "near_streak": 0}


class NavState:
    """Per-session navigation state. Dict-style access (``st["mode"]``) keeps the
    ported decision code identical to the field-tested prototype."""

    def __init__(self) -> None:
        self._d: dict[str, Any] = _initial_fields()
        self.door_hist: deque = deque(maxlen=WINDOW)
        self.scan_counts: Counter = Counter()
        self.prev_small: Optional[np.ndarray] = None

    # -- dict-style access ---------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        return self._d[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._d[key] = value

    # -- phase transitions ---------------------------------------------------

    def _reset_scan_fields(self) -> None:
        self._d["scan_age"] = 0
        self._d["phase"] = 0
        self._d["phase_age"] = 0

    def enter_discover(self) -> None:
        """Pass 1: one guided 360 turn of a (new) room, collecting flags."""
        d = self._d
        d["mode"] = "discover"
        d["door_seen"] = False
        d["ref_heading"] = None
        d["last_heading"] = None
        d["net_rotation"] = 0.0
        d["covered"] = set()
        d["sector_objs"] = {}
        d["milestones"] = set()
        d["door_bearings"] = []
        d["ind_bearings"] = []
        d["target_kind"] = None
        d["skip_scan_prompt"] = False
        d["last_door_dist"] = None
        self._reset_scan_fields()
        self.scan_counts.clear()  # fresh room: don't carry indicator evidence across
        self.door_hist.clear()

    def enter_go_indicator(self) -> None:
        """Pass 2a: re-confirm the goal's indicators with a fresh scan, then arrive."""
        self._d["mode"] = "go_indicator"
        self._reset_scan_fields()
        self._d["confirm_settle"] = 0  # settle window before announcing (CONFIRM_SETTLE_SEC)
        self.scan_counts.clear()  # fresh evidence so the confirm scan is a real re-check
        self.door_hist.clear()

    def enter_face_target(self) -> None:
        """Pass 2b-pre: actively walk the user through turning to face the chosen door."""
        self._d["mode"] = "face_target"
        self._reset_scan_fields()

    def enter_go_door(self) -> None:
        """Pass 2b: locate and guide the user to a door (door-only concern)."""
        d = self._d
        d["mode"] = "go_door"
        self._reset_scan_fields()
        d["phase_age"] = 0.5  # (seconds) delay the first "no door yet" so it doesn't double up
        d["last_door_dist"] = None  # fresh approach: no stale "we were close" memory
        d["approach_frac"] = 0.0
        d["near_age"] = 0
        d["obst_hits"] = 0      # fresh approach -> fresh obstacle debounce
        d["obst_clear"] = 0
        d["obst_cool"] = 0
        d["obst_active"] = False
        d["door_announced"] = False  # the ONE static door call-out for this approach
        d["reorient"] = False        # pending post-obstacle re-orientation
        d["obst_hold"] = 0           # obstacle-speech holdoff while the call-out plays
        d["path_checked"] = True     # armed (set False) by the call-out itself
        d["walk_frames"] = 0         # camera-motion frames = evidence the user MOVED
        d["near_streak"] = 0         # consecutive at-door frames (spike immunity)
        d["near_latch"] = False      # fresh approach = fresh latch
        d["gone"] = 0
        d["close_gone"] = 0.0        # tracked-close-then-vanished at-door signal
