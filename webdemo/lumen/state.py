"""All mutable controller state, in one place.

`_state` is the single source of truth for the exploration FSM; `_scan_counts` and
`_door_hist` are the per-room evidence buffers; `prev_small` is the last downscaled
frame for motion gating. The `enter_*` helpers are the only sanctioned way to switch
phase — they reset the fields that phase depends on.

Other modules do `from .state import _state, _scan_counts, _door_hist` and mutate them
in place. `prev_small` is reassigned each frame, so access it as `state.prev_small`.
"""
from __future__ import annotations

from collections import Counter, deque

from .config import WINDOW

_door_hist: deque = deque(maxlen=WINDOW)  # (region, distance) of strongest door, per frame
_scan_counts: Counter = Counter()         # indicator-class hit counts for the CURRENT room
prev_small = None                         # last downscaled grayscale frame, for motion gating

# Two-pass, single-concern exploration so prompts never override each other:
#   discover     -> Pass 1: ONE instruction, then scan SILENTLY for a full sweep,
#                   accumulating two flags: indicator_ok and door_seen.
#   go_indicator -> Pass 2a (chosen when indicator_ok): re-confirm the goal's
#                   objects, then announce arrival. Beats doors.
#   go_door      -> Pass 2b (chosen when only door_seen): locate + guide to a door.
# Walking through a near door resets us to discover for the new room.
#   mode       : current phase
#   scan_age   : usable (non-blurred) frames spent in the current scan
#   phase_age  : paces spoken re-prompts (detection runs ~3x/s)
#   near_latch : we got right up to a door (it filled the view)
#   gone       : cycles with no door since near_latch -> infer we walked through
#   door_seen  : a door was confirmed at some point during the current discover
#   ref_heading: compass heading (deg) the user faced when this scan began = "ahead"
#   last_heading / net_rotation: track cumulative turn to know when 360 is done
#   covered    : set of 30-deg sector indices the turn has passed through
#   milestones : which progress lines (25/50/75%) were already spoken
#   sector_objs: {sector -> Counter of detected classes (incl 'door')} for the summary
#   door_bearings: absolute compass headings (deg) where a door was seen, each
#                  corrected by the door's horizontal position in the frame. Clustered
#                  at scan-end so ONE physical door = one target (not one per sector).
_state = {"goal": None, "mode": "discover", "scan_age": 0,
          "phase": 0, "phase_age": 0, "near_latch": False, "gone": 0,
          "door_seen": False,
          "ref_heading": None, "last_heading": None, "net_rotation": 0.0,
          "covered": set(), "sector_objs": {}, "milestones": set(),
          "door_bearings": [], "ind_bearings": [],
          "target_heading": None, "target_kind": None,
          "skip_scan_prompt": False, "last_door_dist": None,
          "approach_frac": 0.0, "near_age": 0,
          "obst_hits": 0, "obst_clear": 0, "obst_cool": 0, "obst_active": False,
          "fast_frames": 0, "door_announced": False, "confirm_settle": 0,
          "obst_hold": 0, "path_checked": True, "walk_frames": 0, "near_streak": 0}


def _reset_scan_fields() -> None:
    _state["scan_age"] = 0
    _state["phase"] = 0
    _state["phase_age"] = 0


def _enter_discover() -> None:
    """Pass 1: one guided 360 turn of a (new) room, collecting flags."""
    _state["mode"] = "discover"
    _state["door_seen"] = False
    _state["ref_heading"] = None
    _state["last_heading"] = None
    _state["net_rotation"] = 0.0
    _state["covered"] = set()
    _state["sector_objs"] = {}
    _state["milestones"] = set()
    _state["door_bearings"] = []
    _state["ind_bearings"] = []
    _state["target_kind"] = None
    _state["skip_scan_prompt"] = False
    _state["last_door_dist"] = None
    _reset_scan_fields()
    _scan_counts.clear()  # fresh room: don't carry indicator evidence across
    _door_hist.clear()


def _enter_go_indicator() -> None:
    """Pass 2a: re-confirm the goal's indicators with a fresh scan, then arrive."""
    _state["mode"] = "go_indicator"
    _reset_scan_fields()
    _state["confirm_settle"] = 0  # settle window before announcing (see CONFIRM_SETTLE)
    _scan_counts.clear()  # fresh evidence so the confirm scan is a real re-check
    _door_hist.clear()


def _enter_face_target() -> None:
    """Pass 2b-pre: actively walk the user through turning to face the chosen door."""
    _state["mode"] = "face_target"
    _reset_scan_fields()


def _enter_go_door() -> None:
    """Pass 2b: locate and guide the user to a door (door-only concern)."""
    _state["mode"] = "go_door"
    _reset_scan_fields()
    _state["phase_age"] = 1  # delay the first "no door yet" so it doesn't double up
    _state["last_door_dist"] = None  # fresh approach: no stale "we were close" memory
    _state["approach_frac"] = 0.0
    _state["near_age"] = 0
    _state["obst_hits"] = 0      # fresh approach -> fresh obstacle debounce
    _state["obst_clear"] = 0
    _state["obst_cool"] = 0
    _state["obst_active"] = False
    _state["door_announced"] = False  # the ONE static door call-out for this approach
    _state["obst_hold"] = 0           # obstacle-speech holdoff while the call-out plays
    _state["path_checked"] = True     # armed (set False) by the call-out itself
    _state["walk_frames"] = 0         # camera-motion frames = evidence the user MOVED
    _state["near_streak"] = 0         # consecutive at-door frames (spike immunity)
    _state["near_latch"] = False      # fresh approach = fresh latch
    _state["gone"] = 0
    _state["near_age"] = 0
