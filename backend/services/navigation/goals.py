"""Goal → room-indicator mapping (the destination knowledge).

Answers *"what objects tell me I've arrived in this room?"*. The controller uses it for
the **arrival test**: a room is the goal once the objects detected there match the
goal's indicator set.

- Hardcoded tables, lowercased keys, no I/O at import.
- A goal name is resolved loosely (aliases + leading article/filler strip).
- Indicators are split into PRIMARY (strong, near-unique to the room) and SECONDARY
  (supporting; often shared across rooms). Arrival requires **>=1 primary OR
  >=2 secondary**, so a single shared object (e.g. a lone ``sink``, which appears in
  both kitchens and bathrooms) never declares a room on its own.

Every indicator is a COCO-80 class, so the YOLOv8m detector produces them with no model
change. (``door`` is the only out-of-COCO class and is intentionally not a room indicator.)

The tables are deliberately simple and tunable — adjust membership as real-world
testing shows false arrivals or misses.
"""

from __future__ import annotations

import re
from typing import Final, Iterable, Optional


# ---------------------------------------------------------------------------
# Goal-name aliases → canonical goal key
# ---------------------------------------------------------------------------

GOAL_ALIASES: Final[dict[str, str]] = {
    "kitchen": "kitchen",
    "kitchenette": "kitchen",

    "bathroom": "bathroom",
    "restroom": "bathroom",
    "washroom": "bathroom",
    "toilet": "bathroom",
    "wc": "bathroom",
    "lavatory": "bathroom",

    "bedroom": "bedroom",
    "bed room": "bedroom",

    "living room": "living room",
    "lounge": "living room",
    "sitting room": "living room",
    "family room": "living room",
    "den": "living room",

    "office": "office",
    "study": "office",
    "study room": "office",

    "dining room": "dining room",
    "dining": "dining room",
    "dining area": "dining room",
}


# ---------------------------------------------------------------------------
# Canonical goal → indicator objects (all COCO-80 class names)
# ---------------------------------------------------------------------------

GOAL_INDICATORS: Final[dict[str, dict[str, list[str]]]] = {
    "kitchen": {
        "primary": ["refrigerator", "oven", "microwave", "toaster"],
        "secondary": ["sink", "dining table"],
    },
    "bathroom": {
        "primary": ["toilet"],
        "secondary": ["sink"],
    },
    "bedroom": {
        "primary": ["bed"],
        "secondary": [],
    },
    "living room": {
        "primary": ["couch", "tv"],
        "secondary": ["potted plant", "remote"],
    },
    "office": {
        "primary": ["laptop", "keyboard"],
        "secondary": ["mouse", "tv"],
    },
    "dining room": {
        "primary": ["dining table"],
        "secondary": ["chair"],
    },
}


# ---------------------------------------------------------------------------
# Spoken-name helpers (a few COCO names aren't natural out loud)
# ---------------------------------------------------------------------------

_DISPLAY: Final[dict[str, str]] = {
    "refrigerator": "fridge",
    "tv": "TV",
    "potted plant": "plant",
    "dining table": "table",
}


def _display(cls: str) -> str:
    return _DISPLAY.get(cls, cls)


def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def _render_list(classes: list[str]) -> str:
    """['refrigerator', 'oven'] -> 'a fridge and an oven'."""
    parts = [f"{_article(_display(c))} {_display(c)}" for c in classes]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f" and {parts[-1]}"


_NORM_STRIP_RE = re.compile(
    r"^(the|a|an|to|go to|take me to|get me to|navigate to|bring me to)\b\s*"
)


def _norm(text: str) -> str:
    t = re.sub(r"[^\w\s]", " ", (text or "").lower())
    t = re.sub(r"\s+", " ", t).strip()
    # Strip a leading article / navigation filler ("take me to the kitchen").
    prev = None
    while prev != t:
        prev = t
        t = _NORM_STRIP_RE.sub("", t).strip()
    return t


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_goal(text: str) -> Optional[str]:
    """Resolve a spoken destination to a canonical goal key, or None.

    Exact alias match first, then substring (longest alias wins) so phrases like
    "take me to the master bedroom" still resolve to "bedroom".
    """
    key = _norm(text)
    if not key:
        return None
    if key in GOAL_ALIASES:
        return GOAL_ALIASES[key]
    for alias in sorted(GOAL_ALIASES, key=len, reverse=True):
        if alias in key:
            return GOAL_ALIASES[alias]
    return None


def is_known_goal(text: str) -> bool:
    """True if resolve_goal() would return a goal we have indicators for."""
    return resolve_goal(text) is not None


def indicators_for(goal: str) -> tuple[list[str], list[str]]:
    """Return (primary, secondary) indicator class lists for a goal."""
    canon = resolve_goal(goal) or goal
    spec = GOAL_INDICATORS.get(canon)
    if not spec:
        return [], []
    return list(spec.get("primary", [])), list(spec.get("secondary", []))


def evaluate_arrival(goal: str, confirmed_classes: Iterable[str]) -> dict:
    """Decide whether the current room is the goal.

    `confirmed_classes` is the set of object classes the detector has confirmed in
    this room (temporal consistency is the caller's job). Arrival rule: >=1 primary
    OR >=2 secondary indicators.
    """
    primary, secondary = indicators_for(goal)
    confirmed = set(confirmed_classes)
    matched_primary = [c for c in primary if c in confirmed]
    matched_secondary = [c for c in secondary if c in confirmed]
    arrived = len(matched_primary) >= 1 or len(matched_secondary) >= 2
    return {
        "arrived": arrived,
        "matched_primary": matched_primary,
        "matched_secondary": matched_secondary,
    }


def arrival_phrase(goal: str, result: dict) -> str:
    """Build the spoken arrival line from an evaluate_arrival() result."""
    canon = resolve_goal(goal) or goal
    seen = list(result.get("matched_primary", [])) + list(result.get("matched_secondary", []))
    if not seen:
        return f"We've reached the {canon}."
    return f"I can see {_render_list(seen)} — we've reached the {canon}."
