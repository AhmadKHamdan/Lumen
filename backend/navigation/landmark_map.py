"""Landmark alias → COCO class mapping (Sprint 4 Part 2).

Decisions implemented here:
- A1: Hardcoded Python dict in this module.
- A2: One alias maps to a LIST of COCO classes (e.g. "seat" -> [chair, couch, bench]).
- A3: Resolution order is exact -> substring -> fuzzy (rapidfuzz, threshold=80).
- A4: When nothing matches, resolve() returns []. Caller re-prompts the user.
- A5: Coverage = furniture + appliances + common indoor objects (decision C).
- A6: Architectural terms ("doorway", "hallway", etc.) get best-effort mapping
      where possible (e.g. "doorway" -> ["door"]). Some terms map to [] today
      and become detectable only after a custom-indoor fine-tune in Sprint 5.
- A7: Keys stored lowercased; lookup also lowercases first.
- A8: This module's public API is resolve(normalized_text) -> list[str].

The COCO 80-class label set (from the Ultralytics YOLOv8 default weights) is
the universe of valid target class names. We keep the table within that
universe; aliases that have no COCO equivalent map to [].
"""

from __future__ import annotations

import re
from typing import Final

# rapidfuzz is the fuzzy-match library named in A3. It's a runtime dependency
# of this module; if it isn't installed we fall back to a pure-Python
# similarity that's good enough for the small alias table we have today.
try:
    from rapidfuzz import process as _rf_process
    from rapidfuzz import fuzz as _rf_fuzz
    _HAS_RAPIDFUZZ = True
except ImportError:  # pragma: no cover - tested via integration only
    _HAS_RAPIDFUZZ = False


# ---------------------------------------------------------------------------
# COCO class names (the 80 classes YOLOv8n was trained on)
# ---------------------------------------------------------------------------

COCO_CLASSES: Final[frozenset[str]] = frozenset({
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
})


# ---------------------------------------------------------------------------
# The alias table (A5 = C: furniture + appliances + common indoor objects)
#
# All keys are lowercased and free of leading/trailing whitespace. Values are
# lists of COCO class names; the parser may pick any of them at runtime.
# ---------------------------------------------------------------------------

LANDMARK_ALIASES: Final[dict[str, list[str]]] = {
    # ---- Architectural (A6 = E: best-effort now, may improve in Sprint 5) ----
    "door": ["door"],            # not in vanilla COCO; reserved for fine-tune
    "doorway": ["door"],
    "entrance": ["door"],
    "exit": ["door"],
    "gateway": ["door"],
    "window": [],                # no COCO class; gap acknowledged
    "wall": [],
    "stairs": [],
    "staircase": [],
    "steps": [],
    "hallway": [],
    "corridor": [],
    "hall": [],

    # ---- Seating ----
    "chair": ["chair"],
    "seat": ["chair", "couch", "bench"],
    "armchair": ["chair"],
    "couch": ["couch"],
    "sofa": ["couch"],
    "settee": ["couch"],
    "loveseat": ["couch"],
    "bench": ["bench"],

    # ---- Tables / surfaces ----
    "table": ["dining table"],
    "dining table": ["dining table"],
    "desk": ["dining table"],
    "kitchen table": ["dining table"],
    "coffee table": ["dining table"],

    # ---- Bedroom ----
    "bed": ["bed"],
    "mattress": ["bed"],

    # ---- Electronics / displays ----
    "tv": ["tv"],
    "television": ["tv"],
    "screen": ["tv", "laptop"],
    "monitor": ["tv", "laptop"],
    "laptop": ["laptop"],
    "computer": ["laptop"],
    "keyboard": ["keyboard"],
    "mouse": ["mouse"],
    "remote": ["remote"],
    "phone": ["cell phone"],
    "cell phone": ["cell phone"],
    "mobile": ["cell phone"],

    # ---- Kitchen appliances ----
    "refrigerator": ["refrigerator"],
    "fridge": ["refrigerator"],
    "freezer": ["refrigerator"],
    "microwave": ["microwave"],
    "oven": ["oven"],
    "stove": ["oven"],
    "toaster": ["toaster"],
    "sink": ["sink"],
    "kitchen sink": ["sink"],

    # ---- Bathroom ----
    "toilet": ["toilet"],
    "bathroom sink": ["sink"],

    # ---- Common small objects ----
    "bottle": ["bottle"],
    "water bottle": ["bottle"],
    "cup": ["cup"],
    "mug": ["cup"],
    "glass": ["wine glass", "cup"],
    "wine glass": ["wine glass"],
    "bowl": ["bowl"],
    "container": ["bottle", "cup", "bowl"],

    # ---- Reading / objects ----
    "book": ["book"],
    "books": ["book"],
    "clock": ["clock"],
    "vase": ["vase"],
    "plant": ["potted plant"],
    "potted plant": ["potted plant"],
    "flowers": ["vase", "potted plant"],

    # ---- Bags / personal ----
    "backpack": ["backpack"],
    "bag": ["backpack", "handbag", "suitcase"],
    "handbag": ["handbag"],
    "purse": ["handbag"],
    "suitcase": ["suitcase"],
    "luggage": ["suitcase"],

    # ---- People (used by Part 3 obstacle warnings as well) ----
    "person": ["person"],
    "people": ["person"],
    "someone": ["person"],

    # ---- Pets ----
    "cat": ["cat"],
    "dog": ["dog"],
    "pet": ["cat", "dog"],
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")
FUZZY_THRESHOLD: Final[int] = 80  # A3: rapidfuzz score 0..100.


def _norm_key(text: str) -> str:
    """Internal normalizer: lowercase, collapse whitespace, strip ends."""
    return _WHITESPACE_RE.sub(" ", (text or "").lower()).strip()


def _exact_lookup(key: str) -> list[str] | None:
    return LANDMARK_ALIASES.get(key)


def _substring_lookup(key: str) -> list[str] | None:
    """Return the first alias entry whose key is contained in `key`.

    Order matters: longer aliases first to avoid 'chair' matching before
    'armchair'. We sort by length once on import.
    """
    for alias in _SUBSTRING_ORDER:
        if alias and alias in key:
            return LANDMARK_ALIASES[alias]
    return None


_SUBSTRING_ORDER: list[str] = sorted(LANDMARK_ALIASES.keys(), key=len, reverse=True)


def _fuzzy_lookup(key: str) -> list[str] | None:
    """rapidfuzz best-match against alias keys; threshold = FUZZY_THRESHOLD."""
    if not _HAS_RAPIDFUZZ or not key:
        return None
    match = _rf_process.extractOne(
        key, LANDMARK_ALIASES.keys(), scorer=_rf_fuzz.WRatio
    )
    if match is None:
        return None
    best, score, _ = match
    if score >= FUZZY_THRESHOLD:
        return LANDMARK_ALIASES[best]
    return None


def resolve(normalized_text: str) -> list[str]:
    """Resolve a normalized waypoint phrase to a list of COCO class names.

    Returns [] when no alias matches at any level. Callers (the manager)
    are expected to treat [] as "unmappable" and either re-prompt the user
    or warn-and-continue depending on context (decisions A4, B4).
    """
    key = _norm_key(normalized_text)
    if not key:
        return []

    result = _exact_lookup(key)
    if result is not None:
        return list(result)

    result = _substring_lookup(key)
    if result is not None:
        return list(result)

    result = _fuzzy_lookup(key)
    if result is not None:
        return list(result)

    return []


def is_known(normalized_text: str) -> bool:
    """Convenience: True if resolve() would return a non-empty list."""
    return len(resolve(normalized_text)) > 0


def snapshot_dict() -> dict[str, list[str]]:
    """Return the full alias table for the snapshot test (G7 = C).

    A copy is returned so test code can't accidentally mutate the live table.
    """
    return {k: list(v) for k, v in sorted(LANDMARK_ALIASES.items())}
