"""Obstacle classification (Sprint 4 Part 3).

Decisions from the Part 3 MCQs:
- A1+A4+A5 (combined): broad obstacle set — every COCO class that could
  plausibly be an indoor path hazard. Excludes outdoor/sky classes.
- A5 (size gate): "small" objects (cup, book, bottle, fork, etc.) need to be
  BOTH near AND occupy enough of the frame to indicate a real foot-strike or
  collision risk. Otherwise we get spurious warnings about a teacup across
  the room.
- A3: "person" is its own category for distinct phrasing and priority (E6).
- A2: the active landmark's class is exempted at the manager level, not here.

The obstacle classifier is a pure function module:
    is_obstacle(class_name) -> bool
    is_small_object(class_name) -> bool
    SMALL_OBJECT_MIN_BBOX_AREA_RATIO -> threshold for the size gate
"""

from __future__ import annotations

from typing import Final


# ---------------------------------------------------------------------------
# Large / structural / human obstacles — warn whenever they are "near" + center
# ---------------------------------------------------------------------------

# People are a separate category for priority and phrasing (A3 + E6).
PERSON_CLASSES: Final[frozenset[str]] = frozenset({"person"})

# Large furniture / appliances / fixtures — physically large enough to walk
# into without any size gate.
LARGE_OBSTACLES: Final[frozenset[str]] = frozenset({
    # Furniture
    "chair", "couch", "bed", "dining table", "bench", "potted plant",
    # Appliances / kitchen
    "refrigerator", "oven", "microwave", "toaster", "sink",
    # Bathroom
    "toilet",
    # Electronics / fixtures
    "tv", "laptop",
    # Reserved architectural — currently not in COCO but kept here for the
    # Sprint 5 fine-tune (A6=E).
    "door",
})

# Medium objects that could trip a user or be walked into — bbox already large
# at "near" so no extra size gate needed.
MEDIUM_OBSTACLES: Final[frozenset[str]] = frozenset({
    "backpack", "handbag", "suitcase", "umbrella",
    "keyboard", "mouse", "remote",
    "bowl",  # large bowl on the floor / table edge
})

# Small objects — toe-strike / shin-strike hazards. These trigger ONLY if
# bbox area is also large enough (the size gate) to avoid warning about a
# distant teacup the user will never approach.
SMALL_OBSTACLES: Final[frozenset[str]] = frozenset({
    "cup", "bottle", "wine glass",
    "book", "vase", "clock",
    "fork", "knife", "spoon",
    "banana", "apple", "orange",
    "cell phone", "scissors",
})

# Pets — moving low-to-the-ground hazards, treated as obstacles but not "person".
PET_CLASSES: Final[frozenset[str]] = frozenset({"cat", "dog"})


# All obstacle classes combined for fast membership check.
OBSTACLE_CLASSES: Final[frozenset[str]] = (
    PERSON_CLASSES | LARGE_OBSTACLES | MEDIUM_OBSTACLES
    | SMALL_OBSTACLES | PET_CLASSES
)


# Size gate threshold for SMALL_OBSTACLES (decision A5).
# A typical YOLOv8 frame is 640x640 = 409,600 px. 8% = ~32,800 px which
# corresponds roughly to an object occupying a ~180x180 region — about the
# size of a cup at arm's length. Tuned empirically; revisit in Sprint 5.
SMALL_OBJECT_MIN_BBOX_AREA_RATIO: Final[float] = 0.08


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_obstacle(class_name: str) -> bool:
    """True if this class is a candidate obstacle.

    Whether it actually triggers a warning depends on region/distance gates
    applied by the LandmarkDetector (decisions B1, B2, B3, and the size gate
    for small objects below).
    """
    return class_name in OBSTACLE_CLASSES


def is_person(class_name: str) -> bool:
    """A3: people are dynamic hazards with their own priority class."""
    return class_name in PERSON_CLASSES


def is_small_object(class_name: str) -> bool:
    """A5: small objects need an additional size gate before warning."""
    return class_name in SMALL_OBSTACLES


def bbox_area_ratio(bbox: list, frame_area: float = 640.0 * 640.0) -> float:
    """Return the fraction of the frame the bbox occupies.

    The frame_area default matches YOLOv8 inference resolution (640x640).
    Ahmad's pipeline normalizes to this size before detection. If a different
    size is ever used, callers can pass the actual frame_area.
    """
    if not bbox or len(bbox) < 4 or frame_area <= 0:
        return 0.0
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return (w * h) / frame_area


def passes_size_gate(class_name: str, bbox: list) -> bool:
    """For SMALL_OBSTACLES, returns True iff bbox area ratio meets the
    threshold. For non-small obstacles, always returns True.
    """
    if not is_small_object(class_name):
        return True
    return bbox_area_ratio(bbox) >= SMALL_OBJECT_MIN_BBOX_AREA_RATIO


def obstacle_priority(class_name: str) -> int:
    """E6: priority for tie-break when multiple obstacles share the path.

    Higher number = higher priority. Person beats furniture; pets beat small
    objects; tie-break by nearest, handled in the detector.
    """
    if is_person(class_name):
        return 3
    if class_name in PET_CLASSES:
        return 2
    if class_name in LARGE_OBSTACLES:
        return 1
    return 0


def display_name(class_name: str) -> str:
    """Render a class name for the user.

    Most COCO class names are already user-friendly; this hook lets us tweak
    the few that aren't (e.g. 'dining table' is fine, 'tv' might be 'TV').
    """
    overrides = {
        "tv": "TV",
        "potted plant": "plant",
    }
    return overrides.get(class_name, class_name)
