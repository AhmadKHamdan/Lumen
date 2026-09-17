"""
Guidance phrase generator for the Object Allocation task.

Turns a (target, :class:`SpatialInfo`) pair into a short, natural spoken
sentence, plus a handful of situational phrases (scanning, lost-from-view,
timeout). Pure templating - no model, no I/O - so it unit-tests cleanly and
stays fast.

Design notes
------------
- Phrases are short and front-load the direction word, because that's the
  part a moving listener most needs ("left", "right", "ahead").
- We address the user in the second person and call the object "your cup".
- The center+near phrase nudges the user to reach out, but never declares the
  task done: completion is always user-driven (the user says "got it").
"""
from __future__ import annotations

from typing import Optional

# Forward reference only for typing; avoids a hard import cycle.
try:  # pragma: no cover - typing convenience
    from services.spatial_reasoning import SpatialInfo
except Exception:  # pragma: no cover
    SpatialInfo = object  # type: ignore


def _direction_clause(region: str) -> str:
    """Map a region bucket to a leading direction clause."""
    if region == "left":
        return "to your left"
    if region == "right":
        return "to your right"
    return "straight ahead"  # center


def _distance_clause(distance: str) -> str:
    """Map a distance bucket to a trailing distance clause (may be empty)."""
    if distance == "near":
        return "close by"
    if distance == "far":
        return "but it's far away"
    return "a few steps away"  # medium


def _hand_invite(target: str) -> str:
    """The proximity instruction: a CONCRETE action that starts hand guidance.
    Never a vague "reach forward" — the user asked for an explicit cue to put
    their hand where the camera can see it, so the reach guidance can begin."""
    return ("Raise your hand up in front of the camera, and I'll guide "
            f"your hand to the {target}.")


def guidance_phrase(target: str, info: "SpatialInfo", invite: bool = True) -> str:
    """Compose the main guidance sentence for a confirmed detection.

    Examples
    --------
    center + near  -> "Your cup is right in front of you, within arm's reach.
                       Raise your hand up in front of the camera, ..."
    left   + medium-> "Your cup is to your left, a few steps away."
    right  + far   -> "Your cup is to your right, but it's far away."
    """
    target = target or "object"
    region = getattr(info, "region", "center")
    distance = getattr(info, "distance", "medium")

    # In proximity (any direction): say where it is, then invite the hand into
    # the frame so reach guidance takes over. This is the handoff point from
    # body navigation to hand navigation.
    if distance == "near":
        where = ("right in front of you" if region == "center"
                 else _direction_clause(region))
        base = f"Your {target} is {where}, within arm's reach."
        # The caller (GuidanceTracker) rate-limits the invite: repeating the
        # full instruction every re-affirm while nothing changed is chatter.
        return base + (" " + _hand_invite(target) if invite else "")

    direction = _direction_clause(region)
    dist = _distance_clause(distance)

    if region == "center":
        # "straight ahead" already reads as a full clause.
        if distance == "far":
            return f"Your {target} is {direction}, {dist}. Keep moving forward."
        return f"Your {target} is {direction}, {dist}."

    # left / right
    return f"Your {target} is {direction}, {dist}."


def first_seen_phrase(target: str, info: "SpatialInfo", invite: bool = True) -> str:
    """Phrase used the first time the target is spotted - leads with 'Found'."""
    target = target or "object"
    region = getattr(info, "region", "center")
    distance = getattr(info, "distance", "medium")
    if distance == "near":
        where = ("right in front of you" if region == "center"
                 else _direction_clause(region))
        base = f"Found your {target}. It's {where}, within arm's reach."
        return base + (" " + _hand_invite(target) if invite else "")
    direction = _direction_clause(region)
    return f"Found your {target}, {direction}."


def refound_phrase(target: str, info: "SpatialInfo") -> str:
    """Short re-acquisition line: the target came back into view soon after a
    'lost sight' announcement, in the same place. No 'Found your...' fanfare,
    no hand invite — the user did nothing wrong and needs no new instruction."""
    target = target or "object"
    region = getattr(info, "region", "center")
    where = ("right in front of you" if region == "center"
             else _direction_clause(region))
    return f"I see your {target} again, {where}."


def scanning_phrase(target: str) -> str:
    """Spoken periodically while the target hasn't been seen yet."""
    target = target or "object"
    return f"I don't see your {target} yet. Slowly turn around so I can look."


def lost_phrase(target: str, last_region: Optional[str]) -> str:
    """Spoken once when a previously visible target drops out of view."""
    target = target or "object"
    if last_region in ("left", "right"):
        return f"I lost sight of your {target}. It was to your {last_region}."
    if last_region == "center":
        return f"I lost sight of your {target}. It was straight ahead."
    return f"I lost sight of your {target}. Try moving the camera slowly."


def timeout_phrase(target: str) -> str:
    """Spoken when the search times out without ever finding the target."""
    target = target or "object"
    return f"I couldn't find your {target}. Stopping the search."


def complete_phrase(target: str) -> str:
    """Spoken when the user confirms they have the object."""
    target = target or "object"
    return f"Great. Glad you found your {target}."


_DIRECTION_CLAUSE = {
    "left": "on your left",
    "right": "on your right",
    "center": "straight ahead",
}

# YOLO class names that read poorly out loud -> what the voice actually says.
_SPOKEN_NAMES = {
    "tv": "screen",
    "dining table": "table",
    "potted plant": "plant",
}


def describe_scene_phrase(detections, frame_shape) -> str:
    """Compose a spoken scene readout for the "describe" voice command.

    Takes up to the four highest-confidence YOLO detections, classifies each
    into a region via spatial_reasoning, and GROUPS objects that share a
    direction so the clause is spoken once: "a keyboard and a TV straight
    ahead", never "a keyboard straight ahead ... a TV straight ahead".
    Pure - no I/O, no model.
    """
    # Local import so this module stays lightweight; spatial_reasoning is
    # itself pure and cheap.
    from collections import Counter
    from services import spatial_reasoning

    if not detections:
        return "I don't see anything I recognise."
    h, w = int(frame_shape[0]), int(frame_shape[1])
    top = sorted(detections, key=lambda d: d.confidence, reverse=True)[:4]

    by_region: dict[str, list[str]] = {}
    for det in top:
        info = spatial_reasoning.locate(det.box, w, h, label=det.label)
        by_region.setdefault(info.region, []).append(
            _SPOKEN_NAMES.get(det.label, det.label))

    def _join(items: list[str]) -> str:
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return ", ".join(items[:-1]) + f", and {items[-1]}"

    parts: list[str] = []
    # Stable spatial order: left -> ahead -> right (unknown regions last).
    for region in ("left", "center", "right", *by_region.keys()):
        labels = by_region.pop(region, None)
        if not labels:
            continue
        counts = Counter(labels)
        names = _join([f"a {lbl}" if n == 1 else f"{n} {lbl}s"
                       for lbl, n in counts.items()])
        clause = _DIRECTION_CLAUSE.get(region, "in front of you")
        parts.append(f"{names} {clause}")
    return "I see " + _join(parts) + "."


def cancel_phrase(target: str) -> str:
    """Spoken when the user cancels the search.

    Signals readiness for the next command (we drop back to LISTENING rather
    than ending the session) so the user knows they can immediately speak
    another find/navigate command without pressing Start again.
    """
    return "Okay, stopping. Ready for your next command."
