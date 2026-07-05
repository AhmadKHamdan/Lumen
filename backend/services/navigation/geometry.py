"""Compass + bearing math: where things are relative to the scan's start direction,
how to merge repeated sightings of one object, and how far to turn to face a heading.

All directions are anchored to the START heading of the current scan (the fixed
reference), which lives in `st["ref_heading"]`. These helpers are otherwise pure.
"""
from __future__ import annotations

from .config import BUCKET_DEG, DOOR_CLUSTER_DEG


def _signed_from_ref(bucket: int) -> float:
    """Signed degrees of a sector's centre from the START direction (+ right, - left).
    The start direction is our fixed anchor; the summary is described relative to it."""
    deg = (bucket * BUCKET_DEG) % 360.0
    return ((deg + 180.0) % 360.0) - 180.0  # wrap to (-180, 180]


def _signed_from_ref_deg(st, abs_heading: float) -> float:
    """Signed degrees of an absolute compass heading from the START direction
    (+ right, - left), wrapped to (-180, 180]."""
    ref = st["ref_heading"] or 0.0
    return ((abs_heading - ref + 180.0) % 360.0) - 180.0


def _cluster_bearings(st, bearings: list[float]) -> list[tuple[float, int]]:
    """Collapse accumulated sighting bearings into distinct physical objects.

    Sightings of one object land within a few degrees of each other; sightings of two
    different ones are far apart. We sort by angle-from-start and split wherever a
    gap exceeds DOOR_CLUSTER_DEG. Returns [(mean_signed_deg_from_start, n_sightings)],
    so the summary names each object once and we get a precise heading to face."""
    if not bearings:
        return []
    signed = sorted(_signed_from_ref_deg(st, b) for b in bearings)
    groups: list[list[float]] = [[signed[0]]]
    for s in signed[1:]:
        if s - groups[-1][-1] <= DOOR_CLUSTER_DEG:
            groups[-1].append(s)
        else:
            groups.append([s])
    # A door directly behind the user straddles the +/-180 seam and lands in both the
    # first and last group — merge them across the wrap (shift the top group by -360
    # so the mean comes out right, e.g. [+179, -179] -> -180, not 0).
    if len(groups) > 1 and (signed[0] + 360.0) - signed[-1] <= DOOR_CLUSTER_DEG:
        groups[0] = [s - 360.0 for s in groups.pop()] + groups[0]
    out = []
    for g in groups:
        mean = sum(g) / len(g)
        out.append((((mean + 180.0) % 360.0) - 180.0, len(g)))  # re-wrap to (-180, 180]
    return out


def _cluster_doors(st) -> list[tuple[float, int]]:
    return _cluster_bearings(st, st["door_bearings"])


def _turn_to(target_heading: float, current_heading: float) -> float:
    """How far to turn from where you face NOW to a heading (+ = right, - = left).
    Needed only to *walk* the user through the turn — the door's position itself
    stays anchored to the start direction."""
    return ((target_heading - current_heading + 180.0) % 360.0) - 180.0


def _track_turn(st, heading: float | None) -> None:
    """Advance the discover-scan rotation total. Runs on EVERY frame (even blurred
    ones the detector skips) so a fast segment never stalls the full-circle check.
    Ignores large compass glitches."""
    if heading is None or st["last_heading"] is None:
        return
    d = ((heading - st["last_heading"] + 180.0) % 360.0) - 180.0
    st["last_heading"] = heading
    if abs(d) <= 120.0:
        st["net_rotation"] += d


def _direction(signed_deg: float) -> str:
    """Map a signed angle from 'ahead' to a spoken relative direction."""
    a = signed_deg
    if -30 <= a <= 30:
        return "ahead"
    if 30 < a <= 90:
        return "on your right"
    if 90 < a <= 150:
        return "behind you, to the right"
    if a > 150 or a < -150:
        return "behind you"
    if -150 <= a < -90:
        return "behind you, to the left"
    return "on your left"  # -90 <= a < -30
