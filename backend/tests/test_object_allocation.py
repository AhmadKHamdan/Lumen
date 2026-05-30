"""
Unit tests for the Object Allocation guidance policy (GuidanceTracker).

The tracker is the brain of the detection loop: temporal consistency (3-of-5),
throttled re-speaking, the scan / lost / timeout edges. It takes an injected
clock (``now``), so we can drive a scripted sequence of frames deterministically
without a real model, real audio, or real time.

Frame is 640x480 throughout. Detections are hand-built yolo_service.Detection
objects positioned to land in known region/distance buckets.

Run with::

    cd backend && python -m pytest tests/test_object_allocation.py
"""
from __future__ import annotations

from services.object_allocation import GuidanceTracker
from services.yolo_service import Detection

SHAPE = (480, 640)  # (H, W)

# A box dead-ahead and large -> (center, near) -> "right in front of you".
CENTER_NEAR = Detection("cup", 0.9, (195, 115, 445, 365))   # cx=320 (0.5), area_frac~0.20
# A small box hugging the left edge -> (left, far).
LEFT_FAR = Detection("cup", 0.9, (10, 200, 110, 300))       # cx=60 (0.094), area_frac~0.03


def _confirm(tracker, t0=100.0, step=0.2):
    """Drive three consecutive CENTER_NEAR frames to reach the confirmed state.

    Returns the (action, phrase) emitted on the third frame.
    """
    assert tracker.update([CENTER_NEAR], SHAPE, t0) is None          # hits=1
    assert tracker.update([CENTER_NEAR], SHAPE, t0 + step) is None    # hits=2
    return tracker.update([CENTER_NEAR], SHAPE, t0 + 2 * step)        # hits=3 -> guide


# ---------- temporal consistency ----------

def test_requires_three_of_five_before_speaking():
    tracker = GuidanceTracker("cup")
    res = _confirm(tracker)
    assert res is not None
    action, phrase = res
    assert action == "guide"
    assert phrase.lower().startswith("found")          # first sighting
    assert "right in front of you" in phrase.lower()    # center + near


def test_single_frame_flicker_is_ignored():
    tracker = GuidanceTracker("cup")
    # One detection then nothing: never reaches 3 hits, never guides.
    assert tracker.update([CENTER_NEAR], SHAPE, 100.0) is None
    assert tracker.update([], SHAPE, 100.2) is None
    assert tracker.update([], SHAPE, 100.4) is None


# ---------- throttling ----------

def test_same_position_within_window_does_not_repeat():
    tracker = GuidanceTracker("cup")
    _confirm(tracker, t0=100.0)
    # Same bucket, only 0.2s later -> stay quiet (reaffirm window is 6s).
    assert tracker.update([CENTER_NEAR], SHAPE, 100.6) is None


def test_position_change_triggers_new_guidance():
    tracker = GuidanceTracker("cup")
    _confirm(tracker, t0=100.0)
    res = tracker.update([LEFT_FAR], SHAPE, 100.6)
    assert res is not None
    action, phrase = res
    assert action == "guide"
    assert "to your left" in phrase.lower()
    assert not phrase.lower().startswith("found")   # not the first-seen phrase anymore


def test_reaffirm_after_interval_even_if_unchanged():
    tracker = GuidanceTracker("cup")
    _confirm(tracker, t0=100.0)            # last guide at t=100.4
    # Same bucket but past the 6s reaffirm window -> speak again.
    res = tracker.update([CENTER_NEAR], SHAPE, 100.4 + 6.0)
    assert res is not None and res[0] == "guide"


# ---------- scan prompts (never seen) ----------

def test_no_scan_prompt_immediately_at_start():
    tracker = GuidanceTracker("cup")
    assert tracker.update([], SHAPE, 500.0) is None


def test_scan_prompt_after_interval():
    tracker = GuidanceTracker("cup")
    assert tracker.update([], SHAPE, 500.0) is None          # sets the clock
    res = tracker.update([], SHAPE, 508.0)                   # +8s -> scan
    assert res is not None and res[0] == "scan"
    assert "turn" in res[1].lower()
    # Not again until another interval elapses.
    assert tracker.update([], SHAPE, 512.0) is None
    res2 = tracker.update([], SHAPE, 516.0)
    assert res2 is not None and res2[0] == "scan"


# ---------- lost from view ----------

def test_lost_announced_once_after_window_flushes():
    tracker = GuidanceTracker("cup")
    _confirm(tracker, t0=100.0)   # confirmed, last_region=center
    # Need the full 5-frame window to flush to all-absent before "lost" fires.
    t = 100.6
    results = []
    for _ in range(5):
        results.append(tracker.update([], SHAPE, t))
        t += 0.2
    actions = [r[0] for r in results if r is not None]
    assert actions.count("lost") == 1
    lost_phrase = next(r[1] for r in results if r is not None and r[0] == "lost")
    assert "straight ahead" in lost_phrase.lower()   # recalled last region (center)
    # Further absent frames stay silent.
    assert tracker.update([], SHAPE, t) is None


# ---------- timeout ----------

def test_timeout_when_never_seen():
    tracker = GuidanceTracker("cup", timeout_sec=60.0)
    assert tracker.update([], SHAPE, 1000.0) is None
    res = tracker.update([], SHAPE, 1060.0)   # exactly at the timeout
    assert res is not None and res[0] == "timeout"
    assert "couldn't find" in res[1].lower()


def test_no_timeout_once_seen():
    tracker = GuidanceTracker("cup", timeout_sec=60.0)
    _confirm(tracker, t0=1000.0)              # ever_seen = True
    # Long after the timeout window, but since we saw it, no auto-abort.
    res = tracker.update([], SHAPE, 1100.0)
    assert res is None or res[0] != "timeout"
