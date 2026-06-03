"""Unit tests for navigation.context."""

import time

from navigation.context import NavigationContext, Waypoint, WaypointStatus


# ----------------------------------------------------------------------
# Waypoint dataclass
# ----------------------------------------------------------------------

class TestWaypoint:
    def test_defaults(self):
        wp = Waypoint(raw_text="go to the doorway", normalized_text="doorway")
        assert wp.target_classes == []
        assert wp.locked_class is None
        assert wp.status == WaypointStatus.PENDING
        assert wp.added_at > 0

    def test_added_at_is_recent(self):
        before = time.time()
        wp = Waypoint(raw_text="x", normalized_text="x")
        after = time.time()
        assert before <= wp.added_at <= after


# ----------------------------------------------------------------------
# NavigationContext
# ----------------------------------------------------------------------

class TestNavigationContext:
    def test_initial_state(self):
        ctx = NavigationContext(destination="kitchen")
        assert ctx.destination == "kitchen"
        assert ctx.waypoints == []
        assert ctx.current_index == 0
        assert ctx.awaiting_waypoint is True
        assert ctx.retry_count == 0
        assert ctx.frame_processing_enabled is False
        assert ctx.current_waypoint() is None
        assert ctx.has_more_waypoints() is False

    def test_first_waypoint_becomes_active(self):
        ctx = NavigationContext(destination="kitchen")
        wp = ctx.add_waypoint("go to the doorway", "doorway")
        assert wp.status == WaypointStatus.ACTIVE
        assert ctx.current_waypoint() is wp

    def test_second_waypoint_is_pending(self):
        ctx = NavigationContext(destination="kitchen")
        ctx.add_waypoint("doorway", "doorway")
        second = ctx.add_waypoint("turn left", "turn left")
        assert second.status == WaypointStatus.PENDING
        assert ctx.current_waypoint().normalized_text == "doorway"
        assert ctx.has_more_waypoints() is True

    def test_mark_current_reached(self):
        ctx = NavigationContext(destination="kitchen")
        ctx.add_waypoint("doorway", "doorway")
        reached = ctx.mark_current_reached()
        assert reached is not None
        assert reached.status == WaypointStatus.REACHED

    def test_advance(self):
        ctx = NavigationContext(destination="kitchen")
        ctx.add_waypoint("doorway", "doorway")
        ctx.add_waypoint("turn left", "turn left")
        nxt = ctx.advance()
        assert nxt is not None
        assert nxt.normalized_text == "turn left"
        assert nxt.status == WaypointStatus.ACTIVE
        assert ctx.current_index == 1

    def test_advance_at_end_returns_none(self):
        ctx = NavigationContext(destination="kitchen")
        ctx.add_waypoint("doorway", "doorway")
        assert ctx.advance() is None
        assert ctx.current_index == 0

    def test_completed_waypoints_preserved_for_history(self):
        # Decision 4: list (not deque) — past waypoints stay accessible.
        ctx = NavigationContext(destination="kitchen")
        ctx.add_waypoint("doorway", "doorway")
        ctx.add_waypoint("turn left", "turn left")
        ctx.mark_current_reached()
        ctx.advance()
        # Past waypoint is still in the list and still marked REACHED.
        assert len(ctx.waypoints) == 2
        assert ctx.waypoints[0].status == WaypointStatus.REACHED

    def test_cancel_remaining_marks_skipped(self):
        ctx = NavigationContext(destination="kitchen")
        ctx.add_waypoint("doorway", "doorway")
        ctx.add_waypoint("turn left", "turn left")
        ctx.add_waypoint("kitchen", "kitchen")
        ctx.mark_current_reached()
        ctx.advance()
        # Now: reached, active, pending. Cancel from here.
        ctx.cancel_remaining()
        assert ctx.waypoints[0].status == WaypointStatus.REACHED   # untouched
        assert ctx.waypoints[1].status == WaypointStatus.SKIPPED
        assert ctx.waypoints[2].status == WaypointStatus.SKIPPED
