"""
Unit tests for the task FSM.

Covers:
- Valid transitions are accepted and the state actually changes
- Invalid transitions are rejected (handle_event returns False, state unchanged)
- No autonomous transitions (state only changes on explicit handle_event calls)
- Subscribers fire with (old, new) on every transition

Run with::

    cd backend && pytest tests/test_fsm.py
"""
from __future__ import annotations

import pytest

from fsm.task_fsm import FSMState, TaskFSM


# ---------- happy paths ----------

def test_initial_state_is_idle():
    fsm = TaskFSM()
    assert fsm.state == FSMState.IDLE


def test_idle_to_listening_on_user_start():
    fsm = TaskFSM()
    assert fsm.handle_event("user_start") is True
    assert fsm.state == FSMState.LISTENING


def test_listening_to_object_active_on_command():
    fsm = TaskFSM()
    fsm.handle_event("user_start")
    ok = fsm.handle_event("command_recognized",
                          payload={"task_type": "object_allocation",
                                   "target": "cup"})
    assert ok is True
    assert fsm.state == FSMState.OBJECT_ACTIVE


def test_listening_to_nav_active_on_command():
    fsm = TaskFSM()
    fsm.handle_event("user_start")
    ok = fsm.handle_event("command_recognized",
                          payload={"task_type": "navigation",
                                   "target": "kitchen"})
    assert ok is True
    assert fsm.state == FSMState.NAV_ACTIVE


def test_object_active_to_returning_on_user_stop():
    fsm = TaskFSM()
    fsm.handle_event("user_start")
    fsm.handle_event("command_recognized",
                     payload={"task_type": "object_allocation", "target": "cup"})
    assert fsm.handle_event("user_stop") is True
    assert fsm.state == FSMState.RETURNING


def test_returning_to_idle_on_cleanup_done():
    fsm = TaskFSM()
    fsm.handle_event("user_start")
    fsm.handle_event("command_recognized",
                     payload={"task_type": "object_allocation", "target": "cup"})
    fsm.handle_event("user_stop")
    assert fsm.handle_event("cleanup_done") is True
    assert fsm.state == FSMState.IDLE


def test_full_object_allocation_round_trip():
    """A full happy-path round trip from idle back to idle."""
    fsm = TaskFSM()
    states = [fsm.state]
    listener = []
    fsm.subscribe(lambda old, new: listener.append((old, new)))

    assert fsm.handle_event("user_start")
    states.append(fsm.state)
    assert fsm.handle_event("command_recognized",
                            payload={"task_type": "object_allocation", "target": "cup"})
    states.append(fsm.state)
    assert fsm.handle_event("task_complete")
    states.append(fsm.state)
    assert fsm.handle_event("cleanup_done")
    states.append(fsm.state)

    assert states == [
        FSMState.IDLE,
        FSMState.LISTENING,
        FSMState.OBJECT_ACTIVE,
        FSMState.RETURNING,
        FSMState.IDLE,
    ]
    # Subscriber should have fired four times (one per transition)
    assert len(listener) == 4


# ---------- invalid transitions ----------

def test_command_recognized_rejected_in_idle():
    fsm = TaskFSM()
    ok = fsm.handle_event("command_recognized",
                          payload={"task_type": "object_allocation", "target": "cup"})
    assert ok is False
    assert fsm.state == FSMState.IDLE


def test_user_start_rejected_in_listening():
    fsm = TaskFSM()
    fsm.handle_event("user_start")
    ok = fsm.handle_event("user_start")
    assert ok is False
    assert fsm.state == FSMState.LISTENING


def test_user_stop_rejected_in_idle():
    fsm = TaskFSM()
    ok = fsm.handle_event("user_stop")
    assert ok is False
    assert fsm.state == FSMState.IDLE


def test_cleanup_done_rejected_outside_returning():
    fsm = TaskFSM()
    assert fsm.handle_event("cleanup_done") is False
    fsm.handle_event("user_start")
    assert fsm.handle_event("cleanup_done") is False


def test_task_complete_rejected_in_listening():
    fsm = TaskFSM()
    fsm.handle_event("user_start")
    ok = fsm.handle_event("task_complete")
    assert ok is False


def test_unknown_event_rejected():
    fsm = TaskFSM()
    assert fsm.handle_event("totally_made_up_event") is False
    assert fsm.state == FSMState.IDLE


def test_unknown_task_type_stays_listening():
    """command_recognized with task_type=='unknown' is a no-op (stay listening)."""
    fsm = TaskFSM()
    fsm.handle_event("user_start")
    ok = fsm.handle_event("command_recognized",
                          payload={"task_type": "unknown"})
    assert ok is True
    assert fsm.state == FSMState.LISTENING


# ---------- no autonomous transitions ----------

def test_state_does_not_change_without_event():
    fsm = TaskFSM()
    fsm.handle_event("user_start")
    snapshot = fsm.state
    # No event, no transition.
    assert fsm.state == snapshot


def test_subscribe_not_called_on_rejected_event():
    fsm = TaskFSM()
    calls = []
    fsm.subscribe(lambda old, new: calls.append((old, new)))
    fsm.handle_event("user_stop")  # invalid in IDLE
    assert calls == []


def test_subscribe_not_called_on_no_op_event():
    """command_unknown in LISTENING is a valid no-op - state is the same,
    so we should NOT fire the subscriber."""
    fsm = TaskFSM()
    fsm.handle_event("user_start")
    calls = []
    fsm.subscribe(lambda old, new: calls.append((old, new)))
    fsm.handle_event("command_unknown")
    assert calls == []


# ---------- subscriber semantics ----------

def test_subscriber_receives_old_and_new():
    fsm = TaskFSM()
    seen = []
    fsm.subscribe(lambda old, new: seen.append((old, new)))
    fsm.handle_event("user_start")
    assert seen == [(FSMState.IDLE, FSMState.LISTENING)]


def test_listener_exception_does_not_break_fsm():
    """A buggy subscriber must not break the FSM for other subscribers."""
    fsm = TaskFSM()

    def bad(old, new):
        raise RuntimeError("oops")

    good_calls = []
    fsm.subscribe(bad)
    fsm.subscribe(lambda old, new: good_calls.append(new))
    fsm.handle_event("user_start")
    assert good_calls == [FSMState.LISTENING]
    assert fsm.state == FSMState.LISTENING


# ---------- stop from various active states ----------

@pytest.mark.parametrize("active_state_setup", [
    pytest.param(
        lambda fsm: (fsm.handle_event("user_start"),
                     fsm.handle_event("command_recognized",
                                      payload={"task_type": "object_allocation",
                                               "target": "cup"})),
        id="from-object-active",
    ),
    pytest.param(
        lambda fsm: (fsm.handle_event("user_start"),
                     fsm.handle_event("command_recognized",
                                      payload={"task_type": "navigation",
                                               "target": "kitchen"})),
        id="from-nav-active",
    ),
    pytest.param(
        lambda fsm: (fsm.handle_event("user_start"),),
        id="from-listening",
    ),
])
def test_user_stop_works_from_any_active_state(active_state_setup):
    fsm = TaskFSM()
    active_state_setup(fsm)
    assert fsm.state != FSMState.IDLE
    assert fsm.handle_event("user_stop") is True
    assert fsm.state == FSMState.RETURNING
