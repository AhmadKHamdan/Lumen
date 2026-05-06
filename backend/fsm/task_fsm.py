"""
Lumen task FSM.

Five states, deterministic transitions, no autonomous transitions. Every
transition is driven by an explicit event from outside (a user gesture, a
recognized voice command, a task completion signal, or a cancellation).

States
------
- ``Idle``                 : no active task, awaiting user
- ``ListeningForCommand``  : user pressed Start, captured a PTT recording, waiting for parse
- ``ObjectAllocationActive`` : object-allocation task in progress
- ``NavigationActive``       : navigation task in progress
- ``ReturningToIdle``        : transitional state - cleaning up before returning to Idle

Events
------
- ``user_start``        : Idle → ListeningForCommand
- ``user_stop``         : any active state → ReturningToIdle
- ``command_recognized`` : ListeningForCommand → ObjectAllocationActive | NavigationActive,
                           depending on payload['task_type']
- ``command_unknown``   : ListeningForCommand → ListeningForCommand (no-op, for symmetry)
- ``task_complete``     : Object/Nav/Returning → ReturningToIdle → Idle
- ``cleanup_done``      : ReturningToIdle → Idle (caller signals cleanup is done)

Subscribers
-----------
``subscribe(callback)`` registers a callable that will be invoked on every
state change with ``(old_state, new_state)``. Used by Session to push
``fsm_state`` JSON to the client.
"""
from __future__ import annotations

import enum
import logging
from typing import Callable, Optional

log = logging.getLogger("lumen.fsm")


class FSMState(enum.Enum):
    IDLE = "Idle"
    LISTENING = "ListeningForCommand"
    OBJECT_ACTIVE = "ObjectAllocationActive"
    NAV_ACTIVE = "NavigationActive"
    RETURNING = "ReturningToIdle"


# Type alias for subscriber callbacks.
StateChangeListener = Callable[[FSMState, FSMState], None]


class TaskFSM:
    """Per-session task FSM.

    Not thread-safe; the Session is single-threaded (asyncio).
    """

    def __init__(self) -> None:
        self._state: FSMState = FSMState.IDLE
        self._listeners: list[StateChangeListener] = []

    @property
    def state(self) -> FSMState:
        return self._state

    def subscribe(self, callback: StateChangeListener) -> None:
        self._listeners.append(callback)

    # ---------- transition entry point ----------

    def handle_event(self, event: str, payload: Optional[dict] = None) -> bool:
        """Try to apply ``event`` to the current state.

        Returns True if the event was accepted and a transition occurred (or
        the event was a valid no-op), False if the event was rejected as
        invalid in the current state.
        """
        next_state = self._next_state(event, payload or {})
        if next_state is None:
            log.warning("FSM rejected event %r in state %s",
                        event, self._state.name)
            return False
        if next_state == self._state:
            # Valid event, no state change (e.g. command_unknown stays in Listening)
            return True
        old = self._state
        self._state = next_state
        log.info("FSM %s -> %s on %r", old.name, next_state.name, event)
        self._notify(old, next_state)

        # ReturningToIdle is a transitional state that auto-completes if the
        # caller emits cleanup_done. We don't auto-fire it here - the rule is
        # "no autonomous transitions". The Session is responsible for emitting
        # cleanup_done after it tears down task context.
        return True

    # ---------- transition table ----------

    def _next_state(self, event: str, payload: dict) -> Optional[FSMState]:
        s = self._state

        if event == "user_start":
            if s == FSMState.IDLE:
                return FSMState.LISTENING
            return None

        if event == "user_stop":
            # Stop is valid from any non-Idle state and goes to Returning.
            # From Idle it's a no-op rejection (nothing to stop).
            if s == FSMState.IDLE:
                return None
            return FSMState.RETURNING

        if event == "command_recognized":
            if s != FSMState.LISTENING:
                return None
            ttype = payload.get("task_type")
            if ttype == "object_allocation":
                return FSMState.OBJECT_ACTIVE
            if ttype == "navigation":
                return FSMState.NAV_ACTIVE
            # Unknown task_type: stay listening for the next attempt.
            return FSMState.LISTENING

        if event == "command_unknown":
            if s == FSMState.LISTENING:
                return FSMState.LISTENING  # explicit no-op
            return None

        if event == "task_complete":
            if s in (FSMState.OBJECT_ACTIVE, FSMState.NAV_ACTIVE):
                return FSMState.RETURNING
            return None

        if event == "cleanup_done":
            if s == FSMState.RETURNING:
                return FSMState.IDLE
            return None

        log.warning("FSM unknown event %r", event)
        return None

    # ---------- internals ----------

    def _notify(self, old: FSMState, new: FSMState) -> None:
        for cb in list(self._listeners):
            try:
                cb(old, new)
            except Exception:
                log.exception("FSM listener raised; continuing")
