"""Lumen goal-directed exploration controller — self-contained.

Layered modules:
  config      tuning constants (the one place to adjust behavior)
  goals       goal -> room-indicator map + arrival rule (the destination knowledge)
  models      model loading + warmup (loads on import)
  state       the FSM state + per-room evidence buffers + phase-entry helpers
  geometry    compass/bearing math (start-anchored directions, clustering, turns)
  perception  one frame -> trustworthy detections (COCO + door stack + door geometry)
  obstacles   the walking-lane watchdog (YOLO corridor + depth tripwire)
  controller  the two-phase state machine + everything Lumen says

`webdemo/server.py` is the thin FastAPI layer that wires these together.
"""
from __future__ import annotations
