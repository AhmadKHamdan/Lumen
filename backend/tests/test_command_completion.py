"""
Unit tests for the completion/cancel branch of the command parser (Sprint 3).

Confirms that short "I'm done" / "abort" style utterances are classified as
``task_type == "completion"`` with target ``confirm`` or ``cancel``, while the
existing find/navigate commands are NOT mis-stolen by the new branch.

Run with::

    cd backend && python -m pytest tests/test_command_completion.py
"""
from __future__ import annotations

import pytest

from services.command_parser import confirmation_phrase, parse


# ---------- confirm phrases ----------

@pytest.mark.parametrize("text", [
    "got it",
    "i got it",
    "found it",
    "i found it",
    "i have it",
    "thanks",
    "thank you",
    "done",
    "okay got it thanks",   # filler around the core phrase
])
def test_confirm_phrases(text):
    intent = parse(text)
    assert intent["task_type"] == "completion", text
    assert intent["target"] == "confirm", text
    assert intent["needs_clarification"] is False


# ---------- cancel phrases ----------

@pytest.mark.parametrize("text", [
    "stop",
    "stop it",
    "cancel",
    "never mind",
    "nevermind",
    "forget it",
    "quit",
    "abort",
])
def test_cancel_phrases(text):
    intent = parse(text)
    assert intent["task_type"] == "completion", text
    assert intent["target"] == "cancel", text


# ---------- regression: real commands are NOT classified as completion ----------

@pytest.mark.parametrize("text, expected_type, expected_target", [
    ("find my cup", "object_allocation", "cup"),
    ("find the bottle", "object_allocation", "bottle"),
    ("where is my book", "object_allocation", "book"),
    ("find my dining table", "object_allocation", "dining table"),
    ("navigate to the kitchen", "navigation", "kitchen"),
    ("take me to the office", "navigation", "office"),
])
def test_real_commands_not_stolen(text, expected_type, expected_target):
    intent = parse(text)
    assert intent["task_type"] == expected_type, text
    assert intent["target"] == expected_target, text


@pytest.mark.parametrize("text", ["hello there", "what time is it", "tell me a joke"])
def test_non_commands_still_unknown(text):
    assert parse(text)["task_type"] == "unknown", text


# ---------- confirmation phrase for completion ----------

def test_confirmation_phrase_completion():
    assert confirmation_phrase({"task_type": "completion", "target": "confirm"}) == "Got it."
    assert confirmation_phrase({"task_type": "completion", "target": "cancel"}) == "Okay."
