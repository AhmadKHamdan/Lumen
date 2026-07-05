"""
Unit tests for the info (describe / repeat) branch of the command parser.

Confirms short "what's around me" / "repeat that" style utterances are
classified as ``task_type == "info"`` with target ``describe`` or ``repeat``,
and don't cannibalise real find / navigate / completion commands.

Run with::

    cd backend && python -m pytest tests/test_command_info.py
"""
from __future__ import annotations

import pytest

from services.command_parser import parse


@pytest.mark.parametrize("text", [
    "describe",
    "describe the scene",
    "what do you see",
    "what can you see",
    "what is in front of me",
    "whats in front of me",
    "whats around me",
    "look around",
    "tell me what you see",
])
def test_describe_phrases(text):
    intent = parse(text)
    assert intent["task_type"] == "info", text
    assert intent["target"] == "describe", text
    assert intent["needs_clarification"] is False


@pytest.mark.parametrize("text", [
    "repeat",
    "repeat that",
    "say that again",
    "say it again",
    "what did you say",
    "once more",
])
def test_repeat_phrases(text):
    intent = parse(text)
    assert intent["task_type"] == "info", text
    assert intent["target"] == "repeat", text


# Regressions: info phrases must NOT swallow real find / navigate / cancel /
# unknown utterances.

@pytest.mark.parametrize("text, expected_type", [
    ("find my cup", "object_allocation"),
    ("navigate to the kitchen", "navigation"),
    ("got it", "completion"),
    ("never mind", "completion"),
    ("hello there", "unknown"),
])
def test_info_does_not_steal_other_intents(text, expected_type):
    assert parse(text)["task_type"] == expected_type, text


def test_describe_helper_composes_scene_phrase():
    """describe_scene_phrase takes YOLO Detections and builds a sentence."""
    from services.guidance_generator import describe_scene_phrase
    from services.yolo_service import Detection

    dets = [
        Detection("cup", 0.9, (50, 50, 100, 100)),        # cx=75/640=~0.12 (left)
        Detection("laptop", 0.85, (280, 200, 400, 350)),  # cx=340/640=~0.53 (center)
        Detection("chair", 0.8, (500, 100, 620, 400)),    # cx=560/640=~0.87 (right)
    ]
    phrase = describe_scene_phrase(dets, (480, 640))
    p = phrase.lower()
    # All three objects mentioned, in some order.
    assert "cup" in p
    assert "laptop" in p
    assert "chair" in p
    # Direction clauses for all three regions.
    assert "on your left" in p
    assert "straight ahead" in p
    assert "on your right" in p


def test_describe_helper_handles_empty_scene():
    from services.guidance_generator import describe_scene_phrase
    phrase = describe_scene_phrase([], (480, 640))
    assert "don't see" in phrase.lower() or "nothing" in phrase.lower()
