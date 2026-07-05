"""
Unit tests for the command parser.

≥20 realistic transcription samples covering:
- Clean cases ("find my cup", "navigate to the kitchen")
- Mishearings ("find my cop" → cup, "where is my fone" → cell phone, "go to the bafroom" → bathroom)
- Phrasing variations ("look for the bottle", "take me to the office")
- Multi-word targets ("find my dining table", "navigate to the living room")
- Empty / whitespace input
- Ambiguous / unsupported ("hello there", "what time is it") → unknown
- Synonym substitution ("phone" → "cell phone", "fridge" → "refrigerator")
- Confirmation phrase generation

Run with::

    cd backend && pytest tests/test_command_parser.py
"""
from __future__ import annotations

import pytest

from services.command_parser import (
    OBJECT_NOUNS,
    NAVIGATION_DESTINATIONS,
    confirmation_phrase,
    parse,
)


# ---------- happy-path object allocation ----------

@pytest.mark.parametrize("text, expected_target", [
    ("find my cup", "cup"),
    ("find the cup", "cup"),
    ("find a bottle", "bottle"),
    ("where is my book", "book"),
    ("where is the laptop", "laptop"),
    ("look for the chair", "chair"),
    ("look for my keyboard", "keyboard"),
    ("i need my remote", "remote"),
    ("i want my scissors", "scissors"),
    ("get me the cup", "cup"),
])
def test_object_allocation_clean_phrases(text, expected_target):
    result = parse(text)
    assert result["task_type"] == "object_allocation", f"for {text!r}: {result}"
    assert result["target"] == expected_target, f"for {text!r}: {result}"
    assert result["needs_clarification"] is False
    assert result["raw"] == text


# ---------- happy-path navigation ----------

@pytest.mark.parametrize("text, expected_target", [
    ("navigate to the kitchen", "kitchen"),
    ("navigate to bathroom", "bathroom"),
    ("take me to the bedroom", "bedroom"),
    ("take me to the living room", "living room"),
    ("go to the office", "office"),
    ("go to dining room", "dining room"),
    ("lead me to the bathroom", "bathroom"),
    ("guide me to the bedroom", "bedroom"),
])
def test_navigation_clean_phrases(text, expected_target):
    result = parse(text)
    assert result["task_type"] == "navigation", f"for {text!r}: {result}"
    assert result["target"] == expected_target, f"for {text!r}: {result}"
    assert result["needs_clarification"] is False


# ---------- mishearings (the classic Sprint_1.pdf cop→cup case) ----------

@pytest.mark.parametrize("text, expected_target", [
    ("find my cop", "cup"),                  # the canonical example
    ("find my cap", "cup"),                  # also close to cup
    ("where is my fone", "cell phone"),      # via SYNONYMS map
    ("find the cuup", "cup"),                # extra letters
    ("look for the buttle", "bottle"),       # b/u typo
    ("find my booook", "book"),              # repeated letters
])
def test_object_allocation_mishearings(text, expected_target):
    result = parse(text)
    assert result["task_type"] == "object_allocation", f"for {text!r}: {result}"
    assert result["target"] == expected_target, f"for {text!r}: {result}"


@pytest.mark.parametrize("text, expected_target", [
    ("go to the kichen", "kitchen"),         # missing 't'
    ("take me to the bafroom", "bathroom"),  # th→f
    ("navigate to the office", "office"),    # baseline (clean)
])
def test_navigation_mishearings(text, expected_target):
    result = parse(text)
    assert result["task_type"] == "navigation", f"for {text!r}: {result}"
    assert result["target"] == expected_target


# ---------- synonyms ----------

@pytest.mark.parametrize("text, expected_target", [
    ("find my phone", "cell phone"),
    ("find my mobile", "cell phone"),
    ("where is the television", "tv"),
    ("find my fridge", "refrigerator"),
    ("look for the sofa", "couch"),
])
def test_object_synonyms(text, expected_target):
    result = parse(text)
    assert result["task_type"] == "object_allocation"
    assert result["target"] == expected_target


@pytest.mark.parametrize("text, expected_target", [
    ("take me to the restroom", "bathroom"),
    ("go to the lounge", "living room"),
    ("take me to the washroom", "bathroom"),
    ("navigate to the lounge", "living room"),
])
def test_navigation_synonyms(text, expected_target):
    result = parse(text)
    assert result["task_type"] == "navigation"
    assert result["target"] == expected_target


# ---------- multi-word targets ----------

def test_multiword_object_target():
    result = parse("find my dining table")
    assert result["task_type"] == "object_allocation"
    assert result["target"] == "dining table"


def test_multiword_navigation_target():
    result = parse("take me to the living room")
    assert result["task_type"] == "navigation"
    assert result["target"] == "living room"


# ---------- ambiguous / unsupported ----------

@pytest.mark.parametrize("text", [
    "hello there",
    "what time is it",
    "tell me a joke",
    "the weather is nice today",
    "asdf qwerty",
    "",
    "   ",
])
def test_unsupported_returns_unknown(text):
    result = parse(text)
    assert result["task_type"] == "unknown", f"for {text!r}: {result}"
    assert result["target"] is None
    assert result["needs_clarification"] is True


# ---------- punctuation / casing tolerance ----------

def test_uppercase_input():
    result = parse("FIND MY CUP")
    assert result["task_type"] == "object_allocation"
    assert result["target"] == "cup"


def test_punctuation_tolerated():
    result = parse("Find my cup, please.")
    assert result["task_type"] == "object_allocation"
    assert result["target"] == "cup"


def test_extra_whitespace_tolerated():
    result = parse("  find    my   cup  ")
    assert result["task_type"] == "object_allocation"
    assert result["target"] == "cup"


# ---------- confirmation phrase generator ----------

def test_confirmation_object_allocation():
    intent = {"task_type": "object_allocation", "target": "cup"}
    assert confirmation_phrase(intent) == "Looking for your cup."


def test_confirmation_navigation():
    intent = {"task_type": "navigation", "target": "kitchen"}
    assert confirmation_phrase(intent) == "Navigating to the kitchen."


def test_confirmation_unknown():
    intent = {"task_type": "unknown", "target": None}
    assert confirmation_phrase(intent) == "I didn't catch that, please repeat."


def test_confirmation_object_with_multiword_target():
    intent = {"task_type": "object_allocation", "target": "dining table"}
    assert confirmation_phrase(intent) == "Looking for your dining table."


# ---------- score sanity ----------

def test_clean_input_has_high_score():
    result = parse("find my cup")
    assert result["score"] >= 90


def test_misheard_input_has_lower_score():
    clean = parse("find my cup")
    fuzzy = parse("find my cop")
    # Both should resolve to cup, but fuzzy's score should be lower than clean's.
    assert fuzzy["target"] == "cup"
    assert fuzzy["score"] < clean["score"]


# ---------- coverage sanity ----------

def test_all_object_nouns_have_at_least_one_template_match():
    """For each supported COCO noun, "find my <noun>" should parse correctly.
    Catches accidental noun-list / template-list mismatches."""
    for noun in OBJECT_NOUNS:
        result = parse(f"find my {noun}")
        assert result["task_type"] == "object_allocation", \
            f"noun={noun} -> {result}"
        assert result["target"] == noun, f"noun={noun} -> {result}"


def test_all_destinations_have_at_least_one_template_match():
    for dest in NAVIGATION_DESTINATIONS:
        result = parse(f"go to the {dest}")
        assert result["task_type"] == "navigation", f"dest={dest} -> {result}"
        assert result["target"] == dest, f"dest={dest} -> {result}"
