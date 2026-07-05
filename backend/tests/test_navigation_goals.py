"""Tests for services.navigation.goals - destination resolution + arrival rules.

Pure logic, no models: resolve_goal's alias/substring matching, the
primary/secondary indicator tables, and the >=1-primary-OR->=2-secondary
arrival rule.
"""
from __future__ import annotations

import pytest

from services.navigation.goals import (
    arrival_phrase,
    evaluate_arrival,
    indicators_for,
    is_known_goal,
    resolve_goal,
)


class TestResolveGoal:
    @pytest.mark.parametrize("text,expected", [
        ("kitchen", "kitchen"),
        ("the kitchen", "kitchen"),
        ("get me to the kitchen", "kitchen"),
        ("take me to the kitchen", "kitchen"),
        ("navigate to the washroom", "bathroom"),
        ("restroom", "bathroom"),
        ("toilet", "bathroom"),
        ("bed room", "bedroom"),
        ("the master bedroom", "bedroom"),   # substring match
        ("lounge", "living room"),
        ("study", "office"),
        ("dining area", "dining room"),
    ])
    def test_known_goals(self, text, expected):
        assert resolve_goal(text) == expected

    @pytest.mark.parametrize("text", ["", "the moon", "garage", "hallway"])
    def test_unknown_goals(self, text):
        assert resolve_goal(text) is None

    def test_is_known_goal(self):
        assert is_known_goal("get me to the kitchen")
        assert not is_known_goal("the moon")


class TestIndicators:
    def test_kitchen_split(self):
        primary, secondary = indicators_for("kitchen")
        assert "refrigerator" in primary
        assert "oven" in primary
        assert "sink" in secondary       # shared with bathroom -> secondary

    def test_alias_resolution(self):
        # indicators_for accepts unresolved phrases too.
        primary, _ = indicators_for("the washroom")
        assert "toilet" in primary

    def test_unknown_goal_is_empty(self):
        assert indicators_for("garage") == ([], [])


class TestEvaluateArrival:
    def test_one_primary_arrives(self):
        assert evaluate_arrival("kitchen", {"refrigerator"})["arrived"]

    def test_one_secondary_does_not_arrive(self):
        # A lone sink must never declare a kitchen (bathrooms have sinks too).
        assert not evaluate_arrival("kitchen", {"sink"})["arrived"]

    def test_two_secondary_arrive(self):
        assert evaluate_arrival("kitchen", {"sink", "dining table"})["arrived"]

    def test_empty_confirmed(self):
        result = evaluate_arrival("kitchen", set())
        assert not result["arrived"]
        assert result["matched_primary"] == []

    def test_irrelevant_classes_ignored(self):
        assert not evaluate_arrival("bedroom", {"refrigerator", "oven"})["arrived"]


class TestArrivalPhrase:
    def test_names_what_it_saw(self):
        result = evaluate_arrival("kitchen", {"refrigerator", "oven"})
        phrase = arrival_phrase("kitchen", result)
        assert "kitchen" in phrase
        assert "fridge" in phrase        # display name, not the COCO class name
        assert "an oven" in phrase       # correct article

    def test_bare_arrival_without_evidence(self):
        phrase = arrival_phrase("kitchen", {"matched_primary": [], "matched_secondary": []})
        assert phrase == "We've reached the kitchen."
