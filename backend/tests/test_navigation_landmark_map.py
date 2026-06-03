"""Unit tests for navigation.landmark_map."""

import json
from pathlib import Path

import pytest

from navigation.landmark_map import (
    COCO_CLASSES,
    LANDMARK_ALIASES,
    FUZZY_THRESHOLD,
    resolve,
    is_known,
    snapshot_dict,
)


# ----------------------------------------------------------------------
# Exact-match resolution
# ----------------------------------------------------------------------

class TestExactMatch:
    def test_simple_alias(self):
        assert resolve("chair") == ["chair"]

    def test_alias_to_multiple_classes(self):
        # A2: one alias -> several COCO classes.
        result = resolve("seat")
        assert "chair" in result
        assert "couch" in result
        assert "bench" in result

    def test_case_insensitive(self):
        # A7: lowercase keys, lowercase lookup.
        assert resolve("CHAIR") == resolve("chair")
        assert resolve("Doorway") == resolve("doorway")

    def test_whitespace_tolerant(self):
        assert resolve("  doorway  ") == resolve("doorway")
        assert resolve("dining\ttable") == resolve("dining table")

    def test_synonyms_map_to_same_class(self):
        assert resolve("fridge") == resolve("refrigerator") == ["refrigerator"]
        assert resolve("sofa") == resolve("couch") == ["couch"]
        assert resolve("tv") == resolve("television") == ["tv"]

    def test_doorway_maps_to_door_a6_best_effort(self):
        # A6 = E: best-effort architectural mapping.
        assert resolve("doorway") == ["door"]

    def test_architectural_terms_with_no_coco_equivalent(self):
        # A6 acknowledges gap: these map to [] in v1, become detectable
        # after Sprint 5 fine-tune.
        assert resolve("wall") == []
        assert resolve("hallway") == []
        assert resolve("stairs") == []
        # is_known() should reflect the gap.
        assert is_known("wall") is False
        assert is_known("hallway") is False

    def test_empty_input(self):
        assert resolve("") == []
        assert resolve("   ") == []
        assert is_known("") is False


# ----------------------------------------------------------------------
# Substring fallback
# ----------------------------------------------------------------------

class TestSubstring:
    def test_substring_finds_alias_inside_phrase(self):
        # "blue chair" isn't a key, but "chair" is — substring match wins.
        result = resolve("blue chair")
        assert "chair" in result

    def test_substring_picks_longest_matching_alias(self):
        # "wine glass" must match before "glass" in substring fallback.
        assert resolve("the wine glass on the counter") == ["wine glass"]


# ----------------------------------------------------------------------
# Fuzzy fallback (requires rapidfuzz, installed by default)
# ----------------------------------------------------------------------

class TestFuzzy:
    @pytest.mark.parametrize("typo,expected_first", [
        ("doorwya", "door"),       # one transposition
        ("refigerator", "refrigerator"),  # missing letter
        ("couh", "couch"),         # truncated
        ("sofaa", "couch"),        # extra letter on a real alias
    ])
    def test_typo_recovers_via_fuzzy(self, typo, expected_first):
        result = resolve(typo)
        assert result and expected_first in result, (
            f"resolve({typo!r}) = {result}; expected to include {expected_first!r}"
        )

    def test_nonsense_word_returns_empty(self):
        # A4: when nothing matches at any level, return [].
        assert resolve("zxqwflubber") == []
        assert resolve("unicorn") == []


# ----------------------------------------------------------------------
# COCO class universe
# ----------------------------------------------------------------------

class TestCocoMembership:
    def test_every_mapped_class_is_valid_coco(self):
        """Every entry in LANDMARK_ALIASES values must be either a real COCO
        class or one of the architectural reserved labels (currently just
        'door', which we accept knowing it isn't in vanilla COCO yet)."""
        reserved_non_coco = {"door"}
        for alias, classes in LANDMARK_ALIASES.items():
            for c in classes:
                assert (c in COCO_CLASSES) or (c in reserved_non_coco), (
                    f"alias {alias!r} maps to unknown class {c!r}"
                )

    def test_coco_class_count(self):
        # YOLOv8 default is 80 classes.
        assert len(COCO_CLASSES) == 80


# ----------------------------------------------------------------------
# Snapshot test (G7 = C)
# ----------------------------------------------------------------------

SNAPSHOT_PATH = Path(__file__).parent / "snapshots" / "landmark_map.json"


class TestSnapshot:
    """Guards against accidental edits to the alias table.

    If you LEGITIMATELY change LANDMARK_ALIASES, regenerate the snapshot:

        python scripts/regen_snapshots.py

    Then commit the new snapshot alongside your alias edit.
    """

    def test_snapshot_file_exists(self):
        assert SNAPSHOT_PATH.exists(), (
            f"snapshot missing: {SNAPSHOT_PATH}. "
            f"Run `python scripts/regen_snapshots.py` to create it."
        )

    def test_table_matches_snapshot(self):
        if not SNAPSHOT_PATH.exists():
            pytest.skip("no snapshot yet (run regen_snapshots.py)")
        with SNAPSHOT_PATH.open("r", encoding="utf-8") as f:
            saved = json.load(f)
        current = snapshot_dict()
        assert current == saved, (
            "LANDMARK_ALIASES has changed. If intentional, regenerate the "
            "snapshot with `python scripts/regen_snapshots.py` and commit."
        )
