"""Unit tests for navigation.obstacle_map."""

import pytest

from navigation.obstacle_map import (
    OBSTACLE_CLASSES,
    PERSON_CLASSES,
    LARGE_OBSTACLES,
    SMALL_OBSTACLES,
    SMALL_OBJECT_MIN_BBOX_AREA_RATIO,
    is_obstacle,
    is_person,
    is_small_object,
    obstacle_priority,
    passes_size_gate,
    bbox_area_ratio,
    display_name,
)


class TestMembership:
    @pytest.mark.parametrize("cls", ["person", "chair", "couch", "refrigerator",
                                      "cup", "bottle", "cat", "dog"])
    def test_known_obstacles_are_obstacles(self, cls):
        assert is_obstacle(cls) is True

    @pytest.mark.parametrize("cls", ["airplane", "kite", "frisbee", "tie",
                                      "stop sign", "fire hydrant"])
    def test_outdoor_or_irrelevant_not_obstacles(self, cls):
        assert is_obstacle(cls) is False

    def test_is_person(self):
        assert is_person("person") is True
        assert is_person("chair") is False
        assert is_person("dog") is False  # pet, not person

    def test_pets_are_obstacles_but_not_persons(self):
        assert is_obstacle("dog") is True
        assert is_obstacle("cat") is True
        assert is_person("dog") is False
        assert is_person("cat") is False

    def test_no_overlap_between_categories(self):
        # Sanity: a class shouldn't appear in both PERSON and LARGE.
        assert PERSON_CLASSES.isdisjoint(LARGE_OBSTACLES)
        assert PERSON_CLASSES.isdisjoint(SMALL_OBSTACLES)


class TestSizeGate:
    def test_large_obstacle_passes_regardless_of_bbox(self):
        # No size gate for non-small classes — even a tiny bbox passes.
        assert passes_size_gate("chair", [0, 0, 10, 10]) is True

    def test_small_object_at_distance_fails_size_gate(self):
        # Tiny bbox for a cup -> filter out (foot-strike isn't a risk).
        assert passes_size_gate("cup", [0, 0, 50, 50]) is False

    def test_small_object_close_up_passes_size_gate(self):
        # Large bbox -> close enough to be a toe-strike hazard.
        # 8% of 640x640 = 32768 px. 200x200 = 40000 px > threshold.
        assert passes_size_gate("cup", [0, 0, 200, 200]) is True

    def test_threshold_is_8_percent(self):
        assert SMALL_OBJECT_MIN_BBOX_AREA_RATIO == 0.08

    def test_bbox_area_ratio_compute(self):
        assert bbox_area_ratio([0, 0, 320, 320]) == pytest.approx(0.25)
        assert bbox_area_ratio([]) == 0.0
        assert bbox_area_ratio(None) == 0.0


class TestPriority:
    def test_person_highest(self):
        assert obstacle_priority("person") > obstacle_priority("chair")
        assert obstacle_priority("person") > obstacle_priority("dog")

    def test_pet_above_furniture(self):
        assert obstacle_priority("dog") > obstacle_priority("chair")

    def test_large_above_small(self):
        # Large obstacles have priority 1; small have 0.
        assert obstacle_priority("chair") > obstacle_priority("cup")

    def test_unknown_class_lowest(self):
        assert obstacle_priority("anything_random") == 0


class TestDisplayName:
    def test_overrides(self):
        assert display_name("tv") == "TV"
        assert display_name("potted plant") == "plant"

    def test_passthrough(self):
        assert display_name("chair") == "chair"
        assert display_name("refrigerator") == "refrigerator"
