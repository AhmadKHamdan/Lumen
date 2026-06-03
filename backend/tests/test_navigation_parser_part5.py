"""Tests for the Part 5 parser additions: is_yes, is_no, is_done_phrase."""

import pytest
from navigation.parser import is_yes, is_no, is_done_phrase


class TestIsYes:
    @pytest.mark.parametrize("text", [
        "yes", "Yes", "YES", "yeah", "yep", "yup", "sure",
        "yes I am", "yes I'm here", "correct", "that's right",
        "confirmed", "affirmative",
    ])
    def test_recognized(self, text):
        assert is_yes(text) is True

    @pytest.mark.parametrize("text", [
        "no", "the chair", "I don't know", "", "yesterday",
    ])
    def test_not_recognized(self, text):
        assert is_yes(text) is False


class TestIsNo:
    @pytest.mark.parametrize("text", [
        "no", "No", "NO!", "nope", "not yet",
        "no I'm not", "negative",
    ])
    def test_recognized(self, text):
        assert is_no(text) is True

    @pytest.mark.parametrize("text", [
        "yes", "the chair", "", "north",
    ])
    def test_not_recognized(self, text):
        assert is_no(text) is False


class TestIsDonePhrase:
    @pytest.mark.parametrize("text", [
        "done", "Done", "finished", "all done",
        "that's all", "we're done", "I'm done",
        "finish", "complete", "end",
    ])
    def test_recognized(self, text):
        assert is_done_phrase(text) is True

    @pytest.mark.parametrize("text", [
        "the chair", "yes", "", "doorway",
    ])
    def test_not_recognized(self, text):
        assert is_done_phrase(text) is False
