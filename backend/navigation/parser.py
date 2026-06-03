"""Navigation-specific text parsing.

Per decision 6, parsing for the waypoint follow-up turn lives in its own
function rather than extending the top-level command_parser. Omar's
command_parser.parse() owns intent detection; parse_waypoint() owns the
"clean a landmark-phrase from a transcription" job inside an already-
active Navigation task.

Other decisions covered here:
- 5: Both raw_text and normalized_text are produced (raw kept for logs /
     the report, normalized for fuzzy landmark mapping in Part 2).
- 8: Multi-waypoint utterances ("doorway then turn left") are split on
     connectors and returned as separate candidates.
- 11: Voice cancellation phrases recognised during NavigationActive.
"""

from __future__ import annotations

import re

# Filler patterns stripped from the start of a waypoint phrase.
# Order matters: longer patterns first so "go through the" is matched
# before "go through".
_FILLERS: tuple[str, ...] = (
    "go through the",
    "go through",
    "go down the",
    "go down",
    "go to the",
    "go to",
    "go past the",
    "go past",
    "take the",
    "take a",
    "head to the",
    "head to",
    "head towards",
    "head toward",
    "walk to the",
    "walk to",
    "walk through",
    "walk past",
    "through the",
    "down the",
    "past the",
    "to the",
)

# Connectors that split one utterance into sequential waypoints (decision 8).
# Whitespace-padded so "kitchenette" is not split on the bare letters "and".
_SPLIT_PATTERN = re.compile(r"\s*,\s*|\s+then\s+|\s+and then\s+|\s+and\s+", flags=re.I)

# Voice cancellation phrases (decision 11). Match if the whole utterance
# is one of these, OR starts with one.
_CANCEL_PHRASES: tuple[str, ...] = (
    "stop", "cancel", "never mind", "nevermind", "quit", "exit",
    "stop navigation", "cancel navigation", "abort",
    "end navigation", "stop the navigation",
)

# Arrival phrases for Part 5. Kept in this module so all nav-text rules
# live together; the Part 1 manager only uses is_cancellation_phrase.
_ARRIVAL_PHRASES: tuple[str, ...] = (
    "i'm here", "im here", "i am here",
    "got it", "i made it", "i arrived", "arrived",
    "i'm at the", "im at the", "i am at the",
    "here we are",
)

# Yes/no recognizers for Part 5 arrival-confirmation dialog.
_YES_PHRASES: tuple[str, ...] = (
    "yes", "yeah", "yep", "yup", "sure", "correct", "confirmed",
    "i am", "yes i am", "yes i'm here", "thats right", "that's right",
    "affirmative",
)
_NO_PHRASES: tuple[str, ...] = (
    "no", "nope", "not yet", "no i'm not", "no im not", "no i am not",
    "not really", "negative", "incorrect",
)

# "Done" phrases for the completion/what's-next dialog.
_DONE_PHRASES: tuple[str, ...] = (
    "done", "finished", "all done", "that's all", "thats all",
    "we're done", "were done", "i'm done", "im done", "finish",
    "complete", "end",
)


def _strip_punct(text: str) -> str:
    """Lowercase, replace non-word chars with spaces, collapse whitespace."""
    t = re.sub(r"[^\w\s']+", " ", text.lower())
    t = re.sub(r"\s+", " ", t).strip()
    return t


def normalize_phrase(phrase: str) -> str:
    """Clean a waypoint phrase: lowercase, strip punct/articles/leading fillers."""
    p = _strip_punct(phrase)
    if not p:
        return ""

    # Remove leading filler patterns, iterating until none match.
    changed = True
    while changed:
        changed = False
        for f in _FILLERS:
            if p.startswith(f + " "):
                p = p[len(f) + 1:].strip()
                changed = True
                break
            if p == f:
                p = ""
                changed = True
                break

    # Strip a remaining leading article (handles both "the door" and bare "the").
    p = re.sub(r"^(the|a|an)\b\s*", "", p).strip()
    # If only an article was left, treat as empty.
    if p in {"the", "a", "an"}:
        p = ""
    return p


def parse_waypoint(text: str) -> list[dict]:
    """Parse a user response into one or more waypoint candidates.

    Returns a list of dicts: [{"raw_text": str, "normalized_text": str}, ...]

    - Empty input -> [].
    - All-filler input (e.g. "to the") -> [].
    - Multi-waypoint (decision 8) -> one entry per chunk, in order.

    The caller (NavigationTaskManager) treats an empty return as the
    re-prompt trigger per decision 12.
    """
    if not text or not text.strip():
        return []

    chunks = _SPLIT_PATTERN.split(text.strip())
    results: list[dict] = []
    for chunk in chunks:
        raw = chunk.strip()
        if not raw:
            continue
        # Strip any leading connector word that survived a comma-only split
        # ("doorway, then the chair" -> chunks include "then the chair").
        cleaned = re.sub(r"^(then|and)\s+", "", raw, flags=re.I).strip()
        if not cleaned:
            continue
        normalized = normalize_phrase(cleaned)
        if not normalized:
            # Pure filler — skip this chunk but keep any others.
            continue
        results.append({"raw_text": raw, "normalized_text": normalized})
    return results


def is_cancellation_phrase(text: str) -> bool:
    """True if the utterance is (or starts with) a voice cancel phrase."""
    if not text:
        return False
    t = _strip_punct(text)
    if not t:
        return False
    return any(t == p or t.startswith(p + " ") for p in _CANCEL_PHRASES)


def is_arrival_phrase(text: str) -> bool:
    """True if the utterance signals the user has arrived (used in Part 5)."""
    if not text:
        return False
    t = _strip_punct(text)
    if not t:
        return False
    return any(t == p or t.startswith(p + " ") or t.startswith(p) for p in _ARRIVAL_PHRASES)


def is_yes(text: str) -> bool:
    """True if the utterance is an affirmative response (Part 5 confirmation)."""
    if not text:
        return False
    t = _strip_punct(text)
    if not t:
        return False
    # Whole-utterance match or starts-with (so "yes I am" still counts).
    return any(t == p or t.startswith(p + " ") for p in _YES_PHRASES)


def is_no(text: str) -> bool:
    """True if the utterance is a negative response (Part 5 confirmation)."""
    if not text:
        return False
    t = _strip_punct(text)
    if not t:
        return False
    return any(t == p or t.startswith(p + " ") for p in _NO_PHRASES)


def is_done_phrase(text: str) -> bool:
    """True if the utterance signals 'I'm finished with the whole task' (Part 5)."""
    if not text:
        return False
    t = _strip_punct(text)
    if not t:
        return False
    return any(t == p or t.startswith(p + " ") for p in _DONE_PHRASES)
