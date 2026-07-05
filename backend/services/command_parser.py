"""
Command parser.

Bridges raw Whisper transcriptions to FSM-ready intents:

    "find my cup"      ->  {"task_type": "object_allocation", "target": "cup", ...}
    "navigate kitchen" ->  {"task_type": "navigation",        "target": "kitchen", ...}
    "got it"           ->  {"task_type": "completion",        "target": "confirm", ...}
    "never mind"       ->  {"task_type": "completion",        "target": "cancel", ...}
    "what time is it"  ->  {"task_type": "unknown",            ...}

The "completion" task_type is recognized during an active task: a confirm
phrase ("got it", "found it") signals the user succeeded; a cancel phrase
("stop", "never mind") aborts. The FSM/audio layer decides whether the current
state actually accepts it - the parser only classifies the words.

Algorithm
---------
We try every possible (prefix, target) split of the input. For each split we
fuzzy-match the prefix to our list of supported templates and the target to
our list of supported nouns. The split with the best combined score wins.

This handles:
- Multi-word targets ("dining table", "living room")
- Mishearings ("cop" -> "cup", "fone" -> "phone", "microvave" -> "microwave")
- Slight phrasing variation ("find a cup" via "find" -> "find my")
- Trailing filler words ("find my cup, please" -> cup, dropping "please")

Thresholds are deliberately permissive (prefix >= 70, noun >= 60). False
positives at this layer are mostly fine because the worst case is that the
user hears a wrong confirmation TTS and tries again.

Designed to be standalone and importable - run from a Python REPL.
"""
from __future__ import annotations

import logging
import re
import string
from typing import Optional

from rapidfuzz import fuzz, process

log = logging.getLogger("lumen.parser")

# ---------- supported phrases ----------

# Object Allocation prefixes (the "X" target follows).
OBJECT_ALLOCATION_PREFIXES: tuple[str, ...] = (
    "find my",
    "find the",
    "find a",
    "find",
    "where is",
    "where's",
    "where is my",
    "where is the",
    "look for",
    "look for my",
    "look for the",
    "i need my",
    "i need the",
    "i need a",
    "i want my",
    "get me",
)

# Navigation prefixes.
NAVIGATION_PREFIXES: tuple[str, ...] = (
    "navigate to",
    "navigate to the",
    "take me to",
    "take me to the",
    "go to",
    "go to the",
    "lead me to",
    "lead me to the",
    "bring me to",
    "guide me to",
)

# Curated COCO-class noun list of supported objects (~20 for v1). These
# match YOLOv8 class names verbatim so Sprint 3's detection layer can use
# them as a key directly.
OBJECT_NOUNS: tuple[str, ...] = (
    "cup",
    "bottle",
    "chair",
    "couch",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
)

# Navigation destinations. Deliberately EXACTLY the rooms that
# services/navigation/goals.py has arrival-indicator tables for - a destination
# the parser accepts but the explorer can't recognise would produce two
# contradictory voice lines ("Navigating to the hallway." ... "I don't know how
# to find the hallway yet."). Keep the two lists in lockstep.
NAVIGATION_DESTINATIONS: tuple[str, ...] = (
    "kitchen",
    "bathroom",
    "bedroom",
    "living room",
    "dining room",
    "office",
)

# Synonyms / common mishearings -> canonical name. Applied before fuzzy match
# so we don't have to widen thresholds to cover them.
SYNONYMS: dict[str, str] = {
    # Object aliases
    "phone": "cell phone",
    "cellphone": "cell phone",
    "mobile": "cell phone",
    "telephone": "cell phone",
    "fone": "cell phone",
    "television": "tv",
    "fridge": "refrigerator",
    "sofa": "couch",
    "computer": "laptop",
    "remote control": "remote",
    "tablet": "laptop",
    "table": "dining table",
    # Room aliases
    "restroom": "bathroom",
    "washroom": "bathroom",
    "loo": "bathroom",
    "lounge": "living room",
    "study": "office",
}


# Completion / cancel phrases said *during* an active task. "confirm" means
# the user has the object (or arrived); "cancel" means abort the task.
COMPLETION_CONFIRM_PHRASES: tuple[str, ...] = (
    "got it",
    "i got it",
    "i have got it",
    "found it",
    "i found it",
    "i have it",
    "i have got the",
    "thats it",
    "that is it",
    "thank you",
    "thanks",
    "done",
    "all done",
)

COMPLETION_CANCEL_PHRASES: tuple[str, ...] = (
    "stop",
    "stop it",
    "stop searching",
    "cancel",
    "cancel it",
    "never mind",
    "nevermind",
    "forget it",
    "forget about it",
    "quit",
    "abort",
    "give up",
)

# Info queries said at any time; don't change task state, just answer.
#   describe = "what's around me / in front of me"
#   repeat   = "say that again"
INFO_DESCRIBE_PHRASES: tuple[str, ...] = (
    "describe",
    "describe it",
    "describe the scene",
    "what do you see",
    "what can you see",
    "what is around me",
    "what is around",
    "what is in front of me",
    "what is in front",
    "whats around me",
    "whats in front of me",
    "whats around",
    "whats in front",
    "look around",
    "tell me what you see",
)

INFO_REPEAT_PHRASES: tuple[str, ...] = (
    "repeat",
    "repeat that",
    "repeat it",
    "say that again",
    "say it again",
    "say again",
    "what did you say",
    "again please",
    "once more",
)


# ---------- thresholds ----------

PREFIX_SCORE_THRESHOLD = 70   # how close a prefix must be to a template
NOUN_SCORE_THRESHOLD = 60     # how close a target must be to a noun
COMBINED_THRESHOLD = 65       # average of prefix + noun must clear this

# Completion phrases are short, so we match the whole utterance (token-set, to
# tolerate filler like "okay ... thanks") and require a high score. We only
# attempt this on short utterances to avoid stealing real find/navigate
# commands.
COMPLETION_SCORE_THRESHOLD = 86
COMPLETION_MAX_WORDS = 5

# Info queries use the same matching approach with slightly wider utterance
# limits ("what is in front of me" is 6 words).
INFO_SCORE_THRESHOLD = 82
INFO_MAX_WORDS = 8


# ---------- main entry ----------

def parse(text: str) -> dict:
    """Parse a transcription into intent dict.

    Always returns a dict with the same keys, even on failure::

        {"task_type": "object_allocation", "target": "cup", "raw": "...",
         "needs_clarification": False, "score": 95.0}
        {"task_type": "navigation",        "target": "kitchen", ...}
        {"task_type": "unknown",            "target": None, ...,
         "needs_clarification": True}
    """
    raw = text or ""
    norm = _normalize(raw)
    words = norm.split()

    if not words:
        return _unknown(raw)

    # Completion / cancel commands are short and structurally unlike the
    # prefix+noun find/navigate commands, so we test them first.
    comp = _match_completion(norm, words)
    if comp is not None:
        target, score = comp
        return {
            "task_type": "completion",
            "target": target,  # "confirm" | "cancel"
            "raw": raw,
            "needs_clarification": False,
            "score": round(score, 1),
        }

    # Info queries: "describe" / "what's in front of me" / "repeat that".
    info = _match_info(norm, words)
    if info is not None:
        target, score = info
        return {
            "task_type": "info",
            "target": target,  # "describe" | "repeat"
            "raw": raw,
            "needs_clarification": False,
            "score": round(score, 1),
        }

    best: Optional[tuple[float, str, str]] = None  # (score, task_type, target)

    # Try every (prefix, target) sub-window. Both endpoints are allowed to
    # slide so we tolerate trailing filler ("please", "thanks", etc.) after
    # the target. The search is O(N^2) but utterances are short.
    for i in range(1, len(words)):
        for j in range(i + 1, len(words) + 1):
            prefix_str = " ".join(words[:i])
            target_str = " ".join(words[i:j])

            # Object allocation
            oa = _match_prefix_and_noun(
                prefix_str, target_str,
                OBJECT_ALLOCATION_PREFIXES, OBJECT_NOUNS,
            )
            if oa is not None:
                score, target = oa
                if best is None or score > best[0]:
                    best = (score, "object_allocation", target)

            # Navigation
            nav = _match_prefix_and_noun(
                prefix_str, target_str,
                NAVIGATION_PREFIXES, NAVIGATION_DESTINATIONS,
            )
            if nav is not None:
                score, target = nav
                if best is None or score > best[0]:
                    best = (score, "navigation", target)

    if best is not None:
        score, task_type, target = best
        return {
            "task_type": task_type,
            "target": target,
            "raw": raw,
            "needs_clarification": False,
            "score": round(score, 1),
        }

    log.info("Parse: no template matched %r", raw)
    return _unknown(raw)


def confirmation_phrase(intent: dict) -> str:
    """Generate the TTS confirmation string for an intent.

    Examples
    --------
    >>> confirmation_phrase({"task_type": "object_allocation", "target": "cup"})
    'Looking for your cup.'
    >>> confirmation_phrase({"task_type": "navigation", "target": "kitchen"})
    'Navigating to the kitchen.'
    >>> confirmation_phrase({"task_type": "unknown"})
    "I didn't catch that, please repeat."
    """
    ttype = intent.get("task_type")
    target = intent.get("target") or ""

    if ttype == "object_allocation":
        return f"Looking for your {target}."
    if ttype == "navigation":
        return f"Navigating to the {target}."
    if ttype == "completion":
        # The object/destination isn't known here (target is confirm/cancel),
        # so the audio layer normally voices a target-aware phrase instead.
        return "Okay." if target == "cancel" else "Got it."
    if ttype == "info":
        # audio_handler builds a target-specific phrase (scene description or
        # last-spoken repeat), so this is only a defensive default.
        return "Okay."
    return "I didn't catch that, please repeat."


# ---------- internals ----------

_PUNCT_RE = re.compile(rf"[{re.escape(string.punctuation)}]")


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace, apply synonyms."""
    t = text.lower()
    t = _PUNCT_RE.sub(" ", t)
    t = " ".join(t.split())
    # Apply synonym replacements at word boundaries.
    for src, dst in SYNONYMS.items():
        t = re.sub(rf"\b{re.escape(src)}\b", dst, t)
    return t


def _unknown(raw: str) -> dict:
    return {
        "task_type": "unknown",
        "target": None,
        "raw": raw,
        "needs_clarification": True,
        "score": 0.0,
    }


def _match_info(norm: str, words: list[str]) -> Optional[tuple[str, float]]:
    """Detect an info query - describe scene or repeat last phrase.

    Uses the same token-set / max-word-count guard as completion so long
    find/navigate commands can't accidentally match. Returns
    ``("describe" | "repeat", score)`` or None.
    """
    if not words or len(words) > INFO_MAX_WORDS:
        return None

    describe = process.extractOne(
        norm, INFO_DESCRIBE_PHRASES, scorer=fuzz.token_set_ratio,
        score_cutoff=INFO_SCORE_THRESHOLD,
    )
    repeat = process.extractOne(
        norm, INFO_REPEAT_PHRASES, scorer=fuzz.token_set_ratio,
        score_cutoff=INFO_SCORE_THRESHOLD,
    )
    d_score = describe[1] if describe else 0.0
    r_score = repeat[1] if repeat else 0.0
    if d_score == 0.0 and r_score == 0.0:
        return None
    if d_score >= r_score:
        return "describe", d_score
    return "repeat", r_score


def _match_completion(norm: str, words: list[str]) -> Optional[tuple[str, float]]:
    """Detect a completion (confirm/cancel) command.

    Returns ``("confirm" | "cancel", score)`` if the utterance matches a
    completion phrase above threshold, else None. Uses token-set ratio so
    "okay i got it thanks" still matches "got it", and bails on long
    utterances (those are find/navigate commands, not completions).
    """
    if not words or len(words) > COMPLETION_MAX_WORDS:
        return None

    confirm = process.extractOne(
        norm, COMPLETION_CONFIRM_PHRASES, scorer=fuzz.token_set_ratio,
        score_cutoff=COMPLETION_SCORE_THRESHOLD,
    )
    cancel = process.extractOne(
        norm, COMPLETION_CANCEL_PHRASES, scorer=fuzz.token_set_ratio,
        score_cutoff=COMPLETION_SCORE_THRESHOLD,
    )

    c_score = confirm[1] if confirm else 0.0
    x_score = cancel[1] if cancel else 0.0

    if c_score == 0.0 and x_score == 0.0:
        return None
    if x_score >= c_score:
        return "cancel", x_score
    return "confirm", c_score


def _match_prefix_and_noun(
    prefix_str: str,
    target_str: str,
    prefix_choices: tuple[str, ...],
    noun_choices: tuple[str, ...],
) -> Optional[tuple[float, str]]:
    """Return ``(combined_score, canonical_noun)`` if both fuzzy-match, else None."""
    p_match = process.extractOne(
        prefix_str, prefix_choices, scorer=fuzz.ratio,
        score_cutoff=PREFIX_SCORE_THRESHOLD,
    )
    if p_match is None:
        return None
    p_score = p_match[1]

    n_match = process.extractOne(
        target_str, noun_choices, scorer=fuzz.ratio,
        score_cutoff=NOUN_SCORE_THRESHOLD,
    )
    if n_match is None:
        return None
    n_score = n_match[1]

    combined = (p_score + n_score) / 2.0
    if combined < COMBINED_THRESHOLD:
        return None
    return combined, n_match[0]
