"""Post-processing: safety regex, length normalisation, language check."""

from __future__ import annotations

import re
import unicodedata

# Kill obvious prompt-injection / self-reflection artifacts the model might spit out.
_SELF_REFERENCE_RE = re.compile(
    r"(?i)(as an ai|as a language model|i am (just )?an ai|i cannot provide)"
)
_TRAILING_INSTRUCTION_RE = re.compile(r"\b(Ekibuuzo|Swali|Asɛmmisa|ጥያቄ|Question)\s*:", re.IGNORECASE)
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def clean_output(text: str, language: str) -> str:
    """Normalise and safety-filter a generated answer.

    Steps:
        1. Unicode NFKC (consistent composition of Amharic, Latin diacritics).
        2. Remove model self-reference boilerplate (rare with Aya but observed).
        3. Cut at the first next-turn instruction marker.
        4. Collapse runs of whitespace.
        5. Strip.
    """
    if not text:
        return ""

    t = unicodedata.normalize("NFKC", text)

    # Truncate at the first stray "Question:" marker the model added.
    m = _TRAILING_INSTRUCTION_RE.search(t)
    if m:
        t = t[: m.start()]

    t = _SELF_REFERENCE_RE.sub("", t)
    t = _MULTI_SPACE_RE.sub(" ", t)
    t = _MULTI_NEWLINE_RE.sub("\n\n", t)
    return t.strip()


def assert_language_ok(
    text: str, expected_language: str, min_confidence: float = 0.4
) -> tuple[bool, str]:
    """True if the output looks like the expected language.

    Returns (is_ok, detected_language). Used to flag catastrophic
    language-drift failures at inference time.
    """
    from afro_health_qa.data.language_id import detect_lang

    detected, conf = detect_lang(text)
    if detected == expected_language:
        return True, detected
    # Allow low-confidence misfires when the text is very short.
    if conf < min_confidence and len(text) < 30:
        return True, detected
    return False, detected
