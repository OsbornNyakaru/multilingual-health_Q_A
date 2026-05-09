"""Language identification for the four target languages.

fastText's lid.176 is fast, free, and covers Swahili, Luganda, and Amharic well.
Akan/Twi is not in lid.176 and is routinely misclassified as English or a
generic Niger-Congo label — we add a character-set + bigram heuristic fallback
to catch it.

The target label space (ISO-639-3):
    lug — Luganda (Latin)
    swa — Kiswahili (Latin)
    aka — Akan / Twi (Latin with special characters ɛ ɔ)
    amh — Amharic (Ge'ez / Ethiopic script)
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

TARGET_LANGS = ("lug", "swa", "aka", "amh")

# Map fastText's ISO-639-1 labels to our ISO-639-3 target space.
_FASTTEXT_TO_TARGET = {
    "sw": "swa",
    "lg": "lug",
    "am": "amh",
    # Akan — not in lid.176; fallthrough to heuristic.
}

# Unicode blocks we care about.
_ETHIOPIC_RE = re.compile(r"[\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF\uAB00-\uAB2F]")
# Akan / Twi signature characters (open-e, open-o, with combining tone marks).
_AKAN_CHAR_RE = re.compile(r"[ɛɔƐƆ]")

# A handful of common-word patterns to disambiguate Latin-script Bantu.
_LUGANDA_MARKERS = re.compile(
    r"\b(n'|na|ne|omu|aba|eby|ky|eky|ko|mu|olw)\b|"
    r"\bekibuuzo\b|\beky'okuddamu\b",
    re.IGNORECASE,
)
_SWAHILI_MARKERS = re.compile(
    r"\b(na|ya|wa|ni|kwa|kuhusu|ninaweza|habari|mimi|swali|jibu|afya)\b",
    re.IGNORECASE,
)
_AKAN_MARKERS = re.compile(
    r"\b(wo|ne|ma|me|yɛ|mmuaeɛ|asɛmmisa|adwuma)\b",
    re.IGNORECASE,
)


@lru_cache(maxsize=1)
def _load_fasttext_model(model_path: str | None = None):
    """Load fastText lid.176. Downloaded once to data/external/lid.176.bin."""
    import fasttext

    path = model_path or "data/external/lid.176.bin"
    if not Path(path).exists():
        raise FileNotFoundError(
            f"fastText lid.176 model not found at {path}. "
            "Run: bash scripts/download_data.sh (downloads lid.176.bin), "
            "or manually: curl -L -o data/external/lid.176.bin "
            "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
        )
    return fasttext.load_model(path)


def detect_lang(
    text: str,
    model_path: str | None = None,
    use_heuristics: bool = True,
) -> tuple[str, float]:
    """Detect the language of ``text``.

    Strategy:
        1. Script check — if the text contains Ethiopic codepoints, return amh.
        2. Akan-character check — if ɛ/ɔ present with Akan bigrams, return aka.
        3. fastText lid.176 — handles swa/lug reliably.
        4. Word-marker fallback — disambiguates Latin-script when fastText is
           uncertain (score < 0.5) or returns a non-target label.

    Args:
        text: input string.
        model_path: override the lid.176.bin location.
        use_heuristics: if False, skip the heuristic steps and return fastText
            raw output (useful for ablations).

    Returns:
        (ISO-639-3 code, confidence in [0.0, 1.0]). Returns ("unk", 0.0) if
        nothing matches.
    """
    if not text or not text.strip():
        return ("unk", 0.0)

    stripped = text.strip()

    if use_heuristics:
        if _ETHIOPIC_RE.search(stripped):
            return ("amh", 1.0)
        if _AKAN_CHAR_RE.search(stripped) and _AKAN_MARKERS.search(stripped):
            return ("aka", 0.9)

    # fastText expects no newlines.
    cleaned = stripped.replace("\n", " ")
    try:
        model = _load_fasttext_model(model_path)
        labels, scores = model.predict(cleaned, k=3)
    except FileNotFoundError:
        # Model not installed yet — heuristic-only path.
        return _heuristic_only(stripped)

    for label, score in zip(labels, scores):
        iso2 = label.replace("__label__", "")
        mapped = _FASTTEXT_TO_TARGET.get(iso2)
        if mapped is not None and score >= 0.3:
            return (mapped, float(score))

    if use_heuristics:
        fallback = _heuristic_only(stripped)
        if fallback[0] != "unk":
            return fallback

    # Nothing confident. Return the top fastText label prefixed with "ft:" so
    # the caller can tell it came from the wire, not our target space.
    top_label = labels[0].replace("__label__", "") if labels else "unk"
    return (f"ft:{top_label}", float(scores[0]) if len(scores) else 0.0)


def _heuristic_only(stripped: str) -> tuple[str, float]:
    """Latin-script heuristics only. Used when fastText is unavailable or unsure."""
    if _ETHIOPIC_RE.search(stripped):
        return ("amh", 1.0)
    if _AKAN_CHAR_RE.search(stripped) or _AKAN_MARKERS.search(stripped):
        return ("aka", 0.7)
    # Tie-break between lug and swa on marker counts.
    lug_hits = len(_LUGANDA_MARKERS.findall(stripped))
    swa_hits = len(_SWAHILI_MARKERS.findall(stripped))
    if lug_hits > swa_hits and lug_hits > 0:
        return ("lug", min(0.6 + 0.05 * lug_hits, 0.9))
    if swa_hits > 0:
        return ("swa", min(0.6 + 0.05 * swa_hits, 0.9))
    return ("unk", 0.0)


def is_target_lang(code: str) -> bool:
    """True if ``code`` is one of the four competition targets."""
    return code in TARGET_LANGS
