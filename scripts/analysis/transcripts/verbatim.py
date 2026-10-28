"""Check that a quote appears verbatim in a text, up to typographic normalization."""

from __future__ import annotations

import re

_TYPOGRAPHY = {"‘": "'", "’": "'", "“": '"', "”": '"', "—": "-", "–": "-", "…": "..."}
_ELISION = re.compile(r"\[\s*(?:\.\.\.|…)\s*\]")


def normalize(s: str) -> str:
    for a, b in _TYPOGRAPHY.items():
        s = s.replace(a, b)
    s = re.sub(r"\*\*|(?<!\\)\*", "", s)
    return re.sub(r"\s+", " ", s).strip()


def is_verbatim(quote: str, text: str) -> bool:
    """True if every fragment of ``quote`` between ``[...]`` elisions occurs in ``text``."""
    norm_text = normalize(text)
    fragments = [f.strip() for f in _ELISION.split(normalize(quote)) if f.strip()]
    return bool(fragments) and all(f in norm_text for f in fragments)
