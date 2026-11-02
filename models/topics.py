"""Rule-based product-feature (aspect) tagging and clause-level aspect extraction."""
from __future__ import annotations

import re

import config


def _compile(words) -> re.Pattern:
    alts = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
    return re.compile(rf"(?<!\w)(?:{alts})(?!\w)", re.I)


_PATTERNS = {topic: _compile(kws) for topic, kws in config.TOPIC_KEYWORDS.items()}
_CLAUSE_SPLIT = re.compile(r"(?<=[.!?;])\s+|\s*,\s*|\s+(?:but|however|although|though|yet)\s+", re.I)


def extract_topics(text: str) -> list[str]:
    return [topic for topic, pat in _PATTERNS.items() if pat.search(text)]


def split_clauses(text: str) -> list[str]:
    return [c.strip() for c in _CLAUSE_SPLIT.split(text) if c and c.strip()]


def aspect_clauses(text: str) -> list[tuple[str, list[str]]]:
    """Clauses that mention at least one product feature, with the features found."""
    result = []
    for clause in split_clauses(text):
        topics = extract_topics(clause)
        if topics:
            result.append((clause, topics))
    return result
