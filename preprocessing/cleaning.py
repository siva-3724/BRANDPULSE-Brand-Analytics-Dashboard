"""Text cleaning, language detection and spam heuristics.

Design rule: keep emojis, punctuation and negations - they carry sentiment.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
WS_RE = re.compile(r"\s+")
REPEAT_RE = re.compile(r"(.)\1{3,}")  # "sooooo" -> "sooo"
PROMO_RE = re.compile(
    r"giveaway|click here|dm (me )?for|follow (me )?for|promo code|"
    r"free (iphone|phone)|\bwin a\b|limited offer|\bbuy now\b",
    re.I,
)


def clean_text(text) -> str:
    """Normalize a raw post for analysis (URLs/mentions removed, emojis kept)."""
    if not isinstance(text, str):
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub("@user", t)
    t = REPEAT_RE.sub(r"\1\1\1", t)
    return WS_RE.sub(" ", t).strip()


def extract_hashtags(text: str) -> list[str]:
    return [h.lower() for h in HASHTAG_RE.findall(text or "")]


def text_hash(text: str) -> str:
    norm = re.sub(r"\W+", " ", text.lower()).strip()
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def is_spam(raw_text: str) -> bool:
    """Flag (not delete) likely spam / promotional posts."""
    raw_text = raw_text or ""
    return (
        bool(PROMO_RE.search(raw_text))
        or len(URL_RE.findall(raw_text)) >= 2
        or len(HASHTAG_RE.findall(raw_text)) >= 6
    )


def detect_language(text: str) -> str:
    """Script-based detection for Tamil/Hindi, optional `langdetect` for Latin text."""
    ta = sum(1 for ch in text if 0x0B80 <= ord(ch) <= 0x0BFF)
    hi = sum(1 for ch in text if 0x0900 <= ord(ch) <= 0x097F)
    if ta >= 3:
        return "ta"
    if hi >= 3:
        return "hi"
    if len(text) >= 25:
        try:
            from langdetect import DetectorFactory, detect

            DetectorFactory.seed = 0
            return detect(text)
        except Exception:
            pass
    return "en"
