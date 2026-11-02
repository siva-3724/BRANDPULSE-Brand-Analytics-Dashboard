"""Lightweight lexicon-based emotion tagging (swap for a transformer if desired)."""
from __future__ import annotations

import re

LEXICON = {
    "Joy": ["love", "amazing", "great", "awesome", "excellent", "fantastic", "gorgeous", "impressed",
            "happy", "stunning", "😍", "😊", "😁", "❤️", "👍"],
    "Frustration": ["disappointing", "disappointed", "annoying", "frustrating", "slow", "laggy",
                    "drains", "sluggish", "delayed", "useless", "👎"],
    "Anger": ["terrible", "worst", "awful", "hate", "scam", "ridiculous", "unacceptable", "😡", "🤬"],
    "Sadness": ["sad", "regret", "unfortunately", "let down", "😢", "😞"],
    "Surprise": ["surprised", "unexpected", "shocked", "didn't expect", "😮", "😲"],
}
_PATTERNS = {
    emo: re.compile(r"(?<!\w)(?:%s)(?!\w)" % "|".join(re.escape(w) for w in words), re.I)
    for emo, words in LEXICON.items()
}


def detect_emotion(text: str) -> str:
    scores = {emo: len(p.findall(text)) for emo, p in _PATTERNS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "None"
