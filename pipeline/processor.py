"""Core pipeline: clean -> match brand -> sentiment/emotion/aspects -> store (+ analytics refresh)."""
from __future__ import annotations

import logging
import re
import time
from collections import Counter
from datetime import timedelta

import config
from analytics.alerts import scan_alerts
from analytics.metrics import window_metrics
from collectors.base import BaseCollector, RawPost
from database.db import Database
from models.emotion import detect_emotion
from models.sentiment import SentimentAnalyzer
from models.topics import aspect_clauses, extract_topics
from preprocessing.cleaning import clean_text, detect_language, extract_hashtags, is_spam, text_hash
from utils import fmt_ts, utcnow

log = logging.getLogger(__name__)

_BRAND_PATTERNS = {
    brand: re.compile(r"(?<!\w)(?:%s)(?!\w)" % "|".join(re.escape(k) for k in sorted(kws, key=len, reverse=True)), re.I)
    for brand, kws in config.BRANDS.items()
}


def match_brands(text: str) -> list[str]:
    return [b for b, p in _BRAND_PATTERNS.items() if p.search(text or "")]


def fetch_with_retry(collector: BaseCollector, limit: int, retries: int = 3, backoff: float = 2.0) -> list[RawPost]:
    """Call collector.fetch with exponential backoff for transient API/rate-limit errors."""
    for attempt in range(1, retries + 1):
        try:
            return collector.fetch(limit)
        except Exception as exc:
            if attempt == retries:
                raise
            wait = backoff ** attempt
            log.warning("fetch failed (%s); retry %d/%d in %.0fs", exc, attempt, retries, wait)
            time.sleep(wait)
    return []


class Pipeline:
    def __init__(self, db: Database | None = None, analyzer: SentimentAnalyzer | None = None):
        self.db = db or Database()
        self.analyzer = analyzer or SentimentAnalyzer()

    def process(self, raw_posts: list[RawPost]) -> Counter:
        stats: Counter = Counter(received=len(raw_posts))
        collected = fmt_ts(utcnow())
        prepared, seen = [], set()

        for rp in raw_posts:
            text = clean_text(rp.text)
            if not text:
                stats["empty"] += 1
                continue
            brands = match_brands(text) or match_brands(rp.keyword)
            if not brands:
                stats["no_brand"] += 1
                continue
            lang, spam, digest = detect_language(text), is_spam(rp.text), text_hash(text)
            for brand in brands:
                if (brand, digest) in seen:
                    stats["batch_duplicates"] += 1
                    continue
                seen.add((brand, digest))
                prepared.append({"rp": rp, "text": text, "brand": brand, "lang": lang, "spam": spam})

        # --- batch inference: whole posts, then feature-bearing clauses ---
        scorable = [p for p in prepared if p["lang"] in config.SUPPORTED_LANGUAGES]
        for p, pred in zip(scorable, self.analyzer.predict_batch([p["text"] for p in scorable])):
            p["pred"] = pred
        clause_owner, clause_texts = [], []
        for p in scorable:
            p["aspects"] = []
            for clause, topics in aspect_clauses(p["text"]):
                clause_owner.append((p, topics))
                clause_texts.append(clause)
        for (p, topics), cp in zip(clause_owner, self.analyzer.predict_batch(clause_texts)):
            p["aspects"] += [{"topic": t, "sentiment": cp.label, "score": cp.score} for t in topics]

        # --- persist ---
        with self.db.session() as conn:
            for p in prepared:
                rp, pred, aspects = p["rp"], p.get("pred"), p.get("aspects", [])
                if aspects:
                    topic = max(aspects, key=lambda a: abs(a["score"]))["topic"]
                else:
                    topics = extract_topics(p["text"])
                    topic = topics[0] if topics else "General"
                labels = {a["sentiment"] for a in aspects}
                rec = {
                    "source": rp.source, "source_post_id": rp.source_post_id, "text": p["text"],
                    "created_at": fmt_ts(rp.created_at), "collected_at": collected, "brand": p["brand"],
                    "keyword": rp.keyword, "language": p["lang"],
                    "sentiment": pred.label if pred else "unscored",
                    "sentiment_score": pred.score if pred else None,
                    "confidence": pred.confidence if pred else None,
                    "emotion": detect_emotion(p["text"]) if pred else None,
                    "topic": topic, "hashtags": ",".join(extract_hashtags(p["text"])),
                    "is_spam": int(p["spam"]), "is_mixed": int({"positive", "negative"} <= labels),
                    "model_name": self.analyzer.model_name if pred else None, "permalink": rp.permalink,
                }
                if self.db.insert_post(conn, rec, aspects):
                    stats["inserted"] += 1
                    stats["spam"] += rec["is_spam"]
                    stats["unscored"] += 0 if pred else 1
                else:
                    stats["duplicates"] += 1
        return stats


def refresh_analytics(db: Database, hours: int = 72) -> int:
    """Recompute window metrics and (idempotently) store any new alerts. Returns # new alerts."""
    posts = db.load_posts(since=utcnow() - timedelta(hours=hours))
    if posts.empty:
        return 0
    metrics = window_metrics(posts)
    db.upsert_metrics(metrics)
    return db.save_alerts(scan_alerts(posts, metrics))
