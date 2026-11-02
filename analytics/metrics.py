"""Aggregations: windowed sentiment metrics, keywords, hashtags, feature breakdowns."""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd

import config

SENTIMENTS = ("positive", "negative", "neutral")
METRIC_COLS = ["window_start", "brand", "total_mentions", "positive_count", "negative_count",
               "neutral_count", "negative_share", "mention_velocity"]

_BRAND_WORDS = {w for kws in config.BRANDS.values() for k in kws for w in k.split()}
STOPWORDS = set("""the and for that this with have has had was were are but not you your our they them their its it's
i'm i've don't can't just very really too than then there here about after again all also any because been before
being both each few get got how into more most much now off once only other out over own same some such what when
where which who why will would could should from day days on in is at as be by or an of to so if do does did me my
we he she him her his one two three too still even way like going gone let makes make made want""".split())


def eligible(df: pd.DataFrame) -> pd.DataFrame:
    """Posts that count toward metrics: scored and not flagged as spam."""
    if df.empty:
        return df
    return df[(df["is_spam"] == 0) & df["sentiment"].isin(SENTIMENTS)]


def window_metrics(df: pd.DataFrame, minutes: int | None = None) -> pd.DataFrame:
    """Per-brand sentiment counts per time window (gaps filled with zero-volume windows)."""
    minutes = minutes or config.WINDOW_MINUTES
    d = eligible(df)
    if d.empty:
        return pd.DataFrame(columns=METRIC_COLS)
    d = d.assign(window_start=d["created_at"].dt.floor(f"{minutes}min"))
    frames = []
    for brand, g in d.groupby("brand"):
        counts = g.groupby(["window_start", "sentiment"]).size().unstack(fill_value=0)
        counts = counts.reindex(columns=list(SENTIMENTS), fill_value=0)
        idx = pd.date_range(counts.index.min(), counts.index.max(), freq=f"{minutes}min")
        counts = counts.reindex(idx, fill_value=0)
        out = counts.rename_axis("window_start").reset_index()
        out.columns = ["window_start", "positive_count", "negative_count", "neutral_count"]
        out.insert(1, "brand", brand)
        out["total_mentions"] = out[["positive_count", "negative_count", "neutral_count"]].sum(axis=1)
        out["negative_share"] = out["negative_count"] / out["total_mentions"].where(out["total_mentions"] > 0)
        out["mention_velocity"] = out["total_mentions"] / minutes  # posts per minute
        frames.append(out)
    return pd.concat(frames, ignore_index=True)[METRIC_COLS]


def sentiment_summary(df: pd.DataFrame) -> dict:
    d = eligible(df)
    total = len(d)
    counts = {s: int((d["sentiment"] == s).sum()) if total else 0 for s in SENTIMENTS}
    pct = {s: (counts[s] / total if total else 0.0) for s in SENTIMENTS}
    return {"total": total, "counts": counts, "pct": pct,
            "avg_score": float(d["sentiment_score"].mean()) if total else 0.0}


def top_hashtags(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    c = Counter()
    for tags in eligible(df)["hashtags"].dropna() if not df.empty else []:
        c.update(t for t in tags.split(",") if t)
    return pd.DataFrame(c.most_common(n), columns=["hashtag", "mentions"])


def top_keywords(df: pd.DataFrame, n: int = 15, sentiment: str | None = None) -> pd.DataFrame:
    d = eligible(df)
    if sentiment and not d.empty:
        d = d[d["sentiment"] == sentiment]
    c = Counter()
    for text in d["text"] if not d.empty else []:
        for tok in re.findall(r"[a-z][a-z']{2,}", text.lower()):
            if tok not in STOPWORDS and tok not in _BRAND_WORDS:
                c[tok] += 1
    return pd.DataFrame(c.most_common(n), columns=["keyword", "mentions"])


def topic_sentiment(aspects: pd.DataFrame) -> pd.DataFrame:
    """Feature-level sentiment: counts per topic and a net score in [-1, 1]."""
    cols = ["topic", "positive", "negative", "neutral", "total", "net_score"]
    if aspects.empty:
        return pd.DataFrame(columns=cols)
    t = aspects.groupby(["topic", "sentiment"]).size().unstack(fill_value=0)
    t = t.reindex(columns=list(SENTIMENTS), fill_value=0)
    t["total"] = t.sum(axis=1)
    t["net_score"] = (t["positive"] - t["negative"]) / t["total"].replace(0, np.nan)
    return t.reset_index().sort_values("total", ascending=False)[cols]
