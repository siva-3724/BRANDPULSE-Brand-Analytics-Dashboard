import random
from datetime import timedelta

import pytest

from analytics.alerts import detect_spikes, scan_alerts
from analytics.insights import generate_report
from analytics.metrics import eligible, sentiment_summary, window_metrics
from collectors.base import RawPost
from collectors.synthetic import SyntheticCollector, generate_history
from database.db import Database
from models.sentiment import SentimentAnalyzer
from pipeline.processor import Pipeline, fetch_with_retry, match_brands, refresh_analytics
from utils import utcnow

import pandas as pd


@pytest.fixture(scope="module")
def analyzer():
    return SentimentAnalyzer("vader")


@pytest.fixture()
def db(tmp_path):
    return Database(tmp_path / "t.db")


def test_match_brands():
    assert match_brands("Loving the Nova X1 camera") == ["Nova X1"]
    assert set(match_brands("Apex Pro vs Lumen S5?")) == {"Apex Pro", "Lumen S5"}
    assert match_brands("nothing relevant") == []


def test_pipeline_inserts_and_dedupes(db, analyzer):
    pipe = Pipeline(db, analyzer)
    posts = SyntheticCollector(seed=1).fetch(150)
    s1 = pipe.process(posts)
    assert s1["inserted"] > 100
    s2 = pipe.process(posts)                       # same IDs again -> DB-level dedupe
    assert s2["inserted"] == 0 and s2["duplicates"] > 0
    df = db.load_posts()
    assert set(df["sentiment"]) <= {"positive", "negative", "neutral"}
    assert (df["model_name"] == "vader+domain-lexicon").all()


def test_edge_cases(db, analyzer):
    pipe = Pipeline(db, analyzer)
    now = utcnow()
    stats = pipe.process([
        RawPost("t", "1", "   ", now),                                   # empty
        RawPost("t", "2", "weather is nice", now),                       # no brand
        RawPost("t", "3", "Nova X1 FREE GIVEAWAY click here", now),      # spam
        RawPost("t", "4", "நோவா Nova X1 மிகவும் நன்றாக உள்ளது", now),     # unsupported language
        RawPost("t", "5", "Nova X1 camera is amazing", now),
        RawPost("t", "6", "Nova X1 camera is amazing", now),             # exact duplicate text
    ])
    assert stats["empty"] == 1 and stats["no_brand"] == 1 and stats["spam"] == 1
    assert stats["unscored"] == 1 and stats["batch_duplicates"] == 1 and stats["inserted"] == 3
    assert len(eligible(db.load_posts())) == 1


def test_mixed_post_gets_aspects(db, analyzer):
    Pipeline(db, analyzer).process([RawPost("t", "m", "Nova X1 display looks fantastic, but the battery is disappointing.", utcnow())])
    asp = db.load_aspects().set_index("topic")["sentiment"].to_dict()
    assert asp["Display"] == "positive" and asp["Battery"] == "negative"
    assert db.load_posts().iloc[0]["is_mixed"] == 1


def test_spike_detected_end_to_end(db, analyzer):
    pipe = Pipeline(db, analyzer)
    pipe.process(generate_history(hours=24, posts_per_hour=100, seed=3, spike=True))
    refresh_analytics(db)
    alerts = db.load_alerts()
    neg = alerts[(alerts["rule"] == "negative_spike") & (alerts["brand"] == "Nova X1")]
    assert not neg.empty
    assert "Battery" in neg.iloc[0]["details"]
    assert refresh_analytics(db) == 0               # idempotent: no duplicate alerts


def test_no_alerts_without_spike(db, analyzer):
    Pipeline(db, analyzer).process(generate_history(hours=24, posts_per_hour=100, seed=4, spike=False))
    refresh_analytics(db)
    assert db.load_alerts().empty


def test_tiny_sample_never_alerts():
    ws = pd.date_range("2025-01-01", periods=10, freq="15min")
    m = pd.DataFrame({"window_start": ws, "brand": "X", "total_mentions": [10] * 9 + [4],
                      "positive_count": [7] * 9 + [1], "negative_count": [3] * 9 + [3],
                      "neutral_count": [0] * 10})
    m["negative_share"] = m.negative_count / m.total_mentions
    m["mention_velocity"] = m.total_mentions / 15
    assert detect_spikes(m, "X") == []              # 3/4 negative but below min sample


def test_metrics_report_and_purge(db, analyzer):
    Pipeline(db, analyzer).process(generate_history(hours=6, posts_per_hour=80, seed=5))
    posts = db.load_posts()
    m = window_metrics(posts)
    assert m["total_mentions"].sum() == len(eligible(posts))
    s = sentiment_summary(posts)
    assert abs(sum(s["pct"].values()) - 1) < 1e-9
    report = generate_report(posts, db.load_aspects(), db.load_alerts(), "All brands", 3)
    assert "Overall sentiment" in report and "hypotheses" in report
    assert db.purge_older_than(0) > 0 and db.count_posts() == 0
    assert db.load_aspects().empty                  # cascade delete


def test_fetch_with_retry(monkeypatch):
    monkeypatch.setattr("pipeline.processor.time.sleep", lambda s: None)

    class Flaky:
        calls = 0

        def fetch(self, limit):
            Flaky.calls += 1
            if Flaky.calls < 3:
                raise RuntimeError("rate limited")
            return ["ok"]

    assert fetch_with_retry(Flaky(), 5) == ["ok"] and Flaky.calls == 3
