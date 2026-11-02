"""SQLite persistence layer (swap the connection for PostgreSQL via SQLAlchemy when scaling)."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

import pandas as pd

import config
from utils import fmt_ts, utcnow

SCHEMA = """
CREATE TABLE IF NOT EXISTS social_posts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    source_post_id TEXT NOT NULL,
    text TEXT NOT NULL,
    created_at TEXT NOT NULL,
    collected_at TEXT NOT NULL,
    brand TEXT NOT NULL,
    keyword TEXT,
    language TEXT,
    sentiment TEXT,              -- positive | negative | neutral | unscored
    sentiment_score REAL,
    confidence REAL,
    emotion TEXT,
    topic TEXT,
    hashtags TEXT,
    is_spam INTEGER DEFAULT 0,
    is_mixed INTEGER DEFAULT 0,
    model_name TEXT,
    permalink TEXT,
    -- A post can mention several brands, so brand is part of the key.
    UNIQUE (source, source_post_id, brand)
);
CREATE INDEX IF NOT EXISTS idx_posts_created ON social_posts (created_at);
CREATE INDEX IF NOT EXISTS idx_posts_brand ON social_posts (brand);

CREATE TABLE IF NOT EXISTS post_aspects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id INTEGER NOT NULL REFERENCES social_posts(id) ON DELETE CASCADE,
    brand TEXT NOT NULL,
    topic TEXT NOT NULL,
    sentiment TEXT NOT NULL,
    score REAL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_aspects_created ON post_aspects (created_at);

CREATE TABLE IF NOT EXISTS sentiment_metrics (
    window_start TEXT NOT NULL,
    brand TEXT NOT NULL,
    total_mentions INTEGER,
    positive_count INTEGER,
    negative_count INTEGER,
    neutral_count INTEGER,
    negative_share REAL,
    mention_velocity REAL,
    PRIMARY KEY (window_start, brand)
);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    window_start TEXT NOT NULL,
    brand TEXT NOT NULL,
    rule TEXT NOT NULL,
    observed REAL,
    baseline REAL,
    severity TEXT,
    details TEXT,
    acknowledged INTEGER DEFAULT 0,
    UNIQUE (brand, window_start, rule)
);
"""

POST_COLS = [
    "source", "source_post_id", "text", "created_at", "collected_at", "brand", "keyword",
    "language", "sentiment", "sentiment_score", "confidence", "emotion", "topic", "hashtags",
    "is_spam", "is_mixed", "model_name", "permalink",
]


class Database:
    def __init__(self, path=None):
        self.path = str(path or config.DB_PATH)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self.session() as conn:
            conn.execute("PRAGMA journal_mode=WAL")  # dashboard reads while the pipeline writes
            conn.executescript(SCHEMA)

    @contextmanager
    def session(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ---- writes ---------------------------------------------------------
    def insert_post(self, conn, rec: dict, aspects: list[dict]) -> bool:
        """Insert one post (+ aspects). Returns False if it already existed."""
        cols = ",".join(POST_COLS)
        marks = ",".join(["?"] * len(POST_COLS))
        cur = conn.execute(
            f"INSERT OR IGNORE INTO social_posts ({cols}) VALUES ({marks})",
            [rec.get(c) for c in POST_COLS],
        )
        if cur.rowcount != 1:
            return False
        conn.executemany(
            "INSERT INTO post_aspects (post_id, brand, topic, sentiment, score, created_at) "
            "VALUES (?,?,?,?,?,?)",
            [(cur.lastrowid, rec["brand"], a["topic"], a["sentiment"], a["score"], rec["created_at"])
             for a in aspects],
        )
        return True

    def upsert_metrics(self, metrics: pd.DataFrame) -> None:
        if metrics.empty:
            return
        rows = [
            (fmt_ts(r.window_start), r.brand, int(r.total_mentions), int(r.positive_count),
             int(r.negative_count), int(r.neutral_count),
             None if pd.isna(r.negative_share) else float(r.negative_share), float(r.mention_velocity))
            for r in metrics.itertuples()
        ]
        with self.session() as conn:
            conn.executemany("INSERT OR REPLACE INTO sentiment_metrics VALUES (?,?,?,?,?,?,?,?)", rows)

    def save_alerts(self, alerts: list[dict]) -> int:
        """Persist alerts; duplicates (same brand/window/rule) are ignored. Returns new count."""
        new = 0
        now = fmt_ts(utcnow())
        with self.session() as conn:
            for a in alerts:
                cur = conn.execute(
                    "INSERT OR IGNORE INTO alerts (created_at, window_start, brand, rule, observed, "
                    "baseline, severity, details) VALUES (?,?,?,?,?,?,?,?)",
                    (now, fmt_ts(a["window_start"]), a["brand"], a["rule"], a["observed"],
                     a["baseline"], a["severity"], a["details"]),
                )
                new += max(cur.rowcount, 0)
        return new

    def acknowledge_alert(self, alert_id: int) -> None:
        with self.session() as conn:
            conn.execute("UPDATE alerts SET acknowledged=1 WHERE id=?", (alert_id,))

    def purge_older_than(self, days: int) -> int:
        """Retention policy: delete posts (and their aspects) older than `days`."""
        cutoff = fmt_ts(utcnow() - timedelta(days=days))
        with self.session() as conn:
            return conn.execute("DELETE FROM social_posts WHERE created_at < ?", (cutoff,)).rowcount

    def reset(self) -> None:
        with self.session() as conn:
            for t in ("post_aspects", "social_posts", "sentiment_metrics", "alerts"):
                conn.execute(f"DELETE FROM {t}")

    # ---- reads ----------------------------------------------------------
    def _query(self, table, brands, since, order, date_cols) -> pd.DataFrame:
        q, params = f"SELECT * FROM {table} WHERE 1=1", []
        if brands:
            q += f" AND brand IN ({','.join(['?'] * len(brands))})"
            params += list(brands)
        if since is not None:
            q += " AND created_at >= ?" if table != "alerts" else " AND window_start >= ?"
            params.append(fmt_ts(since))
        q += f" ORDER BY {order}"
        with self.session() as conn:
            df = pd.read_sql_query(q, conn, params=params)
        for c in date_cols:
            df[c] = pd.to_datetime(df[c])
        return df

    def load_posts(self, brands=None, since=None) -> pd.DataFrame:
        return self._query("social_posts", brands, since, "created_at", ["created_at", "collected_at"])

    def load_aspects(self, brands=None, since=None) -> pd.DataFrame:
        return self._query("post_aspects", brands, since, "created_at", ["created_at"])

    def load_alerts(self, brands=None, since=None) -> pd.DataFrame:
        return self._query("alerts", brands, since, "window_start DESC", ["created_at", "window_start"])

    def count_posts(self) -> int:
        with self.session() as conn:
            return conn.execute("SELECT COUNT(*) FROM social_posts").fetchone()[0]
