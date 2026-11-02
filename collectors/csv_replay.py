"""Replay a CSV file as if it were a live stream (offline demo / deterministic tests)."""
from __future__ import annotations

import pandas as pd

import config
from collectors.base import BaseCollector, RawPost
from utils import utcnow


class CSVReplayCollector(BaseCollector):
    """CSV needs a `text` column; optional: source, source_post_id, created_at, keyword, permalink.

    restamp=True gives each replayed post a *current* timestamp so the dashboard looks live.
    """
    name = "csv"

    def __init__(self, path=None, restamp: bool = True):
        self.df = pd.read_csv(path or config.SAMPLE_CSV)
        self.restamp = restamp
        self.pos = 0

    def fetch(self, limit: int = 50) -> list[RawPost]:
        chunk = self.df.iloc[self.pos : self.pos + limit]
        self.pos += len(chunk)
        now = utcnow()
        posts = []
        for i, row in chunk.iterrows():
            created = now if (self.restamp or "created_at" not in row) else pd.to_datetime(row["created_at"]).to_pydatetime()
            posts.append(RawPost(
                source=str(row.get("source", "csv")),
                source_post_id=str(row.get("source_post_id", f"csv-{i}")),
                text=str(row["text"]),
                created_at=created,
                keyword=str(row.get("keyword", "") if pd.notna(row.get("keyword", "")) else ""),
                permalink=str(row.get("permalink", "") if pd.notna(row.get("permalink", "")) else ""),
            ))
        return posts
