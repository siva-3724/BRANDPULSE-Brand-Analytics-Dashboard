"""Small shared helpers. All timestamps are naive UTC."""
from __future__ import annotations

from datetime import datetime, timezone

TS_FMT = "%Y-%m-%d %H:%M:%S"


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def fmt_ts(dt: datetime) -> str:
    return dt.strftime(TS_FMT)