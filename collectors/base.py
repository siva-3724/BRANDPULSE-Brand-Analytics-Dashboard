"""Collector interface. Any source (API, CSV replay, synthetic) just implements `fetch`."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


@dataclass
class RawPost:
    source: str
    source_post_id: str
    text: str
    created_at: datetime          # naive UTC
    keyword: str = ""
    permalink: str = ""


class BaseCollector(ABC):
    name = "base"

    @abstractmethod
    def fetch(self, limit: int = 50) -> list[RawPost]:
        """Return up to `limit` new posts. May raise on transient API errors (caller retries)."""
