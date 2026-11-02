"""Collector factory. To add a source (e.g. X/Tweepy), subclass BaseCollector and register it here."""
from __future__ import annotations

from collectors.base import BaseCollector, RawPost


def get_collector(name: str, **kwargs) -> BaseCollector:
    name = name.lower()
    if name == "synthetic":
        from collectors.synthetic import SyntheticCollector
        return SyntheticCollector(**kwargs)
    if name == "csv":
        from collectors.csv_replay import CSVReplayCollector
        return CSVReplayCollector(**kwargs)
    if name == "reddit":
        from collectors.reddit import RedditCollector
        return RedditCollector(**kwargs)
    raise ValueError(f"Unknown collector '{name}' (choose synthetic | csv | reddit)")


__all__ = ["BaseCollector", "RawPost", "get_collector"]
