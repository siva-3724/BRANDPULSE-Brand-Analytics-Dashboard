"""Ingestion worker.

Examples
  python -m pipeline.run_stream --source synthetic --backfill 48 --iterations 5 --interval 2
  python -m pipeline.run_stream --source csv --batch 25 --interval 3
  python -m pipeline.run_stream --source reddit --interval 120
"""
from __future__ import annotations

import argparse
import logging
import time

from collectors import get_collector
from collectors.synthetic import generate_history
from database.db import Database
from pipeline.processor import Pipeline, fetch_with_retry, refresh_analytics

log = logging.getLogger("brandpulse")


def run(source: str, interval: float, batch: int, iterations: int | None, backfill: int,
        retention_days: int | None, db_path=None) -> None:
    db = Database(db_path)
    pipe = Pipeline(db)
    log.info("Sentiment model: %s", pipe.analyzer.model_name)

    if backfill:
        log.info("Backfilling %dh of synthetic history ...", backfill)
        log.info("Backfill stats: %s", dict(pipe.process(generate_history(hours=backfill))))
        log.info("New alerts: %d", refresh_analytics(db))

    collector = get_collector(source)
    n = 0
    while iterations is None or n < iterations:
        n += 1
        try:
            raw = fetch_with_retry(collector, batch)
            stats = pipe.process(raw)
            new_alerts = refresh_analytics(db)
            log.info("batch %d | %s | new alerts: %d", n, dict(stats), new_alerts)
            if not raw and source == "csv":
                log.info("CSV replay finished.")
                break
        except Exception:
            log.exception("batch %d failed; continuing", n)  # failed-record tracking via logs
        if retention_days:
            db.purge_older_than(retention_days)
        if iterations is None or n < iterations:
            time.sleep(interval)


def main() -> None:
    ap = argparse.ArgumentParser(description="BrandPulse ingestion worker")
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "csv", "reddit"])
    ap.add_argument("--interval", type=float, default=5.0, help="seconds between polls")
    ap.add_argument("--batch", type=int, default=10, help="posts per poll")
    ap.add_argument("--iterations", type=int, default=None, help="stop after N polls (default: run forever)")
    ap.add_argument("--backfill", type=int, default=0, help="hours of synthetic history to load first")
    ap.add_argument("--retention-days", type=int, default=None, help="delete posts older than N days")
    ap.add_argument("--db", default=None, help="override database path")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(a.source, a.interval, a.batch, a.iterations, a.backfill, a.retention_days, a.db)


if __name__ == "__main__":
    main()
