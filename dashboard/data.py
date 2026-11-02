from __future__ import annotations

from datetime import timedelta

from database.db import Database
from utils import utcnow


def load_dashboard_data(db: Database, brands, hours):
    """Return (posts, aspects, alerts) for the selected brands and look-back window."""
    since = None if hours is None else utcnow() - timedelta(hours=hours)
    return db.load_posts(brands, since), db.load_aspects(brands, since), db.load_alerts(brands, since)
