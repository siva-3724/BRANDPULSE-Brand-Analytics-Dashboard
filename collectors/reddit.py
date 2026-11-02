"""Reddit collector using the official API via PRAW.

Respect Reddit's API terms, rate limits and your credentials' scope. Create an app at
https://www.reddit.com/prefs/apps and set REDDIT_* variables in .env.
"""
from __future__ import annotations

from datetime import datetime, timezone

import config
from collectors.base import BaseCollector, RawPost


class RedditCollector(BaseCollector):
    name = "reddit"

    def __init__(self, subreddit: str = "all"):
        try:
            import praw
        except ImportError as exc:
            raise RuntimeError("Install PRAW: pip install praw") from exc
        if not config.REDDIT_CLIENT_ID or not config.REDDIT_CLIENT_SECRET:
            raise RuntimeError("Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env")
        self.reddit = praw.Reddit(
            client_id=config.REDDIT_CLIENT_ID,
            client_secret=config.REDDIT_CLIENT_SECRET,
            user_agent=config.REDDIT_USER_AGENT,
        )
        self.subreddit = subreddit

    def fetch(self, limit: int = 50) -> list[RawPost]:
        per_brand = max(1, limit // len(config.BRANDS))
        posts = []
        for kws in config.BRANDS.values():
            query = kws[0]
            for s in self.reddit.subreddit(self.subreddit).search(query, sort="new", limit=per_brand):
                created = datetime.fromtimestamp(s.created_utc, timezone.utc).replace(tzinfo=None)
                posts.append(RawPost(
                    "reddit", s.id, f"{s.title}. {s.selftext}"[:1000], created, query,
                    "https://www.reddit.com" + s.permalink,
                ))
        return posts
