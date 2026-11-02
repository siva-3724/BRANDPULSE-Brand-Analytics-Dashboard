"""Synthetic social-post generator for demos, tests and offline development.

Brands have different strengths/weaknesses so dashboards show meaningful contrasts,
and an optional "battery complaint spike" exercises the alerting system.

CLI:  python -m collectors.synthetic --n 1500 --hours 72 --out data/sample_posts.csv
"""
from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta

from collectors.base import BaseCollector, RawPost
from utils import fmt_ts, utcnow

CORES = {
    ("Battery", "positive"): ["battery easily lasts a full day", "battery life is excellent", "battery lasts for ages, impressed"],
    ("Battery", "negative"): ["battery drains too quickly", "battery life is disappointing", "battery drains overnight, terrible", "battery is awful, dies by noon"],
    ("Camera", "positive"): ["camera quality is amazing", "camera takes stunning photos", "camera is fantastic in low light"],
    ("Camera", "negative"): ["camera is disappointing indoors", "camera photos look blurry, disappointing", "camera is terrible at night"],
    ("Display", "positive"): ["display is gorgeous and bright", "screen is crisp and beautiful", "display looks fantastic"],
    ("Display", "negative"): ["display is dim outdoors, disappointing", "screen has annoying flickering", "display is awful in sunlight"],
    ("Performance", "positive"): ["performance is excellent and smooth", "apps open fast, great performance", "performance is snappy"],
    ("Performance", "negative"): ["performance is laggy and slow", "apps crash constantly, frustrating", "feels sluggish, disappointing performance"],
    ("Price", "positive"): ["price is a great value", "price is fair for what you get", "is totally worth the price"],
    ("Price", "negative"): ["price is too expensive", "is overpriced, not worth it", "price is ridiculous, way too pricey"],
    ("Design", "positive"): ["design is sleek and beautiful", "build feels premium, love the design", "colours look gorgeous"],
    ("Design", "negative"): ["design feels cheap and boring", "build feels flimsy, disappointing"],
    ("Heating", "positive"): ["stays cool even while gaming"],
    ("Heating", "negative"): ["gets hot while charging", "overheats during video calls", "overheats badly, terrible", "gets hot too quickly, worrying"],
    ("Customer Service", "positive"): ["customer service was helpful and quick", "support team was friendly and helpful"],
    ("Customer Service", "negative"): ["customer service is terrible", "support ignored my emails, awful service", "warranty support is useless, disappointing"],
    ("Delivery", "positive"): ["delivery was fast, arrived early", "shipping was quick and smooth"],
    ("Delivery", "negative"): ["delivery was delayed again, frustrating", "shipping took forever, terrible", "courier damaged the box, awful"],
    ("Availability", "positive"): ["is finally back in stock, great news"],
    ("Availability", "negative"): ["is sold out everywhere, annoying", "availability is terrible in my city"],
}
NEUTRAL_CORES = [
    "is available in three colours", "launches in India next week", "pre-order opens on Friday",
    "has a 50MP main camera", "battery capacity is 5000mAh", "price starts at 599 dollars",
    "uses the latest processor", "unboxing video is up on the channel", "launch event starts at 6pm today",
    "review embargo lifts tomorrow", "comes in black and silver",
]
FRAMES = ["{b} {core}", "{core_cap} on the {b}", "Day {d} with the {b}: {core}", "Honestly, the {b} {core}"]
TAGS = {"Nova X1": ["NovaX1", "NovaLaunch"], "Apex Pro": ["ApexPro", "ApexLaunch"], "Lumen S5": ["LumenS5", "LumenLaunch"]}
PROFILES = {  # probability that a post about a topic is positive
    "Nova X1": {"Camera": .85, "Display": .8, "Battery": .3, "_default": .6},
    "Apex Pro": {"Performance": .85, "Price": .25, "Design": .7, "_default": .55},
    "Lumen S5": {"Design": .85, "Delivery": .2, "Camera": .6, "_default": .55},
}
BRAND_WEIGHTS = {"Nova X1": .45, "Apex Pro": .30, "Lumen S5": .25}
ENDINGS = {
    "positive": [".", "!", "!!", " 😍", " 👍", "!! 😍"],
    "negative": [".", "...", "!", " 😡", " 👎", " 😞", "!!"],
    "neutral": [".", ".", "."],
}
DETAILS = ["so far", "after a week of use", "since the latest update", "compared to my old phone",
           "in daily use", "since launch day", "at the moment", "this week", "after two weeks"]
TOPICS = sorted({t for t, _ in CORES})
DEFAULT_SPIKE = {"brand": "Nova X1", "topic": "Battery", "n": 150, "minutes": 30, "ago_minutes": 105}


def _compose(rng, brand, core, label) -> str:
    frame = rng.choice(FRAMES)
    if rng.random() < 0.7:  # add variety so templates rarely repeat verbatim
        core = f"{core} {rng.choice(DETAILS)}"
    text = frame.format(b=brand, core=core, core_cap=core[0].upper() + core[1:], d=rng.randint(1, 30))
    text += rng.choice(ENDINGS[label])
    if rng.random() < 0.25:
        text += " #" + rng.choice(TAGS[brand])
    return text


def _record(rng, brand, ts, text, label) -> dict:
    return {
        "source": "synthetic", "source_post_id": f"syn-{rng.getrandbits(48):012x}", "text": text,
        "created_at": ts, "keyword": brand.lower(), "permalink": "", "label": label,
    }


def generate_records(n, start: datetime, end: datetime, rng: random.Random, spike: dict | None = None):
    """Return dict records with a ground-truth `label` (positive/negative/neutral/mixed/spam)."""
    span = (end - start).total_seconds()
    brands, weights = zip(*BRAND_WEIGHTS.items())
    out = []
    for _ in range(n):
        brand = rng.choices(brands, weights)[0]
        ts = start + timedelta(seconds=rng.random() * span)
        r = rng.random()
        if r < 0.20:
            out.append(_record(rng, brand, ts, _compose(rng, brand, rng.choice(NEUTRAL_CORES), "neutral"), "neutral"))
        elif r < 0.23:
            txt = f"FREE GIVEAWAY!!! Click here to win a {brand} http://spam.example/w{rng.randint(1, 999)} #giveaway #free"
            out.append(_record(rng, brand, ts, txt, "spam"))
        elif r < 0.30:  # mixed opinion: positive about one feature, negative about another
            a, b = rng.sample(TOPICS, 2)
            pos = rng.choice(CORES[(a, "positive")]) if (a, "positive") in CORES else None
            neg = rng.choice(CORES[(b, "negative")])
            if pos:
                out.append(_record(rng, brand, ts, f"{brand} {pos}, but {neg}.", "mixed"))
        else:
            topic = rng.choice(TOPICS)
            p_pos = PROFILES[brand].get(topic, PROFILES[brand]["_default"])
            label = "positive" if rng.random() < p_pos else "negative"
            out.append(_record(rng, brand, ts, _compose(rng, brand, rng.choice(CORES[(topic, label)]), label), label))
    if spike:
        s_start = spike.get("start", end - timedelta(minutes=spike["ago_minutes"]))
        s_end = s_start + timedelta(minutes=spike["minutes"])
        sec = (s_end - s_start).total_seconds()
        for _ in range(spike["n"]):
            ts = s_start + timedelta(seconds=rng.random() * sec)
            core = rng.choice(CORES[(spike["topic"], "negative")])
            out.append(_record(rng, spike["brand"], ts, _compose(rng, spike["brand"], core, "negative"), "negative"))
    return out


def _to_raw(records) -> list[RawPost]:
    return [RawPost(r["source"], r["source_post_id"], r["text"], r["created_at"], r["keyword"], r["permalink"])
            for r in records]


def generate_history(hours=48, posts_per_hour=120, seed=7, spike=True) -> list[RawPost]:
    """Backfill `hours` of history ending now, optionally with a battery-complaint spike."""
    rng = random.Random(seed)
    end = utcnow()
    recs = generate_records(int(hours * posts_per_hour), end - timedelta(hours=hours), end, rng,
                            DEFAULT_SPIKE if spike else None)
    recs.sort(key=lambda r: r["created_at"])
    return _to_raw(recs)


class SyntheticCollector(BaseCollector):
    """Emits fresh synthetic posts on each call - a stand-in for a live stream."""
    name = "synthetic"

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self._pending: list[dict] = []

    def inject_spike(self, brand="Nova X1", topic="Battery", n=120) -> None:
        """Queue a burst of negative posts that the next fetch() will emit."""
        self._pending.append({"brand": brand, "topic": topic, "n": n, "minutes": 10, "ago_minutes": 10})

    def fetch(self, limit: int = 50) -> list[RawPost]:
        now = utcnow()
        recs = generate_records(limit, now - timedelta(seconds=45), now, self.rng)
        for spike in self._pending:
            recs += generate_records(0, now, now, self.rng, {**spike, "start": now - timedelta(minutes=spike["minutes"])})
        self._pending.clear()
        return _to_raw(recs)


if __name__ == "__main__":
    import pandas as pd

    ap = argparse.ArgumentParser(description="Generate a synthetic sample CSV")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--hours", type=int, default=72)
    ap.add_argument("--out", default="data/sample_posts.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    end = utcnow()
    rows = generate_records(args.n, end - timedelta(hours=args.hours), end, random.Random(args.seed), DEFAULT_SPIKE)
    df = pd.DataFrame(rows).sort_values("created_at")
    df["created_at"] = df["created_at"].map(fmt_ts)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")
