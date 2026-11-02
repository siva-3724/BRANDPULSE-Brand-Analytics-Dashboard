"""Deterministic, template-based business report. Recommendations are hypotheses, not proof."""
from __future__ import annotations

from datetime import timedelta

import pandas as pd

from analytics.metrics import eligible, sentiment_summary, top_hashtags, top_keywords, topic_sentiment
from utils import fmt_ts, utcnow

HINTS = {
    "Battery": "a recent software update, a specific hardware batch, or background-app drain",
    "Heating": "charging hardware, a firmware change, or a specific usage pattern (gaming, video calls)",
    "Performance": "a recent update, memory management or app compatibility",
    "Delivery": "a logistics partner, a region-specific delay or packaging damage",
    "Customer Service": "support response times, warranty policy clarity or staffing during launch",
    "Price": "perceived value versus competitors or regional pricing",
    "Camera": "image-processing tuning or low-light behaviour",
    "Display": "panel batch, brightness tuning or flicker at low brightness",
    "Availability": "stock allocation by region or pre-order fulfilment",
    "Design": "materials, finish or colour options",
}


def _pct(x: float) -> str:
    return f"{x:.0%}"


def generate_report(posts: pd.DataFrame, aspects: pd.DataFrame, alerts: pd.DataFrame,
                    brand: str = "All brands", hours: int = 24) -> str:
    el = eligible(posts)
    if el.empty:
        return f"# BrandPulse report - {brand}\n\nNo eligible posts available."
    anchor = el["created_at"].max()
    cur_start = anchor - timedelta(hours=hours)
    cur = el[el["created_at"] > cur_start]
    prev = el[(el["created_at"] <= cur_start) & (el["created_at"] > cur_start - timedelta(hours=hours))]
    s, p = sentiment_summary(cur), sentiment_summary(prev)

    L = [f"# BrandPulse report - {brand}",
         f"_Generated {fmt_ts(utcnow())} UTC. Period: last {hours}h up to {fmt_ts(anchor.to_pydatetime())}._", "",
         "## 1. Overall sentiment",
         f"- **Mentions:** {s['total']:,}",
         f"- **Positive / Negative / Neutral:** {_pct(s['pct']['positive'])} / {_pct(s['pct']['negative'])} / {_pct(s['pct']['neutral'])}"]
    if p["total"]:
        d = (s["pct"]["negative"] - p["pct"]["negative"]) * 100
        L.append(f"- **Change vs previous {hours}h:** negative share {d:+.1f} pts "
                 f"(mentions {s['total'] - p['total']:+,}).")
    else:
        L.append("- **Change vs previous period:** not enough earlier data.")

    asp = aspects[aspects["created_at"] > cur_start] if not aspects.empty else aspects
    ts = topic_sentiment(asp)
    L += ["", "## 2. Most discussed product features"]
    if ts.empty:
        L.append("No feature mentions detected.")
    else:
        for r in ts.head(5).itertuples():
            L.append(f"- **{r.topic}**: {int(r.total)} mentions "
                     f"({int(r.positive)} positive, {int(r.negative)} negative; net {r.net_score:+.2f})")
        solid = ts[ts["total"] >= 5]
        pos = solid[solid["net_score"] > 0].sort_values("net_score", ascending=False).head(3)
        neg = solid[solid["net_score"] < 0].sort_values("net_score").head(3)
        L += ["", "## 3. Positive themes"] + ([f"- **{r.topic}** (net {r.net_score:+.2f}, {int(r.total)} mentions)" for r in pos.itertuples()] or ["- None above the minimum sample size."])
        L += ["", "## 4. Top complaints"]
        if neg.empty:
            L.append("- No feature has net-negative sentiment above the minimum sample size.")
        for r in neg.itertuples():
            L.append(f"- **{r.topic}**: {int(r.negative)} negative mentions ({int(r.negative) / r.total:.0%} of its mentions)")
        worst = cur[cur["sentiment"] == "negative"].drop_duplicates("text").nsmallest(3, "sentiment_score")
        if not worst.empty:
            L += ["", "Representative negative posts:"]
            for r in worst.itertuples():
                link = f" ([link]({r.permalink}))" if r.permalink else ""
                L.append(f'> "{r.text}"{link}')

    tags, kws = top_hashtags(cur, 5), top_keywords(cur, 8)
    L += ["", "## 5. Trending hashtags and keywords",
          "- Hashtags: " + (", ".join(f"#{h} ({n})" for h, n in zip(tags.hashtag, tags.mentions)) or "none"),
          "- Keywords: " + (", ".join(f"{k} ({n})" for k, n in zip(kws.keyword, kws.mentions)) or "none")]

    if not alerts.empty:
        recent = alerts[alerts["window_start"] > cur_start]
        L += ["", "## 6. Alerts in period"]
        L += [f"- [{a.severity.upper()}] {a.brand} - {a.rule} at {fmt_ts(a.window_start.to_pydatetime())}: {a.details}"
              for a in recent.head(10).itertuples()] or ["- None."]

    if not ts.empty:
        L += ["", "## 7. Suggested issues to investigate (hypotheses, not conclusions)"]
        for r in ts[ts["net_score"] < 0].head(3).itertuples():
            L.append(f"- **{r.topic}**: check whether complaints relate to {HINTS.get(r.topic, 'a recent change')}. "
                     "Validate with support tickets and telemetry before acting.")
    return "\n".join(L)
