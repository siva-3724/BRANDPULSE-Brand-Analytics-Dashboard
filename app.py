"""BrandPulse AI - Streamlit dashboard.   Run:  streamlit run app.py"""
from __future__ import annotations

import time

import pandas as pd
import streamlit as st

import config
from analytics.insights import generate_report
from analytics.metrics import (SENTIMENTS, eligible, sentiment_summary, top_hashtags, top_keywords,
                               topic_sentiment, window_metrics)
from collectors.synthetic import SyntheticCollector, generate_history
from dashboard import charts
from dashboard.data import load_dashboard_data
from database.db import Database
from pipeline.processor import Pipeline, refresh_analytics

st.set_page_config(page_title="BrandPulse AI", page_icon="📡", layout="wide")


@st.cache_resource
def get_db() -> Database:
    return Database()


@st.cache_resource
def get_pipeline() -> Pipeline:
    return Pipeline(get_db())  # loads the sentiment model once


@st.cache_resource
def get_simulator() -> SyntheticCollector:
    return SyntheticCollector()


db, pipe, sim = get_db(), get_pipeline(), get_simulator()
RANGES = {"Last 1 hour": 1, "Last 6 hours": 6, "Last 24 hours": 24, "Last 48 hours": 48, "Last 7 days": 168, "All time": None}

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("📡 BrandPulse AI")
    st.caption(f"Model: `{pipe.analyzer.model_name}`")
    brands = st.multiselect("Brands", list(config.BRANDS), default=list(config.BRANDS))
    range_label = st.selectbox("Time range", list(RANGES), index=3)
    window = st.select_slider("Timeline window (minutes)", [5, 15, 30, 60], value=config.WINDOW_MINUTES)
    st.divider()
    live = st.toggle("Auto-refresh", value=False)
    interval = st.slider("Refresh every (seconds)", 3, 60, 10)
    simulate = st.toggle("Simulate live stream while refreshing", value=True,
                         help="Each refresh ingests a small batch of synthetic posts.")
    with st.expander("Demo data controls"):
        if st.button("Load 48h demo history", use_container_width=True):
            with st.spinner("Processing ~5,800 posts ..."):
                stats = pipe.process(generate_history(hours=48))
                new_alerts = refresh_analytics(db)
            st.success(f"Inserted {stats['inserted']:,} posts, {new_alerts} new alerts")
        if st.button("Ingest one live batch", use_container_width=True):
            stats = pipe.process(sim.fetch(20))
            refresh_analytics(db)
            st.success(f"Inserted {stats['inserted']} posts")
        if st.button("Inject battery-complaint spike", use_container_width=True):
            sim.inject_spike("Nova X1", "Battery", 120)
            stats = pipe.process(sim.fetch(20))
            st.success(f"New alerts: {refresh_analytics(db)}")
        if st.button("Reset database", type="secondary", use_container_width=True):
            db.reset()
            st.warning("Database cleared.")
if live and simulate:
    pipe.process(sim.fetch(5))
    refresh_analytics(db)

posts, aspects, alerts = load_dashboard_data(db, brands, RANGES[range_label])


def render() -> None:
    el = eligible(posts)
    summ = sentiment_summary(el)
    open_alerts = alerts[alerts["acknowledged"] == 0] if not alerts.empty else alerts

    tabs = st.tabs(["Overview", "Timeline", "Topics & hashtags", "Feedback explorer",
                    "Brand comparison", "Alert centre", "Report"])

    # ---- Overview
    with tabs[0]:
        high = open_alerts[open_alerts["severity"] == "high"] if not open_alerts.empty else open_alerts
        if not high.empty:
            a = high.iloc[0]
            st.error(f"🚨 **{a['brand']}** - {a['rule'].replace('_', ' ')}: {a['details']}")
        c = st.columns(5)
        c[0].metric("Total mentions", f"{summ['total']:,}")
        c[1].metric("Positive", f"{summ['pct']['positive']:.0%}")
        c[2].metric("Negative", f"{summ['pct']['negative']:.0%}")
        c[3].metric("Neutral", f"{summ['pct']['neutral']:.0%}")
        c[4].metric("Open alerts", len(open_alerts))
        l, r = st.columns(2)
        l.plotly_chart(charts.sentiment_donut(summ), use_container_width=True)
        r.plotly_chart(charts.brand_sentiment_bar(el), use_container_width=True)
        spam_n = int(posts["is_spam"].sum())
        unscored = int((posts["sentiment"] == "unscored").sum())
        st.caption(f"{spam_n:,} posts flagged as spam and {unscored:,} in unsupported languages are excluded from metrics.")

    # ---- Timeline
    with tabs[1]:
        m = window_metrics(el, window)
        if m.empty:
            st.info("No data in range.")
        else:
            st.plotly_chart(charts.timeline_chart(m), use_container_width=True)
            st.plotly_chart(charts.negative_share_lines(m), use_container_width=True)
            st.caption("Negative share is undefined (gap) for windows with no posts. Short windows are noisy - read with volume.")

    # ---- Topics
    with tabs[2]:
        ts = topic_sentiment(aspects)
        if ts.empty:
            st.info("No feature mentions yet.")
        else:
            st.plotly_chart(charts.topic_chart(ts), use_container_width=True)
        a, b = st.columns(2)
        tags = top_hashtags(el, 10)
        if not tags.empty:
            a.plotly_chart(charts.simple_bar(tags, "hashtag", "mentions", "Top hashtags"), use_container_width=True)
        for col, sent, color in ((a, "negative", charts.COLORS["negative"]), (b, "positive", charts.COLORS["positive"])):
            kw = top_keywords(el, 10, sent)
            if not kw.empty:
                col.plotly_chart(charts.simple_bar(kw, "keyword", "mentions", f"Top {sent} keywords", color), use_container_width=True)
        emo = el["emotion"].dropna()
        emo = emo[emo != "None"].value_counts().reset_index()
        if not emo.empty:
            emo.columns = ["emotion", "posts"]
            b.plotly_chart(charts.simple_bar(emo, "emotion", "posts", "Detected emotions"), use_container_width=True)

    # ---- Explorer
    with tabs[3]:
        f1, f2, f3, f4 = st.columns([3, 2, 2, 2])
        q = f1.text_input("Search text")
        sents = f2.multiselect("Sentiment", list(SENTIMENTS) + ["unscored"], default=list(SENTIMENTS))
        topics = f3.multiselect("Topic", sorted(posts["topic"].dropna().unique()))
        show_spam = f4.checkbox("Include spam", value=False)
        d = posts[posts["sentiment"].isin(sents)]
        if q:
            d = d[d["text"].str.contains(q, case=False, regex=False)]
        if topics:
            d = d[d["topic"].isin(topics)]
        if not show_spam:
            d = d[d["is_spam"] == 0]
        cols = ["created_at", "brand", "sentiment", "sentiment_score", "emotion", "topic", "text", "source", "language"]
        d = d.sort_values("created_at", ascending=False)
        st.caption(f"{len(d):,} posts (showing latest 500)")
        st.dataframe(d[cols].head(500), use_container_width=True, hide_index=True)
        st.download_button("Download filtered CSV", d[cols].to_csv(index=False), "posts.csv", "text/csv")

    # ---- Brand comparison
    with tabs[4]:
        rows = []
        for brand, g in el.groupby("brand"):
            s = sentiment_summary(g)
            rows.append({"brand": brand, "mentions": s["total"], "positive %": s["pct"]["positive"],
                         "negative %": s["pct"]["negative"], "neutral %": s["pct"]["neutral"], "avg score": s["avg_score"]})
        if rows:
            comp = pd.DataFrame(rows)
            st.dataframe(comp.style.format({"positive %": "{:.0%}", "negative %": "{:.0%}", "neutral %": "{:.0%}", "avg score": "{:+.2f}"}),
                         use_container_width=True, hide_index=True)
            st.plotly_chart(charts.brand_compare_bar(comp), use_container_width=True)
            if not aspects.empty:
                st.plotly_chart(charts.feature_heatmap(aspects), use_container_width=True)
            st.caption("Brands share the same time range, collection method and model, so figures are comparable.")
        else:
            st.info("No data.")

    # ---- Alerts
    with tabs[5]:
        if alerts.empty:
            st.success("No alerts in range.")
        else:
            st.caption("Rules: negative spike (share + z-score vs rolling baseline, minimum sample) and volume spike.")
            for a in open_alerts.head(25).itertuples():
                icon = "🔴" if a.severity == "high" else "🟠"
                with st.expander(f"{icon} {a.brand} - {a.rule.replace('_', ' ')} - {a.window_start:%Y-%m-%d %H:%M}"):
                    st.write(a.details)
                    st.write(f"Observed: **{a.observed:.2f}** | Baseline: **{a.baseline:.2f}** | Severity: **{a.severity}**")
                    if st.button("Acknowledge", key=f"ack{a.id}"):
                        db.acknowledge_alert(a.id)
                        st.rerun()
            with st.expander("All alerts (table)"):
                st.dataframe(alerts, use_container_width=True, hide_index=True)

    # ---- Report
    with tabs[6]:
        c1, c2 = st.columns(2)
        who = c1.selectbox("Brand", ["All brands"] + brands)
        hrs = c2.number_input("Report period (hours)", 1, 168, 24)
        sel = posts if who == "All brands" else posts[posts["brand"] == who]
        asp = aspects if who == "All brands" else aspects[aspects["brand"] == who]
        alr = alerts if who == "All brands" or alerts.empty else alerts[alerts["brand"] == who]
        report = generate_report(sel, asp, alr, who, int(hrs))
        st.markdown(report)
        st.download_button("Download report (.md)", report, "brandpulse_report.md", "text/markdown")
        st.download_button("Download window metrics (.csv)", window_metrics(sel, window).to_csv(index=False),
                           "window_metrics.csv", "text/csv")

if posts.empty:
    st.info("No data yet. Open **Demo data controls** in the sidebar and load demo history, "
            "or run `python -m pipeline.run_stream --backfill 48` in a terminal.")
else:
    render()

if live:
    time.sleep(interval)
    st.rerun()
