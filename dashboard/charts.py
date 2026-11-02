"""Plotly figure builders for the Streamlit dashboard."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORS = {"positive": "#2e9e57", "negative": "#d9473f", "neutral": "#9aa0a6"}
LAYOUT = dict(margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.2))


def sentiment_donut(summary: dict) -> go.Figure:
    labels = list(summary["counts"])
    fig = go.Figure(go.Pie(labels=labels, values=[summary["counts"][s] for s in labels], hole=0.55,
                           marker=dict(colors=[COLORS[s] for s in labels]), sort=False))
    fig.update_layout(title="Sentiment distribution", **LAYOUT)
    return fig


def brand_sentiment_bar(df: pd.DataFrame) -> go.Figure:
    g = df.groupby(["brand", "sentiment"]).size().reset_index(name="posts")
    fig = px.bar(g, x="brand", y="posts", color="sentiment", color_discrete_map=COLORS, barmode="stack",
                 title="Mentions by brand")
    fig.update_layout(**LAYOUT)
    return fig


def timeline_chart(metrics: pd.DataFrame) -> go.Figure:
    agg = metrics.groupby("window_start")[["positive_count", "negative_count", "neutral_count"]].sum().reset_index()
    agg["share"] = agg["negative_count"] / agg[["positive_count", "negative_count", "neutral_count"]].sum(axis=1).replace(0, float("nan"))
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for s in ("positive", "neutral", "negative"):
        fig.add_bar(x=agg["window_start"], y=agg[f"{s}_count"], name=s, marker_color=COLORS[s])
    fig.add_scatter(x=agg["window_start"], y=agg["share"], name="negative share", mode="lines",
                    line=dict(color="black", width=2), secondary_y=True)
    fig.update_layout(barmode="stack", title="Sentiment over time", **LAYOUT)
    fig.update_yaxes(title_text="posts", secondary_y=False)
    fig.update_yaxes(title_text="negative share", tickformat=".0%", range=[0, 1], secondary_y=True)
    return fig


def negative_share_lines(metrics: pd.DataFrame) -> go.Figure:
    fig = px.line(metrics.dropna(subset=["negative_share"]), x="window_start", y="negative_share", color="brand",
                  title="Negative share by brand")
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(**LAYOUT)
    return fig


def topic_chart(ts: pd.DataFrame) -> go.Figure:
    long = ts.melt(id_vars="topic", value_vars=["positive", "neutral", "negative"], var_name="sentiment", value_name="mentions")
    fig = px.bar(long, x="mentions", y="topic", color="sentiment", color_discrete_map=COLORS, orientation="h",
                 title="Feature sentiment (aspect level)")
    fig.update_layout(yaxis=dict(categoryorder="total ascending"), **LAYOUT)
    return fig


def simple_bar(df: pd.DataFrame, x: str, y: str, title: str, color: str = "#4c78a8") -> go.Figure:
    fig = px.bar(df, x=y, y=x, orientation="h", title=title, color_discrete_sequence=[color])
    fig.update_layout(yaxis=dict(categoryorder="total ascending"), **LAYOUT)
    return fig


def brand_compare_bar(summary: pd.DataFrame) -> go.Figure:
    long = summary.melt(id_vars="brand", value_vars=["positive %", "negative %", "neutral %"], var_name="sentiment")
    long["sentiment"] = long["sentiment"].str.replace(" %", "", regex=False)
    fig = px.bar(long, x="brand", y="value", color="sentiment", barmode="group", color_discrete_map=COLORS,
                 title="Sentiment mix by brand")
    fig.update_yaxes(title="share of mentions")
    fig.update_layout(**LAYOUT)
    return fig


def feature_heatmap(aspects: pd.DataFrame) -> go.Figure:
    t = aspects.groupby(["brand", "topic", "sentiment"]).size().unstack(fill_value=0)
    for s in ("positive", "negative", "neutral"):
        if s not in t:
            t[s] = 0
    t["net"] = (t["positive"] - t["negative"]) / t[["positive", "negative", "neutral"]].sum(axis=1)
    pivot = t["net"].unstack("brand")
    fig = px.imshow(pivot, color_continuous_scale="RdYlGn", zmin=-1, zmax=1, aspect="auto",
                    title="Net feature sentiment (-1 negative, +1 positive)")
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return fig
