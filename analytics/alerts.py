"""Negative-sentiment spike and volume-spike detection against a rolling baseline."""
from __future__ import annotations

from datetime import timedelta

import pandas as pd

import config
from analytics.metrics import eligible, window_metrics


def detect_spikes(metrics: pd.DataFrame, brand: str, *, min_sample=None, neg_share=None, z_thresh=None,
                  min_lift=None, baseline_windows=None, min_baseline=None, volume_mult=None) -> list[dict]:
    """Scan every window of one brand and return alerts.

    negative_spike : enough posts AND negative share >= threshold AND z-score vs baseline >= z_thresh
                     AND lift over baseline >= min_lift (tiny samples never alert).
    volume_spike   : enough posts AND volume >= volume_mult x baseline mean volume.
    """
    min_sample = min_sample or config.ALERT_MIN_SAMPLE
    neg_share = neg_share or config.ALERT_NEG_SHARE
    z_thresh = z_thresh or config.ALERT_Z
    min_lift = config.ALERT_MIN_LIFT if min_lift is None else min_lift
    baseline_windows = baseline_windows or config.ALERT_BASELINE_WINDOWS
    min_baseline = min_baseline or config.ALERT_MIN_BASELINE
    volume_mult = volume_mult or config.ALERT_VOLUME_MULT

    m = metrics[metrics["brand"] == brand].sort_values("window_start").reset_index(drop=True)
    alerts = []
    for i in range(len(m)):
        cur = m.iloc[i]
        total = float(cur["total_mentions"])
        if total < min_sample:
            continue
        hist = m.iloc[max(0, i - baseline_windows): i]
        usable = hist[hist["total_mentions"] >= 3]
        if len(usable) < min_baseline:
            continue
        base_share = usable["negative_count"].sum() / usable["total_mentions"].sum()
        std = usable["negative_share"].std(ddof=0)
        std = 0.0 if pd.isna(std) else float(std)
        share = float(cur["negative_share"])
        z = (share - base_share) / max(std, 0.05)
        ws = cur["window_start"].to_pydatetime()

        if share >= neg_share and z >= z_thresh and (share - base_share) >= min_lift:
            sev = "high" if (share >= 0.6 and z >= 4) else "medium"
            alerts.append({
                "brand": brand, "window_start": ws, "rule": "negative_spike", "observed": share,
                "baseline": float(base_share), "severity": sev,
                "details": f"Negative share {share:.0%} vs baseline {base_share:.0%} "
                           f"(z={z:.1f}, {int(cur['negative_count'])}/{int(total)} posts).",
            })
        base_vol = float(hist["total_mentions"].mean())
        if base_vol > 0 and total >= volume_mult * base_vol:
            ratio = total / base_vol
            alerts.append({
                "brand": brand, "window_start": ws, "rule": "volume_spike", "observed": total,
                "baseline": base_vol, "severity": "high" if ratio >= 6 else "medium",
                "details": f"Mention volume {int(total)} is {ratio:.1f}x the baseline ({base_vol:.1f}/window).",
            })
    return alerts


def _top_negative_topics(posts: pd.DataFrame, alert: dict, minutes: int) -> str:
    d = eligible(posts)
    ws = alert["window_start"]
    d = d[(d["brand"] == alert["brand"]) & (d["sentiment"] == "negative")
          & (d["created_at"] >= ws) & (d["created_at"] < ws + timedelta(minutes=minutes))]
    if d.empty:
        return ""
    top = d["topic"].value_counts().head(3)
    return "Top complaint topics: " + ", ".join(f"{t} ({n})" for t, n in top.items()) + "."


def scan_alerts(posts: pd.DataFrame, metrics: pd.DataFrame | None = None, minutes: int | None = None) -> list[dict]:
    minutes = minutes or config.WINDOW_MINUTES
    metrics = metrics if metrics is not None else window_metrics(posts, minutes)
    alerts: list[dict] = []
    if metrics.empty:
        return alerts
    for brand in metrics["brand"].unique():
        alerts += detect_spikes(metrics, brand)
    for a in alerts:
        a["details"] = (a["details"] + " " + _top_negative_topics(posts, a, minutes)).strip()
    return alerts
