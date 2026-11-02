"""Evaluate the sentiment backend on a labelled CSV (columns: text,label).

  python evaluate.py --csv data/sample_posts.csv --backend vader

WARNING: the bundled sample labels are synthetic ground truth, so scores will be optimistic.
For a meaningful report, hand-label 300+ real posts and keep them out of development.
"""
from __future__ import annotations

import argparse

import pandas as pd

from models.sentiment import SentimentAnalyzer
from preprocessing.cleaning import clean_text

CLASSES = ["positive", "negative", "neutral"]


def evaluate(df: pd.DataFrame, analyzer: SentimentAnalyzer) -> dict:
    df = df[df["label"].isin(CLASSES)].copy()
    texts = [clean_text(t) for t in df["text"]]
    pred = [p.label for p in analyzer.predict_batch(texts)]
    truth = df["label"].tolist()
    per_class, f1s = {}, []
    for c in CLASSES:
        tp = sum(1 for t, p in zip(truth, pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(truth, pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(truth, pred) if t == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[c] = {"precision": prec, "recall": rec, "f1": f1, "support": truth.count(c)}
        f1s.append(f1)
    acc = sum(t == p for t, p in zip(truth, pred)) / max(len(truth), 1)
    return {"n": len(truth), "accuracy": acc, "macro_f1": sum(f1s) / len(f1s), "per_class": per_class}


if __name__ == "__main__":
    import config

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(config.SAMPLE_CSV))
    ap.add_argument("--backend", default=None, help="vader | transformer")
    a = ap.parse_args()
    analyzer = SentimentAnalyzer(a.backend)
    res = evaluate(pd.read_csv(a.csv), analyzer)
    print(f"Model: {analyzer.model_name} | n={res['n']} | accuracy={res['accuracy']:.3f} | macro-F1={res['macro_f1']:.3f}")
    for c, m in res["per_class"].items():
        print(f"  {c:9s} P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f} (n={m['support']})")
