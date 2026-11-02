"""Sentiment engine: VADER baseline (+ domain lexicon) or a Hugging Face Transformer."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import config

log = logging.getLogger(__name__)
POS_T = 0.05

# Phone-domain terms that stock VADER misses. Values are on VADER's -4..+4 scale.
DOMAIN_LEXICON = {
    "drains": -2.0, "drain": -1.5, "drained": -1.5,
    "overheats": -2.5, "overheating": -2.5, "overheat": -2.5,
    "hot": -1.0,  # in this domain "hot" almost always means "heating"
    "laggy": -2.0, "lag": -1.5, "lags": -1.8, "sluggish": -2.0,
    "crash": -2.0, "crashes": -2.0, "crashed": -2.0,
    "overpriced": -2.0, "pricey": -1.0, "expensive": -1.0,
    "defective": -2.5, "faulty": -2.0, "flimsy": -1.8, "delayed": -1.5, "flickering": -1.5,
    "snappy": 1.8, "sleek": 1.8, "crisp": 1.2, "premium": 1.5,
}


@dataclass
class Prediction:
    label: str         # positive | negative | neutral
    score: float       # signed sentiment in [-1, 1]
    confidence: float  # model confidence (a heuristic, NOT a guarantee of correctness)


def _norm_label(label: str) -> str:
    label = label.lower()
    return {"label_0": "negative", "label_1": "neutral", "label_2": "positive"}.get(label, label)


class SentimentAnalyzer:
    def __init__(self, backend: str | None = None, model_name: str | None = None):
        self.backend = (backend or config.SENTIMENT_BACKEND).lower()
        self._pipe = None
        self._vader = None
        self._transformer_name = model_name or config.TRANSFORMER_MODEL

        if self.backend == "transformer":
            try:
                from transformers import pipeline

                self._pipe = pipeline(
                    "text-classification", model=self._transformer_name,
                    top_k=None, truncation=True, max_length=256,
                )
            except Exception as exc:  # missing deps, no network, ...
                log.warning("Transformer backend unavailable (%s); falling back to VADER.", exc)
                self.backend = "vader"
        if self._pipe is None:
            self.backend = "vader"
            self._init_vader()

    def _init_vader(self) -> None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        except ImportError as exc:
            raise RuntimeError("Install vaderSentiment: pip install vaderSentiment") from exc
        self._vader = SentimentIntensityAnalyzer()
        self._vader.lexicon.update(DOMAIN_LEXICON)

    @property
    def model_name(self) -> str:
        return self._transformer_name if self._pipe is not None else "vader+domain-lexicon"

    def predict_batch(self, texts, batch_size: int = 32) -> list[Prediction]:
        texts = list(texts)
        if not texts:
            return []
        if self._pipe is not None:
            return self._predict_transformer(texts, batch_size)
        return [self._predict_vader(t) for t in texts]

    def predict(self, text: str) -> Prediction:
        return self.predict_batch([text])[0]

    def _predict_vader(self, text: str) -> Prediction:
        c = self._vader.polarity_scores(text)["compound"]
        if c >= POS_T:
            return Prediction("positive", c, min(1.0, abs(c)))
        if c <= -POS_T:
            return Prediction("negative", c, min(1.0, abs(c)))
        return Prediction("neutral", c, max(0.0, 1 - abs(c) / POS_T))

    def _predict_transformer(self, texts, batch_size) -> list[Prediction]:
        out: list[Prediction] = []
        for i in range(0, len(texts), batch_size):
            for item in self._pipe(texts[i : i + batch_size]):
                probs = {_norm_label(d["label"]): float(d["score"]) for d in item}
                label = max(probs, key=probs.get)
                out.append(Prediction(label, probs.get("positive", 0) - probs.get("negative", 0), probs[label]))
        return out
