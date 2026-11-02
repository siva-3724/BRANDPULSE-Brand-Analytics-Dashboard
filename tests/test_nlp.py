import pytest

from models.emotion import detect_emotion
from models.sentiment import SentimentAnalyzer
from models.topics import aspect_clauses, extract_topics


@pytest.fixture(scope="module")
def analyzer():
    return SentimentAnalyzer("vader")


@pytest.mark.parametrize("text,label", [
    ("The camera quality is amazing!", "positive"),
    ("Battery drains too quickly.", "negative"),
    ("The phone gets hot while charging.", "negative"),
    ("Nova X1 is available in three colours.", "neutral"),
    ("Great display and smooth performance.", "positive"),
    ("The camera is not good.", "negative"),
])
def test_sentiment_labels(analyzer, text, label):
    assert analyzer.predict(text).label == label


def test_topics():
    assert "Battery" in extract_topics("Battery drains too quickly.")
    assert {"Heating", "Battery"} <= set(extract_topics("The phone gets hot while charging."))
    assert extract_topics("Nice weather today") == []


def test_aspect_sentiment_mixed_opinion(analyzer):
    clauses = aspect_clauses("The display looks fantastic, but the battery is disappointing.")
    result = {t: analyzer.predict(c).label for c, topics in clauses for t in topics}
    assert result["Display"] == "positive" and result["Battery"] == "negative"


def test_emotion():
    assert detect_emotion("I love this, amazing 😍") == "Joy"
    assert detect_emotion("Worst phone ever, awful") == "Anger"
    assert detect_emotion("It has a box") == "None"
