from preprocessing.cleaning import clean_text, detect_language, extract_hashtags, is_spam, text_hash


def test_clean_removes_urls_keeps_emoji_and_negation():
    out = clean_text("WOW!!! 😍 The Nova camera is NOT bad #NovaX1 https://example.com")
    assert "http" not in out and "😍" in out and "NOT" in out and "#NovaX1" in out


def test_clean_mentions_and_repeats():
    assert clean_text("@john soooooo good") == "@user sooo good"
    assert clean_text(None) == "" and clean_text("   ") == ""


def test_hashtags_and_hash():
    assert extract_hashtags("Love it #NovaX1 #Launch") == ["novax1", "launch"]
    assert text_hash("Great phone!") == text_hash("great   phone")


def test_spam():
    assert is_spam("FREE GIVEAWAY!!! click here http://a.com")
    assert not is_spam("The battery drains too quickly.")


def test_language_script_detection():
    assert detect_language("பேட்டரி மிகவும் நன்றாக உள்ளது") == "ta"
    assert detect_language("बैटरी बहुत अच्छी है") == "hi"
    assert detect_language("good phone") == "en"
