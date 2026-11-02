"""Central configuration. Override with environment variables or a .env file."""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
DB_PATH = Path(os.getenv("BRANDPULSE_DB", DATA_DIR / "brandpulse.db"))
SAMPLE_CSV = DATA_DIR / "sample_posts.csv"

# --- NLP ---------------------------------------------------------------
SENTIMENT_BACKEND = os.getenv("SENTIMENT_BACKEND", "vader").split("#")[0].strip().lower()
TRANSFORMER_MODEL = os.getenv(
    "TRANSFORMER_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest"
).split("#")[0].strip()
# Languages the active sentiment model can score. Others are stored as "unscored".
SUPPORTED_LANGUAGES = {
    s.strip() for s in os.getenv("SUPPORTED_LANGUAGES", "en").split("#")[0].split(",") if s.strip()
}

# --- Monitored brands (name -> lowercase match keywords) ----------------
BRANDS: dict[str, list[str]] = {
    "Nova X1": ["nova x1", "novax1", "nova"],
    "Apex Pro": ["apex pro", "apexpro", "apex"],
    "Lumen S5": ["lumen s5", "lumens5", "lumen"],
}

# --- Product-feature (aspect) dictionaries -------------------------------
TOPIC_KEYWORDS: dict[str, list[str]] = {
    "Battery": ["battery", "charge", "charging", "charger", "drain", "drains", "drained"],
    "Camera": ["camera", "photo", "photos", "lens", "selfie", "video"],
    "Display": ["display", "screen", "brightness", "oled", "flickering"],
    "Performance": ["performance", "lag", "laggy", "sluggish", "snappy", "apps", "crash", "crashes"],
    "Price": ["price", "pricey", "expensive", "overpriced", "value", "cost"],
    "Design": ["design", "build", "colour", "colours", "color", "colors", "sleek"],
    "Heating": ["hot", "heat", "heating", "overheat", "overheats", "overheating", "cool"],
    "Customer Service": ["customer service", "support", "warranty", "refund", "service"],
    "Delivery": ["delivery", "shipping", "shipped", "courier", "delayed", "arrived"],
    "Availability": ["in stock", "sold out", "available", "availability", "pre-order", "preorder"],
    "Packaging": ["packaging", "box", "unboxing"],
}

# --- Alerting ------------------------------------------------------------
WINDOW_MINUTES = int(os.getenv("WINDOW_MINUTES", "15").split("#")[0])
ALERT_MIN_SAMPLE = 20          # ignore windows with fewer eligible posts
ALERT_NEG_SHARE = 0.35         # minimum negative share to alert
ALERT_Z = 2.5                  # z-score vs rolling baseline
ALERT_MIN_LIFT = 0.15          # negative share must exceed baseline by this much
ALERT_BASELINE_WINDOWS = 16    # rolling baseline length (windows)
ALERT_MIN_BASELINE = 4         # baseline windows required before alerting
ALERT_VOLUME_MULT = 3.0        # volume spike = N x baseline mean

# --- Reddit (official API) -----------------------------------------------
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "brandpulse-ai/0.1")
