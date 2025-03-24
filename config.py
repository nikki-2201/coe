"""Configuration settings for the social media sentiment analyzer."""

import os

# Directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(BASE_DIR, "data", "samples")
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
MODEL_DIR = os.path.join(BASE_DIR, "models", "saved")
CUSTOM_MODEL_PATH = os.path.join(MODEL_DIR, "custom_sentiment_model.joblib")

# Create directories if they don't exist
for directory in [SAMPLE_DIR, UPLOAD_DIR, MODEL_DIR]:
    os.makedirs(directory, exist_ok=True)

# Analysis settings
MAX_VIZ_SAMPLES = 10000  # Maximum number of samples to use for visualization
ALLOWED_EXTENSIONS = ["csv", "xlsx", "json"]

# Model settings
DEFAULT_MODEL_TYPE = "transformer"
AVAILABLE_MODELS = ["transformer", "custom"]
MODEL_ALGORITHMS = ["svm", "random_forest"]

# Text processing settings
MAX_TEXT_LENGTH = 1000
MIN_TEXT_LENGTH = 3
LANGUAGE = "english"

# Visualization settings
CHART_HEIGHT = 400
CHART_WIDTH = 800
MAX_WORDCLOUD_WORDS = 100
SENTIMENT_COLORS = {
    "positive": "#2ECC71",
    "negative": "#E74C3C",
    "neutral": "#3498DB"
}

# Paths
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
SAMPLE_DIR = os.path.join(BASE_DIR, "data", "sample")
MODEL_DIR = os.path.join(BASE_DIR, "models", "saved_models")

# Create directories if they don't exist
for directory in [UPLOAD_DIR, PROCESSED_DIR, SAMPLE_DIR, MODEL_DIR]:
    os.makedirs(directory, exist_ok=True)

# Model settings
DEFAULT_MODEL_TYPE = "transformer"  # Options: "transformer", "custom"
CUSTOM_MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_model.pkl")

# Supported file types for upload
ALLOWED_EXTENSIONS = ["csv", "xlsx", "json"]
MAX_UPLOAD_SIZE_MB = 100

# Default text column names to look for in uploaded data
DEFAULT_TEXT_COLUMNS = ["text", "content", "message", "post", "tweet", "comment"]

# Default time/date column names to look for
DEFAULT_TIME_COLUMNS = ["time", "date", "created_at", "timestamp", "datetime"]

# Maximum number of samples for visualization
MAX_VIZ_SAMPLES = 10000

# Cache settings
CACHE_EXPIRY = 3600  # seconds