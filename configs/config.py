"""Central configuration for the predictive maintenance system."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).parent.parent

# Paths
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"

# C-MAPSS sensor columns
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]  # s1–s21
SETTING_COLUMNS = ["setting_1", "setting_2", "setting_3"]

COLUMN_NAMES = (
    ["unit_id", "cycle"]
    + SETTING_COLUMNS
    + SENSOR_COLUMNS
)

# Sensors with near-zero variance in FD001 (drop them)
LOW_VARIANCE_SENSORS = ["sensor_1", "sensor_5", "sensor_6", "sensor_10",
                        "sensor_16", "sensor_18", "sensor_19"]

FEATURE_COLUMNS = [
    c for c in SENSOR_COLUMNS + SETTING_COLUMNS
    if c not in LOW_VARIANCE_SENSORS
]

# RUL clip — cap max RUL at 125 cycles (piecewise linear target)
RUL_CLIP = 125

# LLM
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

def require_groq_key():
    """Call this before making any LLM request. Raises clearly if key is missing."""
    if not GROQ_API_KEY:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Add it to your .env file.\n"
            "Get a free key at https://console.groq.com"
        )

# Phoenix/Arize — open-source local tracing (https://phoenix.arize.com)
PHOENIX_TRACING = os.getenv("PHOENIX_TRACING", "false").lower() == "true"
PHOENIX_HOST = os.getenv("PHOENIX_HOST", "http://localhost:6006")

# Thresholds for alerts
CRITICAL_RUL_THRESHOLD = 30   # cycles — critical maintenance needed
WARNING_RUL_THRESHOLD = 60    # cycles — plan maintenance soon
ANOMALY_ZSCORE_THRESHOLD = 3.0
