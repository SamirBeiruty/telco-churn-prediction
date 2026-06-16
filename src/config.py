"""Central configuration: file paths, column groups, and shared constants.

Keeping these in one place means the notebook and the scripts read from the
exact same definitions, so the two can never silently drift apart.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (resolved relative to the project root, so the code works no matter
# what directory you launch it from)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "telco_clean.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# ---------------------------------------------------------------------------
# Modelling constants
# ---------------------------------------------------------------------------
TARGET = "Churn"
ID_COLUMN = "customerID"
RANDOM_STATE = 42          # fixed seed -> every run is reproducible
TEST_SIZE = 0.20           # 80/20 train-test split

# Service columns whose "Yes" values we count to build a "number of services"
# feature. Phone/internet add-ons all live here.
SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# The three raw numeric columns in the dataset.
NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]
