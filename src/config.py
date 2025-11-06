# File: src/config.py
# Centralized configuration for GreenAI Email Classifier

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Central configuration for the GreenAI Email Classifier"""

    # ------------------------------
    # Path Configuration
    # ------------------------------
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    MODELS_DIR = DATA_DIR / "models"
    LOGS_DIR = PROJECT_ROOT / "logs"
    EMISSIONS_DIR = PROJECT_ROOT / "emissions"

    # ------------------------------
    # Database Configuration
    # ------------------------------
    DATABASE_PATH = PROJECT_ROOT / "greenai.db"
    DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

    # ------------------------------
    # Model Paths
    # ------------------------------
    GREEN_MODEL_PATH = MODELS_DIR / "green"
    MEDIUM_MODEL_PATH = MODELS_DIR / "medium"
    HEAVY_MODEL_PATH = MODELS_DIR / "heavy"

    # ------------------------------
    # Model Configuration
    # ------------------------------
    CLASSES = ["work", "spam", "support"]
    NUM_CLASSES = len(CLASSES)

    TEST_SIZE = 0.15
    VAL_SIZE = 0.15
    RANDOM_STATE = 42

    # Cascade thresholds (default, can be updated in API)
    GREEN_THRESHOLD = 0.85
    MEDIUM_THRESHOLD = 0.80

    # ------------------------------
    # Energy Tracking
    # ------------------------------
    CARBON_TRACKING_ENABLED = os.getenv("CARBON_TRACKING_ENABLED", "true").lower() == "true"
    CARBON_COUNTRY_CODE = os.getenv("CARBON_COUNTRY_CODE", "IND")

    # ------------------------------
    # API & Dashboard
    # ------------------------------
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", 8501))

    # ------------------------------
    # Directory Initialization
    # ------------------------------
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist"""
        for dir_path in [
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.GREEN_MODEL_PATH,
            cls.MEDIUM_MODEL_PATH,
            cls.HEAVY_MODEL_PATH,
            cls.LOGS_DIR,
            cls.EMISSIONS_DIR,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)


# Initialize directories at import
Config.ensure_directories()
