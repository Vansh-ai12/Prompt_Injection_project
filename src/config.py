"""
Configuration settings for the prompt injection defense system
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration class"""

    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    LOGS_DIR = PROJECT_ROOT / "logs"

    # Layer 1: Classifier
    CLASSIFIER_MODEL_PATH = os.getenv("CLASSIFIER_MODEL_PATH", str(MODELS_DIR / "classifier"))
    CLASSIFIER_CONFIDENCE_THRESHOLD = float(os.getenv("CLASSIFIER_CONFIDENCE_THRESHOLD", "70.0"))

    # Layer 2: Canary tokens
    CANARY_TOKEN_LENGTH = int(os.getenv("CANARY_TOKEN_LENGTH", "32"))
    NUM_CANARY_TOKENS = int(os.getenv("NUM_CANARY_TOKENS", "5"))

    # Layer 3: Groq Auditor
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")  # Using smaller/faster model
    TOOL_CALL_HISTORY_LENGTH = int(os.getenv("TOOL_CALL_HISTORY_LENGTH", "3"))

    # Supabase (for logging)
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

    # API
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))

    @classmethod
    def validate(cls):
        """Validate required configuration"""
        errors = []

        if not cls.GROQ_API_KEY:
            errors.append("GROQ_API_KEY is required for Layer 3")

        if cls.SUPABASE_URL and not cls.SUPABASE_KEY:
            errors.append("SUPABASE_KEY is required when SUPABASE_URL is set")

        if errors:
            raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))

        return True
