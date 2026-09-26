"""
Configuration settings for the prompt injection defense system
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Configuration class"""

    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    LOGS_DIR = PROJECT_ROOT / "logs"

    CLASSIFIER_MODEL_PATH = os.getenv("CLASSIFIER_MODEL_PATH", str(MODELS_DIR / "classifier"))
    CLASSIFIER_CONFIDENCE_THRESHOLD = float(os.getenv("CLASSIFIER_CONFIDENCE_THRESHOLD", "70.0"))

    CANARY_TOKEN_LENGTH = int(os.getenv("CANARY_TOKEN_LENGTH", "32"))
    NUM_CANARY_TOKENS = int(os.getenv("NUM_CANARY_TOKENS", "5"))

    LAYER2_COMBINED_THRESHOLD = float(os.getenv("LAYER2_COMBINED_THRESHOLD", "50.0"))
    FUSION_BORDERLINE_L1_MIN = float(os.getenv("FUSION_BORDERLINE_L1_MIN", "45.0"))
    FUSION_BORDERLINE_L2_MIN = float(os.getenv("FUSION_BORDERLINE_L2_MIN", "25.0"))
    ATTACK_PHRASES_PATH = DATA_DIR / "attack_phrases.json"
    TOOL_RISK_TIERS_PATH = DATA_DIR / "tool_risk_tiers.json"
    AUDIT_LOG_PATH = DATA_DIR / "audit_log.jsonl"
    LOW_RISK_SALAMI_WINDOW = int(os.getenv("LOW_RISK_SALAMI_WINDOW", "3"))  # consecutive same-tool LOW calls that trigger full audit
    AUDIT_VOTING_CALLS = int(os.getenv("AUDIT_VOTING_CALLS", "3"))         # number of LLM calls for consensus voting
    AUDIT_INTENT_VERIFY_MODEL = os.getenv("AUDIT_INTENT_VERIFY_MODEL", "")  # empty = use same GROQ_MODEL

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    TOOL_CALL_HISTORY_LENGTH = int(os.getenv("TOOL_CALL_HISTORY_LENGTH", "3"))

    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

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


# Module-level exports for direct import
DATA_DIR = Config.DATA_DIR
LAYER2_COMBINED_THRESHOLD = Config.LAYER2_COMBINED_THRESHOLD
FUSION_BORDERLINE_L1_MIN = Config.FUSION_BORDERLINE_L1_MIN
FUSION_BORDERLINE_L2_MIN = Config.FUSION_BORDERLINE_L2_MIN
ATTACK_PHRASES_PATH = Config.ATTACK_PHRASES_PATH
TOOL_RISK_TIERS_PATH = Config.TOOL_RISK_TIERS_PATH
AUDIT_LOG_PATH = Config.AUDIT_LOG_PATH

