"""
Common database module for Supabase logging
Logs all defense pipeline decisions for research paper evaluation
"""

from typing import Dict, Optional
from datetime import datetime
import logging

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    logging.warning("Supabase not installed. Install with: pip install supabase")

from src.common.config import Config

logger = logging.getLogger(__name__)


class DefenseLogger:
    """Logs defense pipeline decisions to Supabase"""

    def __init__(self):
        """Initialize Supabase client"""
        if not SUPABASE_AVAILABLE:
            logger.warning("Supabase not available - logging to file only")
            self.client = None
            return

        if not Config.SUPABASE_URL or not Config.SUPABASE_KEY:
            logger.warning("Supabase credentials not configured - logging to file only")
            self.client = None
            return

        try:
            self.client: Client = create_client(
                Config.SUPABASE_URL,
                Config.SUPABASE_KEY
            )
            logger.info("Supabase client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            self.client = None

    def log_decision(
        self,
        message: str,
        layer1_result: Optional[Dict],
        layer2_result: Optional[Dict],
        layer3_result: Optional[Dict],
        final_decision: str,
        total_latency_ms: float,
        metadata: Optional[Dict] = None
    ):
        """
        Log a defense pipeline decision

        Args:
            message: Input message
            layer1_result: Layer 1 classification result
            layer2_result: Layer 2 canary check result
            layer3_result: Layer 3 audit result
            final_decision: Final decision (allowed/blocked)
            total_latency_ms: Total pipeline latency
            metadata: Additional metadata
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "message": message[:500],  # Truncate if too long
            "layer1_classification": layer1_result.get("classification") if layer1_result else None,
            "layer1_confidence": layer1_result.get("confidence") if layer1_result else None,
            "layer1_blocked": layer1_result.get("blocked") if layer1_result else None,
            "layer2_canary_triggered": layer2_result.get("canary_triggered") if layer2_result else None,
            "layer2_blocked": layer2_result.get("blocked") if layer2_result else None,
            "layer3_decision": layer3_result.get("decision") if layer3_result else None,
            "layer3_blocked": layer3_result.get("blocked") if layer3_result else None,
            "final_decision": final_decision,
            "total_latency_ms": total_latency_ms,
            "metadata": metadata or {}
        }

        if self.client:
            try:
                self.client.table("defense_logs").insert(log_entry).execute()
                logger.debug("Logged decision to Supabase")
            except Exception as e:
                logger.error(f"Failed to log to Supabase: {e}")
                self._log_to_file(log_entry)
        else:
            self._log_to_file(log_entry)

    def _log_to_file(self, log_entry: Dict):
        """Fallback: log to file"""
        import json
        from pathlib import Path

        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        log_file = log_dir / f"defense_logs_{datetime.now().strftime('%Y%m%d')}.jsonl"

        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        logger.debug(f"Logged decision to file: {log_file}")


# Singleton instance
_logger: Optional[DefenseLogger] = None


def get_defense_logger() -> DefenseLogger:
    """Get or create the singleton defense logger instance"""
    global _logger
    if _logger is None:
        _logger = DefenseLogger()
    return _logger
