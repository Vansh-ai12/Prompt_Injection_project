"""
Layer 2: Canary Token Leak Detector
Pure code implementation - no ML required

Functions:
- Inject randomized canary strings into system prompts
- Scan model outputs/tool-call payloads for canary leaks
- Log any detected leaks with full context
"""

import secrets
import string
from typing import Dict, List
from datetime import datetime
import logging
from layer2_guards import (
    check_obfuscation,
    check_fake_delimiters,
    check_known_phrases,
    check_extraction_probe,
    run_input_guards
)

logger = logging.getLogger(__name__)


class CanaryManager:
    """Manages canary token injection and detection"""

    def __init__(self, token_length: int = 32, num_tokens: int = 5):
        """
        Initialize canary manager

        Args:
            token_length: Length of each canary token (default: 32)
            num_tokens: Number of canary tokens to generate (default: 5)
        """
        self.token_length = token_length
        self.num_tokens = num_tokens
        self.active_canaries: Dict[str, dict] = {}
        self.detected_leaks: List[dict] = []

    def generate_canary(self) -> str:
        """
        Generate a random canary token

        Returns:
            Random string of specified length
        """
        alphabet = string.ascii_letters + string.digits
        return ''.join(secrets.choice(alphabet) for _ in range(self.token_length))

    def inject_canaries(self, system_prompt: str) -> str:
        """
        Inject canary tokens into a system prompt

        Args:
            system_prompt: Original system prompt

        Returns:
            System prompt with canary tokens embedded
        """
        canaries = []
        for i in range(self.num_tokens):
            token = self.generate_canary()
            canaries.append(token)

            self.active_canaries[token] = {
                "position": i,
                "injected_at": datetime.now().isoformat(),
                "token": token
            }

        canary_section = "\n<!-- SYSTEM_CANARY_TOKENS: " + ", ".join(canaries) + " -->\n"
        augmented_prompt = canary_section + system_prompt

        logger.info(f"Injected {len(canaries)} canary tokens into system prompt")
        return augmented_prompt

    def check_for_leaks(self, text: str) -> dict:
        """
        Check if any canary tokens appear in the given text

        Args:
            text: Text to scan for canary leaks

        Returns:
            dict with leak_detected (bool) and details
        """
        if not self.active_canaries:
            return {
                "leak_detected": False,
                "leaked_tokens": [],
                "details": "No active canaries to check"
            }

        leaked_tokens = []
        for token, metadata in self.active_canaries.items():
            if token in text:
                leaked_tokens.append({
                    "token": token,
                    "position": metadata["position"],
                    "injected_at": metadata["injected_at"]
                })
                logger.warning(f"Canary leak detected: {token[:8]}...")

        leak_detected = len(leaked_tokens) > 0

        if leak_detected:
            leak_record = {
                "detected_at": datetime.now().isoformat(),
                "leaked_tokens": leaked_tokens,
                "text_snippet": text[:200] + "..." if len(text) > 200 else text
            }
            self.detected_leaks.append(leak_record)

        return {
            "leak_detected": leak_detected,
            "leaked_tokens": leaked_tokens,
            "details": f"{len(leaked_tokens)} canary tokens detected" if leak_detected else "No leaks detected"
        }

    def check_tool_call_payload(self, tool_call: dict) -> dict:
        """
        Check if canary tokens appear in tool call payload

        Args:
            tool_call: Tool call dictionary to check

        Returns:
            dict with leak_detected (bool) and details
        """
        tool_call_str = str(tool_call)
        return self.check_for_leaks(tool_call_str)

    def clear_canaries(self):
        """Clear all active canary tokens"""
        self.active_canaries.clear()
        logger.info("Cleared all canary tokens")

    def get_leak_history(self) -> List[dict]:
        """
        Get history of detected leaks

        Returns:
            List of leak records
        """
        return self.detected_leaks

    def reset_leak_history(self):
        """Clear leak history"""
        self.detected_leaks.clear()
        logger.info("Cleared leak history")


_canary_manager = None


def get_canary_manager() -> CanaryManager:
    """Get or create the singleton canary manager instance"""
    global _canary_manager
    if _canary_manager is None:
        _canary_manager = CanaryManager()
    return _canary_manager
