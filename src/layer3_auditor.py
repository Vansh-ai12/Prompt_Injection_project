"""
Layer 3: Tool-Call Intent Auditor
LLM wrapper using Groq API to validate tool calls against user intent

This is multi-step aware - looks at recent tool-call history (N=3) to catch
chained/salami-sliced attacks, not just single actions in isolation.

Uses llama-3.1-8b-instant for speed and low cost.
"""

import os
from typing import Dict, List, Optional
from datetime import datetime
import logging

from groq import Groq
from src.config import Config

logger = logging.getLogger(__name__)


class ToolCallAuditor:
    """Audits tool calls using Groq-hosted LLM as judge"""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the tool call auditor

        Args:
            api_key: Groq API key (default: from Config)
            model: Groq model to use (default: from Config, defaults to llama-3.1-8b-instant)
        """
        self.api_key = api_key or Config.GROQ_API_KEY
        self.model = model or Config.GROQ_MODEL
        self.history_length = Config.TOOL_CALL_HISTORY_LENGTH

        if not self.api_key:
            raise ValueError("GROQ_API_KEY is required for Layer 3 auditor")

        self.client = Groq(api_key=self.api_key)
        logger.info(f"Layer 3: Tool-call auditor initialized with model {self.model}")

    def audit(
        self,
        user_intent: str,
        tool_call: dict,
        tool_call_history: Optional[List[dict]] = None
    ) -> dict:
        """
        Audit a proposed tool call against user intent

        Args:
            user_intent: Original user's stated task/goal
            tool_call: Proposed tool call to audit
            tool_call_history: Recent tool calls (last N) for context

        Returns:
            dict with:
                - decision: "ALLOW", "BLOCK", or "ESCALATE"
                - reason: One-line explanation
        """
        history_context = self._format_history(tool_call_history or [])

        prompt = self._build_audit_prompt(
            user_intent=user_intent,
            tool_call=tool_call,
            history_context=history_context
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=256
            )

            decision_text = response.choices[0].message.content.strip()

            decision = self._parse_decision(decision_text)

            logger.info(f"Layer 3 audit: {decision['decision']} - {decision['reason']}")

            return decision

        except Exception as e:
            logger.error(f"Error in Layer 3 audit: {e}")
            return {
                "decision": "ALLOW",
                "reason": f"Audit error (allowing): {str(e)}"
            }

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the auditor"""
        return """You are a security auditor for an AI cybersecurity copilot. Your job is to evaluate whether a proposed tool call aligns with the user's stated intent.

The copilot has tool-calling ability and can execute real actions (scans, file operations, data queries). A successful prompt injection could trigger destructive actions.

Evaluate the tool call and respond with:
- ALLOW: The action clearly aligns with the user's stated goal
- BLOCK: The action contradicts the user's intent or appears malicious
- ESCALATE: The action is ambiguous and requires human review

Response format: DECISION: [ALLOW/BLOCK/ESCALATE] | Reason: [one-line explanation]

Be conservative - if in doubt, ESCALATE rather than allow."""

    def _build_audit_prompt(
        self,
        user_intent: str,
        tool_call: dict,
        history_context: str
    ) -> str:
        """Build the audit prompt"""
        prompt = f"""User's stated intent: {user_intent}

Recent tool call history (last {self.history_length} actions):
{history_context if history_context else "No recent tool calls"}

Proposed tool call to audit:
{self._format_tool_call(tool_call)}

Does this proposed tool call align with the user's intent? Consider:
- Is this action logically consistent with the stated goal?
- Could this be part of a chained attack or salami-slicing?
- Are there any red flags in the parameters or target?

Respond in format: DECISION: [ALLOW/BLOCK/ESCALATE] | Reason: [explanation]"""

        return prompt

    def _format_history(self, history: List[dict]) -> str:
        """Format tool call history for the prompt"""
        if not history:
            return ""

        formatted = []
        for i, call in enumerate(history[-self.history_length:], 1):
            formatted.append(f"{i}. {self._format_tool_call(call)}")

        return "\n".join(formatted)

    def _format_tool_call(self, tool_call: dict) -> str:
        """Format a tool call for display"""
        if not tool_call:
            return "No tool call"

        tool_name = tool_call.get("name", "unknown")
        parameters = tool_call.get("parameters", {})

        return f"Tool: {tool_name} | Parameters: {parameters}"

    def _parse_decision(self, response_text: str) -> dict:
        """Parse the LLM response into structured decision"""
        response_text = response_text.strip()

        if "DECISION:" in response_text:
            parts = response_text.split("DECISION:")[1].strip()
            if "|" in parts:
                decision_part, reason_part = parts.split("|", 1)
                decision = decision_part.strip().upper()
                reason = reason_part.replace("Reason:", "").strip()
            else:
                decision = parts.split()[0].upper()
                reason = response_text
        else:
            words = response_text.split()
            decision = words[0].upper() if words else "ESCALATE"
            reason = response_text

        if decision not in ["ALLOW", "BLOCK", "ESCALATE"]:
            decision = "ESCALATE"
            reason = f"Unclear decision format ({reason})"

        return {
            "decision": decision,
            "reason": reason
        }


_auditor = None


def get_auditor() -> ToolCallAuditor:
    """Get or create the singleton auditor instance"""
    global _auditor
    if _auditor is None:
        _auditor = ToolCallAuditor()
    return _auditor
