"""
Layer 3: Tool-Call Intent Auditor  — Hardened Edition
LLM wrapper (Groq API) that audits proposed tool calls against user intent.

Five reliability & security fixes applied:
  Fix 1  — Fail closed: API errors → ESCALATE (never ALLOW).
  Fix 2  — Intent verification: raw message vs. summarized intent cross-check
             before proceeding to the tool-call audit.
  Fix 3  — Self-consistency voting: ambiguous / ESCALATE-leaning decisions
             are validated by 3 independent LLM calls that must form a majority.
  Fix 4  — Risk-tiered auditing: LOW → auto-allow (with salami-slicing guard),
             MEDIUM → single call, HIGH → voting, CRITICAL → voting + downgrade.
  Fix 5  — Decision logging: every audit outcome written to Supabase or local JSONL.
"""

import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from groq import Groq
from src.config import Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level risk-tier cache
# ---------------------------------------------------------------------------
_RISK_TIERS: Optional[Dict[str, str]] = None


def _load_risk_tiers() -> Dict[str, str]:
    """Load (and cache) tool_risk_tiers.json."""
    global _RISK_TIERS
    if _RISK_TIERS is not None:
        return _RISK_TIERS

    path: Path = Config.TOOL_RISK_TIERS_PATH
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            # Strip meta / comment keys
            _RISK_TIERS = {k: v for k, v in raw.items() if not k.startswith("_")}
            logger.info("Layer 3: loaded %d tool risk-tier entries", len(_RISK_TIERS))
        except Exception as exc:
            logger.error("Layer 3: failed to load tool_risk_tiers.json — %s; defaulting all to MEDIUM", exc)
            _RISK_TIERS = {"DEFAULT": "MEDIUM"}
    else:
        logger.warning("Layer 3: tool_risk_tiers.json not found at %s; all tools default to MEDIUM", path)
        _RISK_TIERS = {"DEFAULT": "MEDIUM"}

    return _RISK_TIERS


# ---------------------------------------------------------------------------
# ToolCallAuditor
# ---------------------------------------------------------------------------

class ToolCallAuditor:
    """Audits tool calls using a Groq-hosted LLM as judge.

    Public interface is unchanged from the original:
        auditor.audit(user_intent, tool_call, tool_call_history)
    Extended signature:
        auditor.audit(user_intent, tool_call, tool_call_history, raw_user_message)
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the tool call auditor.

        Args:
            api_key: Groq API key (default: from Config)
            model:   Groq model (default: from Config → openai/gpt-oss-20b)
        """
        self.api_key = api_key or Config.GROQ_API_KEY
        self.model = model or Config.GROQ_MODEL
        self.history_length = Config.TOOL_CALL_HISTORY_LENGTH
        self.voting_calls = Config.AUDIT_VOTING_CALLS  # default 3

        # Lightweight model for intent verification (can be overridden to a cheaper model)
        self.intent_verify_model = (
            Config.AUDIT_INTENT_VERIFY_MODEL or self.model
        )

        if not self.api_key:
            raise ValueError("GROQ_API_KEY is required for Layer 3 auditor")

        self.client = Groq(api_key=self.api_key)
        logger.info("Layer 3: Tool-call auditor initialized with model %s", self.model)

    # ------------------------------------------------------------------
    # Public entry-point
    # ------------------------------------------------------------------

    def audit(
        self,
        user_intent: str,
        tool_call: dict,
        tool_call_history: Optional[List[dict]] = None,
        raw_user_message: Optional[str] = None,
    ) -> dict:
        """
        Audit a proposed tool call against user intent.

        Args:
            user_intent:       Summarized user task / goal.
            tool_call:         Proposed tool call dict {name, parameters}.
            tool_call_history: Recent tool calls for salami-slicing context.
            raw_user_message:  (Optional) The verbatim user message before
                               intent summarisation. Enables Fix 2 cross-check.

        Returns:
            dict with at minimum:
                decision   : "ALLOW" | "BLOCK" | "ESCALATE"
                reason     : one-line explanation
                risk_tier  : resolved tier for this tool
                consensus  : bool | None
                vote_count : str | None  (e.g. "2/3")
                intent_verified : bool | None
        """
        tool_call_history = tool_call_history or []
        tool_name = tool_call.get("name", "unknown")

        # ── Step 1: Resolve risk tier ─────────────────────────────────────
        risk_tier = self._resolve_risk_tier(tool_name)

        # ── Step 2: LOW-risk fast-path ────────────────────────────────────
        if risk_tier == "LOW":
            salami = self._detect_salami_slice(tool_name, tool_call_history)
            if not salami:
                result = {
                    "decision": "ALLOW",
                    "reason": "Low-risk tool — audit skipped",
                    "risk_tier": "LOW",
                    "consensus": None,
                    "vote_count": None,
                    "intent_verified": None,
                }
                self._log_decision(result, user_intent, tool_call, risk_tier)
                return result
            else:
                logger.warning(
                    "Layer 3: Salami-slicing detected for LOW-risk tool '%s' — escalating to full audit",
                    tool_name,
                )
                # Fall through to full audit at MEDIUM-equivalent

        # ── Step 3: Intent consistency verification (Fix 2) ───────────────
        intent_verified: Optional[bool] = None
        if raw_user_message:
            try:
                check = self.verify_intent_consistency(raw_user_message, user_intent)
                intent_verified = check["intent_consistent"]
                if not intent_verified:
                    result = {
                        "decision": "ESCALATE",
                        "reason": f"Intent verification failed: {check['discrepancy']}",
                        "risk_tier": risk_tier,
                        "consensus": None,
                        "vote_count": None,
                        "intent_verified": False,
                    }
                    logger.warning(
                        "Layer 3: Intent mismatch detected — short-circuiting to ESCALATE. Discrepancy: %s",
                        check["discrepancy"],
                    )
                    self._log_decision(result, user_intent, tool_call, risk_tier)
                    return result
            except Exception as exc:
                # Fail closed on intent verification errors too
                logger.error("Layer 3: Intent verification raised — failing closed. Error: %s", exc)
                result = {
                    "decision": "ESCALATE",
                    "reason": f"Intent verification system error — manual review required: {exc}",
                    "risk_tier": risk_tier,
                    "consensus": None,
                    "vote_count": None,
                    "intent_verified": False,
                }
                self._log_decision(result, user_intent, tool_call, risk_tier)
                return result
        else:
            logger.warning(
                "Layer 3: raw_user_message not supplied — intent verification skipped (Fix 2 inactive)"
            )

        # ── Step 4: Run tool-call audit (single or voting) ────────────────
        if risk_tier in ("HIGH", "CRITICAL"):
            # Voting regardless of first-call outcome
            result = self._audit_with_voting(user_intent, tool_call, tool_call_history)
        else:
            # MEDIUM (or salami-upgraded LOW): single call, vote only if ESCALATE / ambiguous
            result = self._audit_single_call(user_intent, tool_call, tool_call_history)
            if result.get("decision") == "ESCALATE" or result.get("_ambiguous"):
                result = self._audit_with_voting(
                    user_intent, tool_call, tool_call_history, first_result=result
                )

        # ── Step 5: CRITICAL downgrade ────────────────────────────────────
        if risk_tier == "CRITICAL" and result.get("decision") == "ALLOW":
            logger.warning(
                "Layer 3: CRITICAL tool '%s' — downgrading consensus ALLOW → ESCALATE for mandatory human sign-off",
                tool_name,
            )
            result["decision"] = "ESCALATE"
            result["reason"] = (
                "CRITICAL-risk action — requires human authorisation even when AI audit passes. "
                + result.get("reason", "")
            )

        # ── Attach metadata ───────────────────────────────────────────────
        result["risk_tier"] = risk_tier
        result.setdefault("consensus", None)
        result.setdefault("vote_count", None)
        result["intent_verified"] = intent_verified

        # ── Step 6: Log decision ──────────────────────────────────────────
        self._log_decision(result, user_intent, tool_call, risk_tier)

        # Clean internal flags before returning
        result.pop("_ambiguous", None)
        return result

    # ------------------------------------------------------------------
    # Fix 2 — Intent Verification
    # ------------------------------------------------------------------

    def verify_intent_consistency(
        self, raw_user_message: str, user_intent: str
    ) -> dict:
        """
        Send a lightweight Groq call to check whether the derived user_intent
        accurately and *conservatively* reflects the raw user message.

        Args:
            raw_user_message: The verbatim text the user typed.
            user_intent:      The summarised intent string derived upstream.

        Returns:
            {"intent_consistent": bool, "discrepancy": str | None}
        """
        prompt = (
            "You are a security verifier. Determine whether the 'Derived intent' "
            "accurately and CONSERVATIVELY reflects ONLY what the 'Raw message' asked for, "
            "with NO additional actions, targets, or scope added.\n\n"
            f"Raw message: {raw_user_message}\n"
            f"Derived intent: {user_intent}\n\n"
            "Answer EXACTLY in this format:\n"
            "CONSISTENT: YES | Reason: <brief>\n"
            "or\n"
            "CONSISTENT: NO | Reason: <what was added or changed>"
        )

        response = self.client.chat.completions.create(
            model=self.intent_verify_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a conservative security verifier. "
                        "Only answer YES if the intent is a strict subset of the raw message."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=128,
        )
        text = response.choices[0].message.content.strip()
        text_upper = text.upper()

        if "CONSISTENT: YES" in text_upper:
            return {"intent_consistent": True, "discrepancy": None}

        # Extract reason
        reason = text
        if "|" in text:
            reason = text.split("|", 1)[1].replace("Reason:", "").strip()

        return {"intent_consistent": False, "discrepancy": reason}

    # ------------------------------------------------------------------
    # Fix 3 — Self-Consistency Voting
    # ------------------------------------------------------------------

    def _audit_with_voting(
        self,
        user_intent: str,
        tool_call: dict,
        tool_call_history: List[dict],
        first_result: Optional[dict] = None,
    ) -> dict:
        """
        Run up to `self.voting_calls` independent LLM audit calls and reach
        consensus via majority vote.
        """
        decisions: List[dict] = []

        # Re-use a first result if the caller already has one
        if first_result:
            clean = {k: v for k, v in first_result.items() if not k.startswith("_")}
            decisions.append(clean)

        needed = self.voting_calls - len(decisions)
        for _ in range(needed):
            result = self._audit_single_call(user_intent, tool_call, tool_call_history)
            clean = {k: v for k, v in result.items() if not k.startswith("_")}
            decisions.append(clean)

        return self._reach_consensus(decisions)

    def _reach_consensus(self, decisions: List[dict]) -> dict:
        """
        Majority-vote across a list of decision dicts.

        Returns:
            dict with merged fields plus:
                consensus  : bool
                vote_count : str (e.g. "2/3")
                all_votes  : list (only when consensus is False)
        """
        counts = Counter(d["decision"] for d in decisions)
        total = len(decisions)
        majority_decision, majority_count = counts.most_common(1)[0]

        if majority_count >= 2:  # ≥ 2/3 majority
            # Return the reason from the first decision that matched the majority
            representative = next(
                d for d in decisions if d["decision"] == majority_decision
            )
            return {
                "decision": majority_decision,
                "reason": representative.get("reason", ""),
                "consensus": True,
                "vote_count": f"{majority_count}/{total}",
            }

        # No majority — escalate
        logger.warning(
            "Layer 3: No consensus across %d audit votes (%s) — escalating",
            total,
            dict(counts),
        )
        return {
            "decision": "ESCALATE",
            "reason": f"No consensus among {total} audit attempts — requires human review",
            "consensus": False,
            "vote_count": "0/3",
            "all_votes": decisions,
        }

    # ------------------------------------------------------------------
    # Core single-call auditor
    # ------------------------------------------------------------------

    def _audit_single_call(
        self,
        user_intent: str,
        tool_call: dict,
        tool_call_history: List[dict],
    ) -> dict:
        """
        One LLM audit call.  Returns the parsed decision dict.
        On any exception, fails closed → ESCALATE (Fix 1).
        """
        history_context = self._format_history(tool_call_history)
        prompt = self._build_audit_prompt(user_intent, tool_call, history_context)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=256,
            )
            decision_text = response.choices[0].message.content.strip()
            decision = self._parse_decision(decision_text)
            logger.info("Layer 3 audit call: %s — %s", decision["decision"], decision["reason"])
            return decision

        except Exception as exc:
            # FIX 1 — Fail closed
            logger.error(
                "LAYER 3 FAILURE - FAILING CLOSED — audit API error: %s", exc
            )
            return {
                "decision": "ESCALATE",
                "reason": f"Audit system error — manual review required: {exc}",
            }

    # ------------------------------------------------------------------
    # Fix 4 — Risk Tier Helpers
    # ------------------------------------------------------------------

    def _resolve_risk_tier(self, tool_name: str) -> str:
        """Look up (or default) the risk tier for a given tool name."""
        tiers = _load_risk_tiers()
        tier = tiers.get(tool_name, tiers.get("DEFAULT", "MEDIUM")).upper()
        valid = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        if tier not in valid:
            logger.warning("Layer 3: Unknown tier '%s' for tool '%s'; defaulting to MEDIUM", tier, tool_name)
            tier = "MEDIUM"
        return tier

    def _detect_salami_slice(self, tool_name: str, history: List[dict]) -> bool:
        """
        Return True if the last `salami_window` tool calls in history are all
        calls to the *same* LOW-risk tool — a potential salami-slicing pattern
        even among individually safe actions.
        """
        window = Config.LOW_RISK_SALAMI_WINDOW
        if len(history) < window:
            return False
        recent = history[-window:]
        return all(c.get("name") == tool_name for c in recent)

    # ------------------------------------------------------------------
    # Fix 5 — Decision Logging
    # ------------------------------------------------------------------

    def _log_decision(
        self,
        decision: dict,
        user_intent: str,
        tool_call: dict,
        risk_tier: str,
    ) -> None:
        """
        Append an audit record to Supabase (if configured) or to a local
        JSONL file.  A "ground_truth" field is left empty for human review
        and later calibration.
        """
        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "user_intent": user_intent,
            "tool_call": tool_call,
            "risk_tier": risk_tier,
            "final_decision": decision.get("decision"),
            "reason": decision.get("reason"),
            "consensus": decision.get("consensus"),
            "vote_count": decision.get("vote_count"),
            "intent_verified": decision.get("intent_verified"),
            "ground_truth": None,  # filled in later by human reviewer
        }

        # --- Supabase path -----------------------------------------------
        if Config.SUPABASE_URL and Config.SUPABASE_KEY:
            try:
                from supabase import create_client  # type: ignore
                sb = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
                sb.table("audit_log").insert(record).execute()
                logger.debug("Layer 3: audit record written to Supabase")
                return
            except Exception as exc:
                logger.error(
                    "Layer 3: Supabase write failed — falling back to local JSONL: %s", exc
                )

        # --- Local JSONL fallback -----------------------------------------
        try:
            log_path: Path = Config.AUDIT_LOG_PATH
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
            logger.debug("Layer 3: audit record appended to %s", log_path)
        except Exception as exc:
            logger.error("Layer 3: local audit log write failed: %s", exc)

    # ------------------------------------------------------------------
    # Prompt helpers (unchanged structure, kept for paper reproducibility)
    # ------------------------------------------------------------------

    def _get_system_prompt(self) -> str:
        return (
            "You are a security auditor for an AI cybersecurity copilot. "
            "Your job is to evaluate whether a proposed tool call aligns with "
            "the user's stated intent.\n\n"
            "The copilot has tool-calling ability and can execute real actions "
            "(scans, file operations, data queries). A successful prompt injection "
            "could trigger destructive actions.\n\n"
            "Evaluate the tool call and respond with:\n"
            "- ALLOW: The action clearly aligns with the user's stated goal\n"
            "- BLOCK: The action contradicts the user's intent or appears malicious\n"
            "- ESCALATE: The action is ambiguous and requires human review\n\n"
            "Response format: DECISION: [ALLOW/BLOCK/ESCALATE] | Reason: [one-line explanation]\n\n"
            "Be conservative — if in doubt, ESCALATE rather than allow."
        )

    def _build_audit_prompt(
        self,
        user_intent: str,
        tool_call: dict,
        history_context: str,
    ) -> str:
        return (
            f"User's stated intent: {user_intent}\n\n"
            f"Recent tool call history (last {self.history_length} actions):\n"
            f"{history_context if history_context else 'No recent tool calls'}\n\n"
            f"Proposed tool call to audit:\n"
            f"{self._format_tool_call(tool_call)}\n\n"
            "Does this proposed tool call align with the user's intent? Consider:\n"
            "- Is this action logically consistent with the stated goal?\n"
            "- Could this be part of a chained attack or salami-slicing?\n"
            "- Are there any red flags in the parameters or target?\n\n"
            "Respond in format: DECISION: [ALLOW/BLOCK/ESCALATE] | Reason: [explanation]"
        )

    def _format_history(self, history: List[dict]) -> str:
        if not history:
            return ""
        formatted = []
        for i, call in enumerate(history[-self.history_length:], 1):
            formatted.append(f"{i}. {self._format_tool_call(call)}")
        return "\n".join(formatted)

    def _format_tool_call(self, tool_call: dict) -> str:
        if not tool_call:
            return "No tool call"
        return f"Tool: {tool_call.get('name', 'unknown')} | Parameters: {tool_call.get('parameters', {})}"

    def _parse_decision(self, response_text: str) -> dict:
        """Parse LLM response → {decision, reason}.  Unknown formats → ESCALATE."""
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

        if decision not in ("ALLOW", "BLOCK", "ESCALATE"):
            decision = "ESCALATE"
            reason = f"Unclear decision format ({reason})"

        # Flag ambiguity so calling code can optionally vote
        ambiguous = "ambiguous" in reason.lower() or "unclear" in reason.lower()

        return {"decision": decision, "reason": reason, "_ambiguous": ambiguous}


# ---------------------------------------------------------------------------
# Singleton helper
# ---------------------------------------------------------------------------

_auditor: Optional[ToolCallAuditor] = None


def get_auditor() -> ToolCallAuditor:
    """Get or create the singleton auditor instance."""
    global _auditor
    if _auditor is None:
        _auditor = ToolCallAuditor()
    return _auditor
