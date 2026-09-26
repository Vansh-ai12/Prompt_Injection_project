"""
tests/test_layer3_auditor.py
────────────────────────────
Unit and integration tests for the upgraded ToolCallAuditor (5 fixes).

All Groq API calls are mocked — no real network traffic or API keys needed.
"""

import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

os.environ.setdefault("GROQ_API_KEY", "mock_groq_api_key_for_testing")

from layer3_auditor import ToolCallAuditor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_groq_response(text: str):
    """Build a mock object that looks like a Groq chat-completion response."""
    choice = SimpleNamespace(message=SimpleNamespace(content=text))
    return SimpleNamespace(choices=[choice])


def _allow_response():
    return _mock_groq_response("DECISION: ALLOW | Reason: Action aligns with user intent")


def _block_response():
    return _mock_groq_response("DECISION: BLOCK | Reason: Action contradicts user intent")


def _escalate_response():
    return _mock_groq_response("DECISION: ESCALATE | Reason: Ambiguous — requires human review")


def _make_auditor(**kwargs) -> ToolCallAuditor:
    """Instantiate ToolCallAuditor without a real Groq connection."""
    with patch("layer3_auditor.Groq"):
        auditor = ToolCallAuditor(api_key="fake-key", **kwargs)
    return auditor


# ---------------------------------------------------------------------------
# Fix 1 — Fail Closed on API errors
# ---------------------------------------------------------------------------

class TestFix1FailClosed:
    """API errors must result in ESCALATE, never ALLOW."""

    def test_groq_exception_returns_escalate(self):
        auditor = _make_auditor()
        auditor.client.chat.completions.create.side_effect = RuntimeError("Simulated timeout")

        result = auditor.audit(
            user_intent="scan example.com",
            tool_call={"name": "run_scan", "parameters": {"target": "example.com"}},
        )

        assert result["decision"] == "ESCALATE", (
            f"Expected ESCALATE on API failure, got {result['decision']}"
        )
        assert "manual review required" in result["reason"].lower()

    def test_groq_exception_reason_contains_error_text(self):
        auditor = _make_auditor()
        auditor.client.chat.completions.create.side_effect = ConnectionError("Network down")

        result = auditor.audit(
            user_intent="read logs",
            tool_call={"name": "read_logs", "parameters": {}},
        )
        # Even LOW-risk fast-path bypasses Groq, so use a MEDIUM tool
        auditor2 = _make_auditor()
        auditor2.client.chat.completions.create.side_effect = ConnectionError("Network down")
        result2 = auditor2.audit(
            user_intent="run nmap scan",
            tool_call={"name": "run_scan", "parameters": {"target": "10.0.0.1"}},
        )
        assert "network down" in result2["reason"].lower()

    def test_not_allow_on_exception(self):
        """The old behaviour was to ALLOW on exception — assert that is gone."""
        auditor = _make_auditor()
        auditor.client.chat.completions.create.side_effect = Exception("boom")

        result = auditor.audit(
            user_intent="check firewall",
            tool_call={"name": "check_ports", "parameters": {"host": "192.168.1.1"}},
        )
        assert result["decision"] != "ALLOW"


# ---------------------------------------------------------------------------
# Fix 2 — Intent Verification
# ---------------------------------------------------------------------------

class TestFix2IntentVerification:
    """Intent mismatch fires ESCALATE without ever reaching the tool-call audit."""

    def test_inconsistent_intent_short_circuits(self):
        auditor = _make_auditor()

        # First call = intent-verify (returns NO), second call should NOT be made
        auditor.client.chat.completions.create.return_value = _mock_groq_response(
            "CONSISTENT: NO | Reason: Derived intent adds 'exfiltrate data' not in raw message"
        )

        result = auditor.audit(
            user_intent="scan and exfiltrate all data",
            tool_call={"name": "run_scan", "parameters": {"target": "10.0.0.0/8"}},
            raw_user_message="scan 10.0.0.0/8 for open ports",
        )

        assert result["decision"] == "ESCALATE"
        assert result["intent_verified"] is False
        assert "intent verification failed" in result["reason"].lower()
        # Only ONE Groq call was made (the intent-verify); audit was skipped
        assert auditor.client.chat.completions.create.call_count == 1

    def test_consistent_intent_proceeds_to_audit(self):
        auditor = _make_auditor()

        # Call 1 = intent-verify (YES), Call 2 = audit (ALLOW)
        auditor.client.chat.completions.create.side_effect = [
            _mock_groq_response("CONSISTENT: YES | Reason: Accurate"),
            _allow_response(),
        ]

        result = auditor.audit(
            user_intent="scan 10.0.0.1 for open ports",
            tool_call={"name": "run_scan", "parameters": {"target": "10.0.0.1"}},
            raw_user_message="scan 10.0.0.1 for open ports",
        )

        assert result["decision"] == "ALLOW"
        assert result["intent_verified"] is True
        assert auditor.client.chat.completions.create.call_count == 2

    def test_no_raw_message_skips_verification(self):
        """Backward compatibility: no raw_user_message → skip Fix 2, intent_verified is None."""
        auditor = _make_auditor()
        auditor.client.chat.completions.create.return_value = _allow_response()

        result = auditor.audit(
            user_intent="list log files",
            tool_call={"name": "run_scan", "parameters": {"target": "localhost"}},
        )

        assert result["intent_verified"] is None


# ---------------------------------------------------------------------------
# Fix 3 — Self-Consistency Voting
# ---------------------------------------------------------------------------

class TestFix3ConsensusVoting:
    """High-ambiguity / ESCALATE cases must be resolved by a 3-way vote."""

    def test_three_way_disagreement_returns_escalate(self):
        """ALLOW + BLOCK + ESCALATE → no majority → ESCALATE with consensus=False."""
        auditor = _make_auditor()

        # Intent verification YES, then 3 disagreeing audit responses
        auditor.client.chat.completions.create.side_effect = [
            _mock_groq_response("CONSISTENT: YES | Reason: OK"),
            _allow_response(),
            _block_response(),
            _escalate_response(),
        ]

        # HIGH-risk tool triggers automatic voting
        result = auditor.audit(
            user_intent="send report email",
            tool_call={"name": "send_email", "parameters": {"to": "admin@corp.com"}},
            raw_user_message="send the scan report to admin@corp.com",
        )

        assert result["decision"] == "ESCALATE"
        assert result["consensus"] is False
        assert result["vote_count"] == "0/3"
        assert "all_votes" in result

    def test_two_of_three_majority_wins(self):
        """ALLOW + ALLOW + ESCALATE → majority ALLOW."""
        auditor = _make_auditor()

        auditor.client.chat.completions.create.side_effect = [
            _mock_groq_response("CONSISTENT: YES | Reason: OK"),
            _allow_response(),
            _allow_response(),
            _escalate_response(),
        ]

        result = auditor.audit(
            user_intent="send report email",
            tool_call={"name": "send_email", "parameters": {"to": "admin@corp.com"}},
            raw_user_message="send the scan report to admin@corp.com",
        )

        # HIGH-risk + ALLOW consensus → not downgraded (only CRITICAL is)
        assert result["decision"] == "ALLOW"
        assert result["consensus"] is True
        assert result["vote_count"] == "2/3"

    def test_escalate_on_medium_triggers_revote(self):
        """For MEDIUM risk: first call ESCALATE → automatically re-votes 3 times total."""
        auditor = _make_auditor()

        auditor.client.chat.completions.create.side_effect = [
            # intent verify
            _mock_groq_response("CONSISTENT: YES | Reason: OK"),
            # first audit → ESCALATE (triggers voting)
            _escalate_response(),
            # 2nd and 3rd votes
            _escalate_response(),
            _escalate_response(),
        ]

        result = auditor.audit(
            user_intent="run a port scan",
            tool_call={"name": "run_scan", "parameters": {"target": "10.0.0.1"}},
            raw_user_message="run a port scan on 10.0.0.1",
        )

        # 3 ESCALATE = unanimous ESCALATE
        assert result["decision"] == "ESCALATE"
        assert result["consensus"] is True
        assert result["vote_count"] == "3/3"


# ---------------------------------------------------------------------------
# Fix 4 — Risk-Tiered Auditing
# ---------------------------------------------------------------------------

class TestFix4RiskTiers:
    """LOW-risk auto-allows; CRITICAL calls are downgraded even on ALLOW consensus."""

    def test_low_risk_auto_allows_no_api_call(self):
        """A LOW-risk tool must ALLOW immediately WITHOUT calling the Groq API."""
        auditor = _make_auditor()

        result = auditor.audit(
            user_intent="read the application logs",
            tool_call={"name": "read_logs", "parameters": {"path": "/var/log/app.log"}},
        )

        assert result["decision"] == "ALLOW"
        assert result["risk_tier"] == "LOW"
        assert "audit skipped" in result["reason"].lower()
        # Critically: Groq was never called
        auditor.client.chat.completions.create.assert_not_called()

    def test_critical_risk_downgrades_allow_to_escalate(self):
        """CRITICAL tool: even if 3 calls all say ALLOW, final decision must be ESCALATE."""
        auditor = _make_auditor()

        # intent verify + 3 voting calls all say ALLOW
        auditor.client.chat.completions.create.side_effect = [
            _mock_groq_response("CONSISTENT: YES | Reason: OK"),
            _allow_response(),
            _allow_response(),
            _allow_response(),
        ]

        result = auditor.audit(
            user_intent="delete old backup files",
            tool_call={"name": "delete_file", "parameters": {"path": "/backups/old/"}},
            raw_user_message="delete old backup files under /backups/old/",
        )

        assert result["decision"] == "ESCALATE", (
            "CRITICAL tool ALLOW consensus must be downgraded to ESCALATE"
        )
        assert result["risk_tier"] == "CRITICAL"
        # Must still have done voting (consensus should be True from ALLOW votes)
        assert result["consensus"] is True

    def test_unknown_tool_defaults_to_medium(self):
        """An unlisted tool must default to MEDIUM risk, not LOW."""
        auditor = _make_auditor()
        auditor.client.chat.completions.create.side_effect = [
            _allow_response(),
        ]

        result = auditor.audit(
            user_intent="run unknown tool",
            tool_call={"name": "completely_unknown_tool_xyz", "parameters": {}},
        )

        assert result["risk_tier"] == "MEDIUM"
        # Should have made at least one API call (not auto-allowed as LOW)
        auditor.client.chat.completions.create.assert_called()

    def test_low_risk_salami_slice_triggers_full_audit(self):
        """3 consecutive LOW-risk calls to the same tool → full audit on 4th."""
        auditor = _make_auditor()

        history = [
            {"name": "read_logs", "parameters": {"path": "/a"}},
            {"name": "read_logs", "parameters": {"path": "/b"}},
            {"name": "read_logs", "parameters": {"path": "/c"}},
        ]

        auditor.client.chat.completions.create.return_value = _allow_response()

        result = auditor.audit(
            user_intent="read log files",
            tool_call={"name": "read_logs", "parameters": {"path": "/d"}},
            tool_call_history=history,
        )

        # Salami-slicing detected — must have called Groq at least once
        auditor.client.chat.completions.create.assert_called()

    def test_high_risk_always_votes(self):
        """HIGH-risk tool triggers 3-call voting even when first result is ALLOW."""
        auditor = _make_auditor()

        # No raw_user_message → skip intent verify; 3 audit calls
        auditor.client.chat.completions.create.side_effect = [
            _allow_response(),
            _allow_response(),
            _allow_response(),
        ]

        result = auditor.audit(
            user_intent="block a device",
            tool_call={"name": "block_device", "parameters": {"device_id": "dev-42"}},
        )

        assert result["risk_tier"] == "HIGH"
        assert result["consensus"] is True
        assert result["vote_count"] == "3/3"
        # 3 Groq calls were made
        assert auditor.client.chat.completions.create.call_count == 3


# ---------------------------------------------------------------------------
# Fix 5 — Decision Logging
# ---------------------------------------------------------------------------

class TestFix5DecisionLogging:
    """Every audit() call must write a record to the audit log."""

    def test_audit_writes_log_record(self, tmp_path):
        """A completed audit should produce a JSONL record with all required fields."""
        log_file = tmp_path / "audit_log.jsonl"

        auditor = _make_auditor()
        auditor.client.chat.completions.create.return_value = _allow_response()

        with patch("layer3_auditor.Config") as mock_cfg:
            mock_cfg.GROQ_API_KEY = "fake"
            mock_cfg.GROQ_MODEL = "test-model"
            mock_cfg.TOOL_CALL_HISTORY_LENGTH = 3
            mock_cfg.AUDIT_VOTING_CALLS = 3
            mock_cfg.AUDIT_INTENT_VERIFY_MODEL = ""
            mock_cfg.LOW_RISK_SALAMI_WINDOW = 3
            mock_cfg.SUPABASE_URL = None
            mock_cfg.SUPABASE_KEY = None
            mock_cfg.AUDIT_LOG_PATH = log_file
            mock_cfg.TOOL_RISK_TIERS_PATH = Path("data/tool_risk_tiers.json")

            auditor._log_decision(
                decision={
                    "decision": "ALLOW",
                    "reason": "Test allow",
                    "risk_tier": "MEDIUM",
                    "consensus": None,
                    "vote_count": None,
                    "intent_verified": True,
                },
                user_intent="run port scan",
                tool_call={"name": "run_scan", "parameters": {}},
                risk_tier="MEDIUM",
            )

        assert log_file.exists(), "Audit log file was not created"
        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 1

        record = json.loads(lines[0])
        required_fields = {
            "timestamp", "user_intent", "tool_call", "risk_tier",
            "final_decision", "reason", "consensus", "vote_count",
            "intent_verified", "ground_truth",
        }
        missing = required_fields - set(record.keys())
        assert not missing, f"Missing fields in log record: {missing}"

        assert record["ground_truth"] is None, "ground_truth must start as None"
        assert record["final_decision"] == "ALLOW"
        assert record["risk_tier"] == "MEDIUM"

    def test_multiple_audits_append_to_log(self, tmp_path):
        """Multiple audit calls must append separate records."""
        log_file = tmp_path / "audit_log.jsonl"

        tool_call = {"name": "run_scan", "parameters": {"target": "10.0.0.1"}}
        decision_template = {
            "decision": "ALLOW", "reason": "OK",
            "risk_tier": "MEDIUM", "consensus": None,
            "vote_count": None, "intent_verified": None,
        }

        auditor = _make_auditor()

        with patch("layer3_auditor.Config") as mock_cfg:
            mock_cfg.SUPABASE_URL = None
            mock_cfg.SUPABASE_KEY = None
            mock_cfg.AUDIT_LOG_PATH = log_file

            for _ in range(3):
                auditor._log_decision(decision_template, "user intent", tool_call, "MEDIUM")

        lines = [l for l in log_file.read_text().strip().split("\n") if l]
        assert len(lines) == 3, f"Expected 3 log records, got {len(lines)}"


# ---------------------------------------------------------------------------
# End-to-end: HIGH-risk tool full trace
# ---------------------------------------------------------------------------

class TestHighRiskEndToEnd:
    """
    Worked example: HIGH-risk tool going through intent verification,
    3-call voting, and logging.
    """

    def test_high_risk_full_trace(self, tmp_path):
        log_file = tmp_path / "audit_log.jsonl"
        auditor = _make_auditor()

        # Call sequence:
        #   [0] intent verify → YES
        #   [1] vote 1 → ALLOW
        #   [2] vote 2 → ALLOW
        #   [3] vote 3 → ESCALATE
        auditor.client.chat.completions.create.side_effect = [
            _mock_groq_response("CONSISTENT: YES | Reason: Intent accurately reflects message"),
            _allow_response(),
            _allow_response(),
            _escalate_response(),
        ]

        with patch("layer3_auditor.Config") as mock_cfg:
            mock_cfg.GROQ_API_KEY = "fake"
            mock_cfg.GROQ_MODEL = "test-model"
            mock_cfg.AUDIT_INTENT_VERIFY_MODEL = "test-model"
            mock_cfg.TOOL_CALL_HISTORY_LENGTH = 3
            mock_cfg.AUDIT_VOTING_CALLS = 3
            mock_cfg.LOW_RISK_SALAMI_WINDOW = 3
            mock_cfg.SUPABASE_URL = None
            mock_cfg.SUPABASE_KEY = None
            mock_cfg.AUDIT_LOG_PATH = log_file
            mock_cfg.TOOL_RISK_TIERS_PATH = Path("data/tool_risk_tiers.json")

            result = auditor.audit(
                user_intent="send the vulnerability report to the security team",
                tool_call={"name": "send_email", "parameters": {"to": "sec-team@corp.com"}},
                raw_user_message="send the vulnerability report to sec-team@corp.com",
            )

        # 2 ALLOW + 1 ESCALATE → ALLOW wins 2/3
        assert result["decision"] == "ALLOW"
        assert result["risk_tier"] == "HIGH"
        assert result["consensus"] is True
        assert result["vote_count"] == "2/3"
        assert result["intent_verified"] is True
        # 4 total Groq calls made
        assert auditor.client.chat.completions.create.call_count == 4

        # Log record was written
        assert log_file.exists()
        record = json.loads(log_file.read_text().strip())
        assert record["final_decision"] == "ALLOW"
        assert record["risk_tier"] == "HIGH"
        assert record["ground_truth"] is None
