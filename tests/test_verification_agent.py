import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def agent():
    with patch("agents.verification_agent.OpenAI"):
        from agents.verification_agent import VerificationAgent
        return VerificationAgent()


class TestParseVerificationResponse:
    def test_parses_fully_supported_response(self, agent):
        response = (
            "Supported: YES\n"
            "Unsupported Claims: []\n"
            "Contradictions: []\n"
            "Relevant: YES\n"
            "Additional Details: All facts match the context."
        )
        result = agent.parse_verification_response(response)
        assert result["Supported"] == "YES"
        assert result["Relevant"] == "YES"
        assert result["Unsupported Claims"] == []
        assert result["Contradictions"] == []
        assert result["Additional Details"] == "All facts match the context."

    def test_parses_unsupported_claims_list(self, agent):
        response = (
            "Supported: NO\n"
            "Unsupported Claims: [claim one, claim two]\n"
            "Contradictions: []\n"
            "Relevant: YES\n"
            "Additional Details: Some claims unverified."
        )
        result = agent.parse_verification_response(response)
        assert result["Unsupported Claims"] == ["claim one", "claim two"]

    def test_parses_contradictions_list(self, agent):
        response = (
            "Supported: NO\n"
            "Unsupported Claims: []\n"
            "Contradictions: [value was 5, not 10]\n"
            "Relevant: YES\n"
            "Additional Details: "
        )
        result = agent.parse_verification_response(response)
        assert result["Contradictions"] == ["value was 5", "not 10"]

    def test_fills_defaults_for_missing_keys(self, agent):
        result = agent.parse_verification_response("Supported: NO")
        assert result["Relevant"] == "NO"
        assert result["Unsupported Claims"] == []
        assert result["Contradictions"] == []
        assert result["Additional Details"] == ""

    def test_case_insensitive_keys(self, agent):
        # Keys from LLM may come in mixed case — .title() normalises them
        response = (
            "supported: YES\n"
            "relevant: YES\n"
            "unsupported claims: []\n"
            "contradictions: []\n"
            "additional details: ok"
        )
        result = agent.parse_verification_response(response)
        assert result["Supported"] == "YES"
        assert result["Relevant"] == "YES"

    def test_returns_none_on_unexpected_input(self, agent):
        result = agent.parse_verification_response(None)
        assert result is None

    def test_empty_string_returns_defaults(self, agent):
        result = agent.parse_verification_response("")
        assert result["Supported"] == "NO"
        assert result["Relevant"] == "NO"


class TestFormatVerificationReport:
    def test_formats_all_fields(self, agent):
        verification = {
            "Supported": "YES",
            "Unsupported Claims": [],
            "Contradictions": [],
            "Relevant": "YES",
            "Additional Details": "Looks good.",
        }
        report = agent.format_verification_report(verification)
        assert "**Supported:** YES" in report
        assert "**Relevant:** YES" in report
        assert "**Unsupported Claims:** None" in report
        assert "**Contradictions:** None" in report
        assert "**Additional Details:** Looks good." in report

    def test_formats_non_empty_claims(self, agent):
        verification = {
            "Supported": "NO",
            "Unsupported Claims": ["claim A", "claim B"],
            "Contradictions": ["value mismatch"],
            "Relevant": "YES",
            "Additional Details": "",
        }
        report = agent.format_verification_report(verification)
        assert "claim A, claim B" in report
        assert "value mismatch" in report
        assert "**Additional Details:** None" in report

    def test_report_contains_five_lines(self, agent):
        verification = {
            "Supported": "YES",
            "Unsupported Claims": [],
            "Contradictions": [],
            "Relevant": "YES",
            "Additional Details": "",
        }
        lines = agent.format_verification_report(verification).strip().split("\n")
        assert len(lines) == 5
