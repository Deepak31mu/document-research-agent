import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document


def make_state(**overrides):
    base = {
        "question": "What is X?",
        "documents": [Document(page_content="X is Y.")],
        "draft_answer": "X is Y.",
        "verification_report": "",
        "is_relevant": True,
        "retriever": MagicMock(),
        "retry_count": 0,
    }
    base.update(overrides)
    return base


@pytest.fixture
def workflow():
    with (
        patch("agents.workflow.ResearchAgent"),
        patch("agents.workflow.VerificationAgent"),
        patch("agents.workflow.RelevanceChecker"),
        patch("agents.workflow.EnsembleRetriever"),
    ):
        from agents.workflow import AgentWorkflow
        return AgentWorkflow()


class TestDecideNextStep:
    def test_ends_when_verification_passes(self, workflow):
        state = make_state(
            verification_report="**Supported:** YES\n**Relevant:** YES",
            retry_count=0,
        )
        assert workflow._decide_next_step(state) == "end"

    def test_retries_when_supported_no(self, workflow):
        state = make_state(
            verification_report="Supported: NO\nRelevant: YES",
            retry_count=0,
        )
        assert workflow._decide_next_step(state) == "re_research"

    def test_retries_when_relevant_no(self, workflow):
        state = make_state(
            verification_report="Supported: YES\nRelevant: NO",
            retry_count=0,
        )
        assert workflow._decide_next_step(state) == "re_research"

    def test_ends_at_max_retries(self, workflow):
        from agents.workflow import MAX_RETRIES
        state = make_state(
            verification_report="Supported: NO\nRelevant: NO",
            retry_count=MAX_RETRIES,
        )
        assert workflow._decide_next_step(state) == "end"

    def test_never_exceeds_max_retries(self, workflow):
        from agents.workflow import MAX_RETRIES
        state = make_state(
            verification_report="Supported: NO",
            retry_count=MAX_RETRIES + 5,
        )
        assert workflow._decide_next_step(state) == "end"


class TestDecideAfterRelevanceCheck:
    def test_returns_relevant_when_true(self, workflow):
        assert workflow._decide_after_relevance_check(make_state(is_relevant=True)) == "relevant"

    def test_returns_irrelevant_when_false(self, workflow):
        assert workflow._decide_after_relevance_check(make_state(is_relevant=False)) == "irrelevant"


class TestCheckRelevanceStep:
    def test_can_answer_sets_relevant_true(self, workflow):
        workflow.relevance_checker.check.return_value = "CAN_ANSWER"
        result = workflow._check_relevance_step(make_state())
        assert result["is_relevant"] is True

    def test_partial_sets_relevant_true(self, workflow):
        workflow.relevance_checker.check.return_value = "PARTIAL"
        result = workflow._check_relevance_step(make_state())
        assert result["is_relevant"] is True

    def test_no_match_sets_relevant_false(self, workflow):
        workflow.relevance_checker.check.return_value = "NO_MATCH"
        result = workflow._check_relevance_step(make_state())
        assert result["is_relevant"] is False
        assert "draft_answer" in result


class TestResearchStep:
    def test_increments_retry_count(self, workflow):
        workflow.researcher.generate.return_value = {"draft_answer": "answer"}
        result = workflow._research_step(make_state(retry_count=1))
        assert result["retry_count"] == 2

    def test_returns_draft_answer(self, workflow):
        workflow.researcher.generate.return_value = {"draft_answer": "The answer is 42."}
        result = workflow._research_step(make_state())
        assert result["draft_answer"] == "The answer is 42."
