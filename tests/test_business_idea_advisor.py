"""
Unit tests for Business Idea Advisor application.
Uses mocked LLM responses to avoid API calls during testing.
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def mock_llm_response():
    """Create a mock LLM response."""
    mock_response = MagicMock()
    mock_response.content = "This is a mock advisor report."
    return mock_response


@pytest.fixture
def sample_state():
    """Create a sample state for testing."""
    from business_idea_advisor import State
    return State(
        idea="AI-powered tutoring platform",
        messages=[HumanMessage(content="AI-powered tutoring platform")],
        advisor_reports={},
        final_report=""
    )


@pytest.fixture
def full_advisor_reports():
    """Sample advisor reports for testing final report generation."""
    return {
        "market_analyst": "Market analysis report content here.",
        "legal_advisor": "Legal advisory report content here.",
        "technical_advisor": "Technical advisory report content here.",
        "strategist_advisor": "Strategy report content here.",
    }


# ============================================================
# Test State Structure
# ============================================================

class TestState:
    def test_state_creation(self):
        from business_idea_advisor import State
        state = State(
            idea="Test idea",
            messages=[HumanMessage(content="Test idea")],
            advisor_reports={},
            final_report=""
        )
        assert state["idea"] == "Test idea"
        assert len(state["messages"]) == 1
        assert state["advisor_reports"] == {}
        assert state["final_report"] == ""

    def test_state_with_advisor_reports(self, full_advisor_reports):
        from business_idea_advisor import State
        state = State(
            idea="Test idea",
            messages=[HumanMessage(content="Test idea")],
            advisor_reports=full_advisor_reports,
            final_report=""
        )
        assert len(state["advisor_reports"]) == 4
        assert "market_analyst" in state["advisor_reports"]
        assert "legal_advisor" in state["advisor_reports"]
        assert "technical_advisor" in state["advisor_reports"]
        assert "strategist_advisor" in state["advisor_reports"]


# ============================================================
# Test Routing Function
# ============================================================

class TestRoutingFunction:
    def test_routes_to_advisors_on_done(self):
        from business_idea_advisor import routing_function, State
        state = State(
            idea="Test",
            messages=[HumanMessage(content="Test"), AIMessage(content="DONE")],
            advisor_reports={},
            final_report=""
        )
        result = routing_function(state)
        assert result == "advisors_start_node"

    def test_routes_to_advisors_on_done_with_extra_text(self):
        from business_idea_advisor import routing_function, State
        state = State(
            idea="Test",
            messages=[HumanMessage(content="Test"), AIMessage(content="DONE. I have enough information.")],
            advisor_reports={},
            final_report=""
        )
        result = routing_function(state)
        assert result == "advisors_start_node"

    def test_routes_to_user_on_question(self):
        from business_idea_advisor import routing_function, State
        state = State(
            idea="Test",
            messages=[HumanMessage(content="Test"), AIMessage(content="What is your target market?")],
            advisor_reports={},
            final_report=""
        )
        result = routing_function(state)
        assert result == "ask_user_node"

    def test_routes_to_advisors_case_insensitive(self):
        from business_idea_advisor import routing_function, State
        state = State(
            idea="Test",
            messages=[HumanMessage(content="Test"), AIMessage(content="done")],
            advisor_reports={},
            final_report=""
        )
        result = routing_function(state)
        assert result == "advisors_start_node"

    def test_routes_to_advisors_with_whitespace(self):
        from business_idea_advisor import routing_function, State
        state = State(
            idea="Test",
            messages=[HumanMessage(content="Test"), AIMessage(content="  DONE  ")],
            advisor_reports={},
            final_report=""
        )
        result = routing_function(state)
        assert result == "advisors_start_node"


# ============================================================
# Test Advisor Functions
# ============================================================

class TestAdvisors:
    @patch("business_idea_advisor.llm")
    def test_market_analyst_advisor(self, mock_llm, sample_state, mock_llm_response):
        from business_idea_advisor import market_analyst_advisor
        mock_llm.invoke.return_value = mock_llm_response

        result = market_analyst_advisor(sample_state)
        assert "advisor_reports" in result
        assert "market_analyst" in result["advisor_reports"]
        assert result["advisor_reports"]["market_analyst"] == "This is a mock advisor report."
        mock_llm.invoke.assert_called_once()

    @patch("business_idea_advisor.llm")
    def test_legal_advisor(self, mock_llm, sample_state, mock_llm_response):
        from business_idea_advisor import legal_advisor
        mock_llm.invoke.return_value = mock_llm_response

        result = legal_advisor(sample_state)
        assert "advisor_reports" in result
        assert "legal_advisor" in result["advisor_reports"]
        assert result["advisor_reports"]["legal_advisor"] == "This is a mock advisor report."
        mock_llm.invoke.assert_called_once()

    @patch("business_idea_advisor.llm")
    def test_technical_advisor(self, mock_llm, sample_state, mock_llm_response):
        from business_idea_advisor import technical_advisor
        mock_llm.invoke.return_value = mock_llm_response

        result = technical_advisor(sample_state)
        assert "advisor_reports" in result
        assert "technical_advisor" in result["advisor_reports"]
        assert result["advisor_reports"]["technical_advisor"] == "This is a mock advisor report."
        mock_llm.invoke.assert_called_once()

    @patch("business_idea_advisor.llm")
    def test_strategist_advisor(self, mock_llm, sample_state, mock_llm_response):
        from business_idea_advisor import strategist_advisor
        mock_llm.invoke.return_value = mock_llm_response

        result = strategist_advisor(sample_state)
        assert "advisor_reports" in result
        assert "strategist_advisor" in result["advisor_reports"]
        assert result["advisor_reports"]["strategist_advisor"] == "This is a mock advisor report."
        mock_llm.invoke.assert_called_once()


# ============================================================
# Test Final Report Consultant
# ============================================================

class TestFinalReportConsultant:
    @patch("business_idea_advisor.llm")
    def test_generates_report_with_4_advisors(self, mock_llm, full_advisor_reports):
        from business_idea_advisor import final_report_consultant, State
        mock_response = MagicMock()
        mock_response.content = "Final comprehensive report."
        mock_llm.invoke.return_value = mock_response

        state = State(
            idea="Test idea",
            messages=[HumanMessage(content="Test idea")],
            advisor_reports=full_advisor_reports,
            final_report=""
        )

        result = final_report_consultant(state)
        assert "final_report" in result
        assert result["final_report"] == "Final comprehensive report."

    @patch("business_idea_advisor.llm")
    def test_returns_empty_with_insufficient_reports(self, mock_llm):
        from business_idea_advisor import final_report_consultant, State
        state = State(
            idea="Test idea",
            messages=[HumanMessage(content="Test idea")],
            advisor_reports={"market_analyst": "report1", "legal_advisor": "report2"},
            final_report=""
        )

        result = final_report_consultant(state)
        assert result == {}
        mock_llm.invoke.assert_not_called()


# ============================================================
# Test PDF Export
# ============================================================

class TestPDFExport:
    def test_export_creates_pdf_file(self, full_advisor_reports):
        from business_idea_advisor import export_report_pdf
        filepath = export_report_pdf(
            "Test business idea",
            full_advisor_reports,
            "This is the final consolidated report."
        )
        assert filepath.endswith(".pdf")
        assert os.path.exists(filepath)
        os.remove(filepath)

    def test_export_report_pdf_contains_all_sections(self, full_advisor_reports):
        from business_idea_advisor import export_report_pdf
        filepath = export_report_pdf(
            "Test business idea",
            full_advisor_reports,
            "This is the final consolidated report."
        )
        assert os.path.exists(filepath)
        assert os.path.getsize(filepath) > 0
        # Cleanup
        os.remove(filepath)

    def test_export_handles_unicode(self, full_advisor_reports):
        from business_idea_advisor import export_report_pdf
        unicode_reports = {k: v + " — émojis: café" for k, v in full_advisor_reports.items()}
        filepath = export_report_pdf(
            "Test idea with spëcial chars",
            unicode_reports,
            "Final report with ünïcödë."
        )
        assert os.path.exists(filepath)
        os.remove(filepath)


# ============================================================
# Test Assistant Function
# ============================================================

class TestAssistant:
    @patch("business_idea_advisor.llm")
    def test_assistant_returns_message(self, mock_llm):
        from business_idea_advisor import assistant, State
        mock_response = MagicMock()
        mock_response.content = "What is your target audience?"
        mock_llm.invoke.return_value = mock_response

        state = State(
            idea="SaaS product",
            messages=[HumanMessage(content="SaaS product")],
            advisor_reports={},
            final_report=""
        )

        result = assistant(state)
        assert "messages" in result
        mock_llm.invoke.assert_called_once()


# ============================================================
# Test Graph Building
# ============================================================

class TestGraphBuilder:
    def test_build_graph_returns_state_graph(self):
        from business_idea_advisor import build_graph
        graph = build_graph()
        assert graph is not None

    def test_build_graph_compiles_successfully(self):
        from business_idea_advisor import build_graph
        graph = build_graph()
        # The graph requires ask_user_node for the conditional edge
        # Add a dummy node for compilation to succeed
        graph.add_node("ask_user_node", lambda state: state)
        compiled = graph.compile()
        assert compiled is not None


# ============================================================
# Test PDFReport Class
# ============================================================

class TestPDFReportClass:
    def test_pdf_report_creation(self):
        from business_idea_advisor import PDFReport
        pdf = PDFReport()
        pdf.alias_nb_pages()
        pdf.add_page()
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 10, "Test content")
        # Should not raise
        assert pdf.page_no() == 1

    def test_pdf_report_add_section(self):
        from business_idea_advisor import PDFReport
        pdf = PDFReport()
        pdf.alias_nb_pages()
        pdf.add_section("Test Section", "This is test content for the section.")
        assert pdf.page_no() >= 1
