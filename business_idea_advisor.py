"""
Business Idea Evaluator - Agentic AI Application
Supports both Gradio GUI (chatbot style) and CLI modes.
"""

import argparse
import os
import operator
import tempfile
from datetime import datetime
from typing import TypedDict, List, Annotated, Dict, Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langgraph.graph.message import add_messages
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from fpdf import FPDF

load_dotenv()

# ============================================================
# LLM & State
# ============================================================

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)


class State(TypedDict):
    idea: str
    messages: Annotated[List[BaseMessage], add_messages]
    advisor_reports: Annotated[Dict[str, str], operator.or_]
    final_report: str


# ============================================================
# HITL Nodes
# ============================================================

system_message = SystemMessage(content="""
You are a helpful tool that evaluates business ideas.

Your task is to decide whether you have enough information about the business ideas.

If not, ask One precise follow-up question to get more information about the business idea. Do not ask more than one question at a time. 
If yes, say: DONE
""")


def assistant(state: State) -> State:
    response = llm.invoke([system_message] + state["messages"])
    return State(messages=[response])


def routing_function(state: State) -> Literal['advisors_start_node', 'ask_user_node']:
    if state["messages"][-1].content.strip().upper().startswith("DONE"):
        return "advisors_start_node"
    else:
        return "ask_user_node"


# ============================================================
# Expert Advisors
# ============================================================

def market_analyst_advisor(state: State) -> State:
    prompt = f"""
    You are a helpful senior MARKET ANALYST that gets an idea as an input. You need to evaluate the market potential, competition, target customers.
    Conduct market sizing and competitor analysis research for the business idea.
    Identify target customer segments and their needs.
    Provide insights on market trends and dynamics that could impact the success of the business idea.
    Assess timing, trends, and macroeconomic factors that could influence the business idea's success.
    Provide a comprehensive report on the market potential of the business idea, including opportunities and challenges.
    
    Idea: 
    {state["messages"]}
    """
    advisor_report = llm.invoke([SystemMessage(content=prompt)])
    return State(advisor_reports={"market_analyst": advisor_report.content})


def legal_advisor(state: State) -> State:
    prompt = f"""
    You are a helpful senior LEGAL ADVISOR that gets an idea as an input. You need to evaluate the legal aspects of the business idea.
    Identify potential legal risks, regulatory requirements, and compliance issues.
    Assess intellectual property considerations, including patents, trademarks, and copyrights.
    Evaluate liability concerns and suggest appropriate legal structures for the business.
    Analyze industry-specific regulations and licensing requirements.
    Provide a comprehensive report on the legal landscape, including potential challenges and recommendations.
    
    Idea: 
    {state["messages"]}
    """
    advisor_report = llm.invoke([SystemMessage(content=prompt)])
    return State(advisor_reports={"legal_advisor": advisor_report.content})


def technical_advisor(state: State) -> State:
    prompt = f"""
    You are a helpful senior TECHNICAL ADVISOR that gets an idea as an input. You need to evaluate the technical feasibility of the business idea.
    Assess the technology stack required to build and scale the product or service.
    Identify technical challenges, risks, and dependencies.
    Evaluate the development timeline, resource requirements, and infrastructure needs.
    Analyze scalability, security, and performance considerations.
    Provide a comprehensive report on the technical viability of the business idea, including recommendations and potential roadblocks.
    
    Idea: 
    {state["messages"]}
    """
    advisor_report = llm.invoke([SystemMessage(content=prompt)])
    return State(advisor_reports={"technical_advisor": advisor_report.content})


def strategist_advisor(state: State) -> State:
    prompt = f"""
    You are a helpful senior BUSINESS STRATEGIST that gets an idea as an input. You need to evaluate the strategic viability of the business idea.
    Assess the business model, revenue streams, and monetization strategies.
    Evaluate the go-to-market strategy, partnerships, and distribution channels.
    Identify competitive advantages, differentiation factors, and barriers to entry.
    Analyze growth potential, scalability of the business model, and long-term sustainability.
    Provide a comprehensive report on the strategic outlook of the business idea, including actionable recommendations and key success factors.
    
    Idea: 
    {state["messages"]}
    """
    advisor_report = llm.invoke([SystemMessage(content=prompt)])
    return State(advisor_reports={"strategist_advisor": advisor_report.content})


# ============================================================
# Final Report
# ============================================================

def final_report_consultant(state: State) -> State:
    if len(state["advisor_reports"]) < 4:
        return {}

    report_prompt = f"""
    You are a senior consultant that gets 4 reports from different advisors (market analyst, legal advisor, technical advisor, business strategist) as input.
    Your task is to synthesize the information from these reports and provide a comprehensive evaluation of the business idea.
    Analyze the strengths, weaknesses, opportunities, and threats (SWOT analysis) based on the insights provided by the advisors.
    Provide an overall assessment of the viability and potential success of the business idea.
    
    The 4 advisor reports are as follows:
    {state["advisor_reports"]}
    """
    final_report = llm.invoke([SystemMessage(content=report_prompt)])
    return State(final_report=final_report.content)


# ============================================================
# Graph Builder
# ============================================================

def build_graph():
    """Build the LangGraph state graph (without HITL nodes for programmatic use)."""
    graph = StateGraph(State)

    graph.add_node("assistant", assistant)
    graph.add_node("advisors_start_node", lambda state: {})
    graph.add_node("market_analyst_advisor", market_analyst_advisor)
    graph.add_node("legal_advisor", legal_advisor)
    graph.add_node("technical_advisor", technical_advisor)
    graph.add_node("strategist_advisor", strategist_advisor)
    graph.add_node("final_report_consultant", final_report_consultant)

    graph.add_edge(START, "assistant")
    graph.add_conditional_edges(source="assistant", path=routing_function)
    graph.add_edge("advisors_start_node", "market_analyst_advisor")
    graph.add_edge("advisors_start_node", "legal_advisor")
    graph.add_edge("advisors_start_node", "technical_advisor")
    graph.add_edge("advisors_start_node", "strategist_advisor")
    graph.add_edge("market_analyst_advisor", "final_report_consultant")
    graph.add_edge("legal_advisor", "final_report_consultant")
    graph.add_edge("technical_advisor", "final_report_consultant")
    graph.add_edge("strategist_advisor", "final_report_consultant")
    graph.add_edge("final_report_consultant", END)

    return graph


# ============================================================
# PDF Export
# ============================================================

class PDFReport(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "Business Idea Evaluation Report", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def add_section(self, title, content):
        self.add_page()
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(3)
        self.set_font("Helvetica", "", 10)
        # Handle encoding issues
        safe_content = content.encode("latin-1", "replace").decode("latin-1")
        self.multi_cell(0, 5, safe_content)


def export_report_pdf(idea: str, advisor_reports: Dict[str, str], final_report: str) -> str:
    """Export the full evaluation as a PDF. Returns the file path."""
    pdf = PDFReport()
    pdf.alias_nb_pages()

    # Title page
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Business Idea Evaluation", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, f"Idea: {idea}")
    pdf.ln(5)
    pdf.set_font("Helvetica", "I", 9)
    pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", new_x="LMARGIN", new_y="NEXT")

    # Individual advisor reports
    advisor_titles = {
        "market_analyst": "Market Analyst Report",
        "legal_advisor": "Legal Advisor Report",
        "technical_advisor": "Technical Advisor Report",
        "strategist_advisor": "Business Strategist Report",
    }
    for key, title in advisor_titles.items():
        if key in advisor_reports:
            pdf.add_section(title, advisor_reports[key])

    # Final report
    pdf.add_section("Final Consolidated Report", final_report)

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"business_idea_report_{timestamp}.pdf"
    output_dir = os.path.join(os.path.dirname(__file__), "reports")
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    pdf.output(filepath)
    return filepath


# ============================================================
# CLI Mode
# ============================================================

def run_cli():
    """Run the evaluator in CLI (terminal) mode."""
    print("\n" + "=" * 60)
    print("   BUSINESS IDEA EVALUATOR - CLI MODE")
    print("=" * 60)

    while True:
        print("\n\nWhat is your business idea?")
        idea = input("You: ").strip()
        if not idea:
            print("Please enter a business idea.")
            continue

        messages = [HumanMessage(content=idea)]

        # HITL loop
        while True:
            response = llm.invoke([system_message] + messages)
            messages.append(response)

            if response.content.strip().upper().startswith("DONE"):
                print("\n[Sufficient information gathered. Running advisors...]")
                break
            else:
                print(f"\nAssistant: {response.content}\n")
                user_input = input("You: ").strip()
                messages.append(HumanMessage(content=user_input))

        # Run advisors
        state = State(
            idea=idea,
            messages=messages,
            advisor_reports={},
            final_report=""
        )

        print("\n[Running Market Analyst...]")
        state = {**state, **market_analyst_advisor(state)}
        print("[Running Legal Advisor...]")
        state = {**state, **legal_advisor(state)}
        print("[Running Technical Advisor...]")
        state = {**state, **technical_advisor(state)}
        print("[Running Business Strategist...]")
        state = {**state, **strategist_advisor(state)}
        print("[Generating Final Report...]\n")
        state = {**state, **final_report_consultant(state)}

        print("\n" + "=" * 60)
        print("   FINAL REPORT")
        print("=" * 60)
        print(state["final_report"])

        # Export option
        export = input("\n\nWould you like to export the report as PDF? (yes/no): ").strip().lower()
        if export in ("yes", "y"):
            filepath = export_report_pdf(idea, state["advisor_reports"], state["final_report"])
            print(f"\nReport exported to: {filepath}")

        # Ask again
        again = input("\n\nWould you like to evaluate another business idea? (yes/no): ").strip().lower()
        if again not in ("yes", "y"):
            print("\nThank you for using the Business Idea Evaluator. Goodbye!")
            break


# ============================================================
# Gradio GUI Mode
# ============================================================

def run_gui():
    """Run the evaluator in Gradio GUI (chatbot style) mode."""
    import gradio as gr

    # Session state stored in closure
    session_state = {
        "messages": [],
        "idea": "",
        "phase": "gathering",  # gathering | evaluating | done
        "advisor_reports": {},
        "final_report": "",
        "history": [],  # list of past evaluations
    }

    def reset_session():
        session_state["messages"] = []
        session_state["idea"] = ""
        session_state["phase"] = "gathering"
        session_state["advisor_reports"] = {}
        session_state["final_report"] = ""

    def user_message(user_input, chat_history):
        if not user_input.strip():
            return "", chat_history

        # First message is the idea
        if not session_state["messages"]:
            session_state["idea"] = user_input.strip()

        session_state["messages"].append(HumanMessage(content=user_input.strip()))
        chat_history = chat_history + [{"role": "user", "content": user_input.strip()}]

        return "", chat_history

    def bot_response(chat_history):
        if session_state["phase"] == "done":
            # User wants a new idea
            reset_session()
            chat_history = chat_history + [{"role": "assistant", "content": "Great! What is your new business idea?"}]
            return chat_history

        if session_state["phase"] == "gathering":
            # Run assistant to check if we have enough info
            response = llm.invoke([system_message] + session_state["messages"])
            session_state["messages"].append(response)

            if response.content.strip().upper().startswith("DONE"):
                session_state["phase"] = "evaluating"
                chat_history = chat_history + [{"role": "assistant", "content": "I have enough information. Let me evaluate your business idea with our expert advisors. This may take a moment..."}]

                # Run all advisors
                state = State(
                    idea=session_state["idea"],
                    messages=session_state["messages"],
                    advisor_reports={},
                    final_report=""
                )

                ma = market_analyst_advisor(state)
                la = legal_advisor(state)
                ta = technical_advisor(state)
                sa = strategist_advisor(state)

                session_state["advisor_reports"] = {
                    **ma.get("advisor_reports", {}),
                    **la.get("advisor_reports", {}),
                    **ta.get("advisor_reports", {}),
                    **sa.get("advisor_reports", {}),
                }

                state["advisor_reports"] = session_state["advisor_reports"]
                fr = final_report_consultant(state)
                session_state["final_report"] = fr.get("final_report", "")

                # Save to history
                session_state["history"].append({
                    "idea": session_state["idea"],
                    "advisor_reports": session_state["advisor_reports"].copy(),
                    "final_report": session_state["final_report"],
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })

                # Build report message
                report_msg = "## Final Evaluation Report\n\n"
                report_msg += session_state["final_report"]
                report_msg += "\n\n---\n\n**Would you like to evaluate another business idea?** Just type your new idea to start again!"

                session_state["phase"] = "done"
                chat_history = chat_history + [{"role": "assistant", "content": report_msg}]
            else:
                chat_history = chat_history + [{"role": "assistant", "content": response.content}]

        return chat_history

    def get_history_display():
        if not session_state["history"]:
            return "No evaluations yet."

        output = ""
        for i, entry in enumerate(reversed(session_state["history"]), 1):
            output += f"## Evaluation #{len(session_state['history']) - i + 1}\n"
            output += f"**Idea:** {entry['idea']}\n\n"
            output += f"**Time:** {entry['timestamp']}\n\n"
            output += f"### Final Report\n{entry['final_report']}\n\n"
            output += "---\n\n"
        return output

    def export_latest_pdf():
        if not session_state["history"]:
            return None

        latest = session_state["history"][-1]
        filepath = export_report_pdf(
            latest["idea"],
            latest["advisor_reports"],
            latest["final_report"]
        )
        return filepath

    def export_selected_pdf(eval_index):
        try:
            idx = int(eval_index) - 1
            if 0 <= idx < len(session_state["history"]):
                entry = session_state["history"][idx]
                filepath = export_report_pdf(
                    entry["idea"],
                    entry["advisor_reports"],
                    entry["final_report"]
                )
                return filepath
        except (ValueError, IndexError):
            pass
        return None

    # Build Gradio UI
    with gr.Blocks(title="Business Idea Evaluator") as app:
        gr.Markdown("# 🧠 Business Idea Evaluator\nChat with our AI advisors to evaluate your business idea.")

        with gr.Tabs():
            with gr.TabItem("💬 Chat"):
                chatbot = gr.Chatbot(
                    value=[{"role": "assistant", "content": "Welcome! What is your business idea? Tell me about it and I'll have our expert advisors evaluate it."}],
                    height=500,
                )
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Type your business idea or answer here...",
                        show_label=False,
                        scale=9,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

                with gr.Row():
                    export_btn = gr.Button("📄 Export Latest Report as PDF")
                    export_file = gr.File(label="Download PDF", visible=True)

                # Wire up chat
                send_btn.click(
                    user_message, [msg_input, chatbot], [msg_input, chatbot]
                ).then(
                    bot_response, [chatbot], [chatbot]
                )
                msg_input.submit(
                    user_message, [msg_input, chatbot], [msg_input, chatbot]
                ).then(
                    bot_response, [chatbot], [chatbot]
                )
                export_btn.click(export_latest_pdf, outputs=[export_file])

            with gr.TabItem("📜 History"):
                gr.Markdown("### Past Evaluations")
                history_display = gr.Markdown("No evaluations yet.")
                refresh_btn = gr.Button("🔄 Refresh History")
                refresh_btn.click(get_history_display, outputs=[history_display])

                with gr.Row():
                    eval_num = gr.Number(label="Evaluation # to export", value=1, precision=0)
                    export_hist_btn = gr.Button("📄 Export as PDF")
                    export_hist_file = gr.File(label="Download PDF")

                export_hist_btn.click(export_selected_pdf, inputs=[eval_num], outputs=[export_hist_file])

    app.launch(theme=gr.themes.Soft())


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Business Idea Evaluator")
    parser.add_argument(
        "--mode",
        choices=["gui", "cli"],
        default="gui",
        help="Run mode: 'gui' for Gradio web interface, 'cli' for terminal (default: gui)"
    )
    args = parser.parse_args()

    if args.mode == "cli":
        run_cli()
    else:
        run_gui()
