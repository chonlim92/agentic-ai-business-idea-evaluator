# Implementation

## Architecture Overview

The Business Idea Evaluator is built using a multi-agent architecture powered by LangGraph. The system uses multiple expert AI agents that work in parallel to evaluate business ideas from different perspectives.

### Agent Graph

![Agent Graph](images/graph.png)

## Components

### 1. Human-in-the-Loop (HITL)

The system starts with a conversational assistant that gathers sufficient information about the business idea before routing to the expert advisors. It asks one follow-up question at a time until it determines it has enough context.

**Nodes:**
- `assistant` — Decides if enough information has been gathered
- `ask_user_node` — Prompts the user for more details
- `routing_function` — Routes to advisors when ready, or back to user for more info

### 2. Expert Advisors (Parallel Execution)

Once sufficient information is gathered, the system routes to 4 expert advisors that run **in parallel**:

| Advisor | Role |
|---------|------|
| **Market Analyst** | Evaluates market potential, competition, target customers, market sizing, trends |
| **Legal Advisor** | Identifies legal risks, regulatory requirements, IP considerations, compliance |
| **Technical Advisor** | Assesses technical feasibility, tech stack, scalability, security |
| **Business Strategist** | Evaluates business model, revenue streams, go-to-market strategy, competitive advantages |

### 3. Final Report Consultant

After all 4 advisors complete their analysis, a senior consultant synthesizes the reports into a comprehensive evaluation including:
- SWOT Analysis (Strengths, Weaknesses, Opportunities, Threats)
- Overall viability assessment
- Actionable recommendations

### 4. PDF Export

Reports can be exported as PDF documents containing:
- Title page with idea summary and timestamp
- Individual advisor reports (one section per advisor)
- Final consolidated report

## Technology Stack

| Technology | Purpose |
|-----------|---------|
| **LangChain / LangGraph** | Agent orchestration, state management, parallel execution |
| **OpenAI GPT** | LLM for all advisor agents |
| **Gradio** | Web-based chatbot GUI |
| **fpdf2** | PDF report generation |
| **python-dotenv** | Environment variable management |
| **LangSmith** | Optional tracing and monitoring |

## Project Structure

```
agentic-ai-business-idea-evaluator/
├── business_idea_advisor.py    # Main application (GUI + CLI)
├── BusinessIdeaEvaluator.ipynb # Jupyter notebook (development/exploration)
├── requirements.txt            # Python dependencies
├── .env                        # API keys (not committed)
├── reports/                    # Generated PDF reports
├── documents/
│   ├── IMPLEMENTATION.md       # This file
│   ├── USAGE.md                # Usage guide
│   └── images/
│       ├── graph.png           # Agent graph visualization
│       ├── example_gui_chat.jpg
│       └── example_gui_history.jpg
└── tests/
    └── test_business_idea_advisor.py
```

## Key Design Decisions

1. **Parallel Advisor Execution** — All 4 advisors run simultaneously for faster evaluation
2. **Dual Interface** — Both GUI (Gradio) and CLI modes from a single codebase
3. **Session-based History** — Evaluations are stored in memory for the session duration
4. **Modular Architecture** — Each advisor is an independent function, easy to add/remove advisors
5. **Temperature 0** — Deterministic outputs for consistent business advice
