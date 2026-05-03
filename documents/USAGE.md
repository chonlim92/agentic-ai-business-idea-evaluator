# Usage Guide

**Author: Chong Kiat Lim**

## Prerequisites

1. Python 3.10+
2. OpenAI API key
3. (Optional) LangSmith API key for tracing

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd agentic-ai-business-idea-evaluator

# Create and activate virtual environment
python -m venv .agenticai_venv
.agenticai_venv\Scripts\activate  # Windows
# source .agenticai_venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root:

```env
OPENAI_API_KEY="your_openai_api_key_here"

# Optional: LangSmith tracing
LANGCHAIN_API_KEY="your_langsmith_api_key_here"
LANGSMITH_API_KEY="your_langsmith_api_key_here"
LANGSMITH_TRACING=true
LANGCHAIN_TRACING_V2=true
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_PROJECT="business-idea-evaluator"
```

## Running the Application

### GUI Mode (Gradio Web Interface)

```bash
python business_idea_advisor.py --mode gui
```

This launches a web-based chatbot interface at `http://localhost:7860`.

![GUI Chat Interface](images/example_gui_chat.jpg)

**How to use the GUI:**

1. Type your business idea in the text box and click **Send**
2. The assistant will ask follow-up questions — answer them to provide more context
3. Once enough information is gathered, the system automatically runs all 4 expert advisors
4. The **Final Evaluation Report** appears in the chat with a full SWOT analysis
5. After the report, you can type a new idea to start another evaluation
6. Click **Export Latest Report as PDF** to download the report

**History Tab:**

![GUI History Tab](images/example_gui_history.jpg)

- Switch to the **History** tab to view all past evaluations
- Click **Refresh History** to update the display
- Export any past evaluation by entering its number and clicking **Export as PDF**

### CLI Mode (Terminal)

```bash
python business_idea_advisor.py --mode cli
```

**How to use the CLI:**

1. Enter your business idea when prompted
2. Answer follow-up questions from the assistant
3. Wait for all 4 advisors to generate their reports
4. Read the final consolidated report in the terminal
5. Choose whether to export as PDF
6. Choose whether to evaluate another idea

### Example Interaction

```
============================================================
   BUSINESS IDEA EVALUATOR - CLI MODE
============================================================

What is your business idea?
You: AI-generated short videos for luxury car brand advertising