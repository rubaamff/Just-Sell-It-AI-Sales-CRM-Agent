# Just Sell It — AI Sales & CRM Agent

**Just Sell It** is an AI-assisted Sales and CRM Agent prototype built as a Streamlit application for the Saudi market. It was developed as a capstone project for the **Saudi Digital Academy Agentic AI Bootcamp, delivered in collaboration with WeCloudData**.

The project is built around a **BeamData sales use case**. It supports the early B2B sales process by combining company discovery, filtering, analytics, lead scoring, personalized outreach, Gmail reply handling, opportunity qualification, and proposal generation in one connected workflow.

## Project Overview

Sales and business development teams often need to search for relevant companies, qualify leads, write personalized emails, follow up on replies, and prepare proposals manually. This prototype shows how AI agents can reduce repetitive work while keeping a human user in control of important customer-facing decisions.

The workflow starts with a Saudi company directory and turns it into an actionable sales pipeline. Users can browse and filter companies, score leads, generate campaign emails, classify replies, qualify opportunities, and create proposal-ready documents.

## Key Features

- **Company Directory**: browse, search, and filter Saudi company records.
- **Market Analytics**: analyze sectors, cities, employee sizes, and startup status.
- **Lead Scoring**: rank companies using BeamData-style criteria or custom user criteria.
- **Campaign Agent**: generate personalized B2B outreach emails with subject, body, campaign goal, and suggested service.
- **Email Agent**: connect to Gmail, fetch unread replies, classify reply intent, extract sales signals, and draft responses.
- **Opportunity Agent**: qualify interested prospects using BANT or custom criteria.
- **Proposal Agent**: combine company context, web research, and internal BeamData knowledge to generate proposal content and export it as PDF.
- **Guardrails**: email validation, output validation, score caps, rate limiting, RAG dependency checks, and human approval before sending emails.

## Multi-Agent Workflow

1. **Lead Generation** — starts from a Saudi company dataset and supports filtering by sector, city, company size, and startup status.
2. **Lead Scoring Agent** — scores companies based on fit, sector, size, location, digital presence, and selected criteria.
3. **Campaign Agent** — generates personalized outreach emails based on company context and BeamData-style services.
4. **Email Agent** — reads Gmail replies, classifies intent, extracts signals, and drafts follow-up responses.
5. **Opportunity Agent** — evaluates interested companies using BANT: Budget, Authority, Need, and Timeline.
6. **Proposal Agent** — retrieves BeamData knowledge, researches the company, generates proposal content, and exports the result as PDF.

## AI Components Used

- LLM prompting for generation, classification, scoring explanations, reply drafting, and proposal writing.
- Structured JSON outputs for predictable fields such as score, status, reason, next step, subject, and body.
- RAG with FAISS for proposal grounding using BeamData PDF knowledge files.
- LangChain tool use for proposal research and retrieval.
- Guardrails for safer and more reliable customer-facing workflows.

## Project Structure

```text
.
├── app.py                         # Main Streamlit application
├── companies.json                 # Saudi company dataset
├── agents/
│   ├── campaign_agent.py           # Campaign email generation
│   ├── email_agent.py              # Gmail reply handling and classification
│   ├── opportunity_agent.py        # BANT/custom opportunity qualification
│   └── proposal_agent.py           # Proposal generation workflow
├── utils/
│   ├── scorer.py                   # Lead scoring logic
│   ├── email_finder.py             # Email discovery helper with rate limiting
│   ├── guardrails.py               # Validation models and safety checks
│   ├── rag.py                      # FAISS-based PDF retrieval
│   └── proposal_pdf.py             # PDF proposal export
├── requirements.txt
├── .env.example
└── .streamlit/config.toml
```

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\\Scripts\\activate     # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy the sample environment file:

```bash
cp .env.example .env
```

Then add your API keys to `.env`:

```env
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=optional_anthropic_api_key_here
```

### 4. Optional: configure Gmail integration

For Gmail reply handling and email sending, add your local Google OAuth files when running the project locally:

```text
credentials.json
token.json
```

These files are ignored by Git and should not be committed.

### 5. Optional: add BeamData knowledge files for RAG

The Proposal Agent can use local BeamData PDF knowledge files for grounded proposal generation. Place the PDF files inside a local `data/` folder:

```text
data/
├── Beamdata Past Project Descriptions.pdf
└── Beam Data AI Hub Intro.pdf
```

The `data/` folder is ignored by Git because it may contain private or internal files.

## Run the Application

```bash
streamlit run app.py
```

The app will open in your browser and display the Just Sell It dashboard.

## Notes

- Customer-facing emails should always be reviewed before sending.
- Lead scores and opportunity scores are decision-support signals, not guaranteed predictions.
- Proposal outputs should be reviewed by a human before being shared with a client.
- The current dataset is suitable for prototype/demo purposes and should be verified before production outreach.

## Team

Group RCP 6:

- Hadi Asiri
- Abdulrahman Hassan
- Nedaa Bajabir
- Ruba Alfahidah
