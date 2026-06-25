"""
Campaign Agent — LangGraph implementation.

Graph:
    START → validate_input → call_llm → validate_output ─► (ok)   → END
                                              │
                                              ▼ (invalid / retries exhausted)
                                          fallback → END

Guardrails:
  - Input validation : company_name required.
  - Output schema   : Pydantic CampaignEmailOutput (subject/body length, etc.).
  - Retries         : bounded retries on LLM/JSON errors, then a deterministic
                      fallback email so the pipeline never returns nothing.
"""

from __future__ import annotations

import os
import csv
import json
import logging
from typing import Any, Callable, Optional, TypedDict

from dotenv import load_dotenv
from openai import OpenAI

from langgraph.graph import StateGraph, START, END

from utils.guardrails import (
    GuardrailError,
    require_fields,
    validate_output,
    CampaignEmailOutput,
)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MAX_RETRIES = int(os.getenv("CAMPAIGN_MAX_RETRIES", "3"))
MODEL = os.getenv("CAMPAIGN_MODEL", "gpt-4o-mini")

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# BeamData knowledge (unchanged content — only data, not behaviour)
# ─────────────────────────────────────────────────────────────────────

BEAMDATA_PAST_PROJECTS = [
    {"title": "RAG Chatbot & Knowledge Base",
     "description": "AI-powered chatbot using Retrieval-Augmented Generation (RAG) that answers questions from company documents, policies, and knowledge bases. Reduced support tickets by 60%.",
     "sectors": ["Technology", "E-Commerce", "Customer Service", "Telecom", "Banking", "Insurance"]},
    {"title": "Document Verification & Extraction System",
     "description": "Automated document verification using AI/ML to check authenticity, extract structured data from unstructured documents, and reduce manual review time by 80%.",
     "sectors": ["Banking", "FinTech", "Government", "Healthcare", "Insurance", "Real Estate"]},
    {"title": "Healthcare AI Clinical Assistant",
     "description": "Clinical decision support system using AI to assist doctors with diagnosis, drug interaction checks, patient record analysis, and medical coding automation.",
     "sectors": ["Healthcare", "Pharmaceuticals", "Biotech", "Hospitals"]},
    {"title": "Dynamic Pricing Engine",
     "description": "Real-time pricing optimization using machine learning to maximize revenue based on demand signals, competitor pricing, and inventory levels. Increased revenue by 18%.",
     "sectors": ["E-Commerce", "Retail", "Travel", "Hospitality", "FMCG"]},
    {"title": "Marketing Attribution & ROI Model",
     "description": "Multi-touch attribution model that identifies which marketing channels drive the most conversions, enabling smarter ad spend decisions and reducing CAC by 25%.",
     "sectors": ["E-Commerce", "Marketing", "Media", "Retail", "Telecom"]},
    {"title": "Personalized Recommender System",
     "description": "Product and content recommendation engine using collaborative filtering and deep learning. Increased cross-sell revenue by 30% for a major e-commerce client.",
     "sectors": ["E-Commerce", "Media", "Retail", "Streaming", "Publishing"]},
    {"title": "AI-Powered Loan Approval & Credit Scoring",
     "description": "Automated loan approval using AI risk scoring, fraud detection, and alternative credit assessment. Reduced approval time from 5 days to 3 minutes.",
     "sectors": ["FinTech", "Banking", "Financial Services", "Insurance"]},
    {"title": "Predictive Maintenance System",
     "description": "IoT + AI system that predicts equipment failures before they happen using sensor data and anomaly detection, reducing downtime by 40% and maintenance costs by 35%.",
     "sectors": ["Manufacturing", "Energy", "Oil & Gas", "Industrial", "Utilities", "Mining"]},
    {"title": "Customer Segmentation & Churn Prediction",
     "description": "AI-powered customer segmentation grouping customers by behavior, lifetime value, and churn risk. Enabled targeted retention campaigns that reduced churn by 22%.",
     "sectors": ["Telecom", "E-Commerce", "Banking", "Retail", "Insurance"]},
    {"title": "Sales & Demand Forecasting",
     "description": "Machine learning model forecasting sales demand by product, region, and season with 92% accuracy. Helped clients optimize inventory and reduce stockouts by 45%.",
     "sectors": ["Retail", "FMCG", "Manufacturing", "Supply Chain", "Wholesale"]},
    {"title": "Real Estate AI Valuation Platform",
     "description": "AI-powered property valuation, market trend analysis, and investment opportunity scoring. Automated AVM (Automated Valuation Model) for thousands of properties in real time.",
     "sectors": ["Real Estate", "PropTech", "Construction", "Banking"]},
    {"title": "Computer Vision Quality Control",
     "description": "Automated visual inspection system using computer vision to detect product defects on production lines, achieving 99.2% defect detection accuracy and replacing manual inspection.",
     "sectors": ["Manufacturing", "Food & Beverage", "Industrial", "Pharmaceuticals", "Electronics"]},
    {"title": "Fleet & Logistics Route Optimization",
     "description": "AI-based route optimization and fleet distribution management system that reduced delivery times by 28% and fuel costs by 20% for a large logistics operator.",
     "sectors": ["Logistics", "Transportation", "Supply Chain", "E-Commerce", "Retail"]},
    {"title": "Data Warehouse & ETL Automation",
     "description": "Centralized data warehouse with automated ETL pipelines that consolidate data from 15+ sources, enabling a single source of truth and cutting data engineering effort by 70%.",
     "sectors": ["All Industries", "Technology", "Banking", "Retail", "Telecom"]},
    {"title": "Business Intelligence Dashboards",
     "description": "Real-time interactive BI dashboards giving executives and operations teams live visibility into KPIs, sales, logistics, and customer data.",
     "sectors": ["All Industries", "Retail", "Banking", "Manufacturing", "Telecom"]},
    {"title": "Web Scraping & Competitive Intelligence",
     "description": "Automated data collection pipelines for market intelligence, competitor price tracking, and industry research, delivering daily structured datasets.",
     "sectors": ["E-Commerce", "Research", "Market Intelligence", "Retail", "Finance"]},
    {"title": "Digital Process Automation (DPA/RPA)",
     "description": "End-to-end digital automation of paper-based and manual workflows using AI + RPA, cutting operational costs by 50% and processing time by 80%.",
     "sectors": ["Government", "Banking", "Insurance", "HR", "Healthcare", "Telecom"]},
    {"title": "AI Sales CRM Platform",
     "description": "AI-powered CRM with lead scoring, opportunity qualification, campaign automation, and proposal generation. Increased sales team productivity by 3x.",
     "sectors": ["Sales", "B2B", "Technology", "Real Estate", "Financial Services"]},
    {"title": "Smart Agriculture & Crop Intelligence",
     "description": "AI/IoT system for crop health monitoring, yield prediction, and water/fertilizer optimization. Increased crop yield by 35% while reducing resource waste.",
     "sectors": ["Agriculture", "AgriTech", "Food & Beverage"]},
    {"title": "Clinical LLM for Medical Documentation",
     "description": "Large language model fine-tuned on clinical data for medical documentation automation, clinical coding, and research summarization.",
     "sectors": ["Healthcare", "Pharmaceuticals", "Biotech", "Hospitals"]},
    {"title": "AI Hub Enterprise Platform",
     "description": "Comprehensive enterprise AI platform with AI Assistant, Knowledge Hub (RAG-based), and Agentic Workflow automation for end-to-end business process intelligence.",
     "sectors": ["Enterprise", "Technology", "Banking", "Telecom", "Government"]},
    {"title": "AI-Powered Learning Management System (LMS)",
     "description": "AI-enhanced LMS with personalized learning paths, content recommendations, performance analytics, and automated skills gap detection.",
     "sectors": ["Education", "HR", "Training", "Government", "Corporate"]},
]

BEAMDATA_SERVICES = """
BeamData Core Services:
1. Data & AI Strategy — Define your AI roadmap, identify high-ROI use cases, and plan your data transformation.
2. Proof of Concepts (PoC) — Rapidly validate AI/data solutions with working prototypes before committing to full build.
3. Deployment & Integration — Production-ready model deployment, API integration, and system connectivity.
4. Operations & Monitoring — Ongoing model performance monitoring, retraining, and continuous optimization.
5. AI Governance & Compliance — Ensure AI systems are transparent, auditable, and compliant with regulations.

BeamData AI Hub Products:
- AI Assistant: Company-specific intelligent chatbot powered by your documents and knowledge.
- Knowledge Hub: RAG-based document intelligence — search and retrieve answers from any internal content.
- Agentic Workflow: Autonomous AI agents that handle complex multi-step business processes end-to-end.
"""


def _get_relevant_projects(sector: str, sub_sector: str, description: str, tags: str, top_k: int = 3) -> list:
    combined = f"{sector} {sub_sector} {description} {tags}".lower()
    scored = []
    for project in BEAMDATA_PAST_PROJECTS:
        score = 0
        for s in project["sectors"]:
            if s.lower() in combined or any(w in combined for w in s.lower().split()):
                score += 2
            if "All Industries" in project["sectors"]:
                score += 1
        scored.append((score, project))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:top_k]]


# ─────────────────────────────────────────────────────────────────────
# LangGraph state
# ─────────────────────────────────────────────────────────────────────

class CampaignState(TypedDict, total=False):
    company: dict           # normalized company
    prompt: str             # built prompt
    raw_response: str       # LLM JSON string
    parsed: dict            # parsed dict from raw_response
    validated: Optional[CampaignEmailOutput]  # schema-valid output
    attempts: int           # how many LLM tries so far
    error: str              # last error message (used for retry decisions)
    final_email: dict       # final email dict returned to caller


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _build_prompt(company: dict) -> str:
    relevant = _get_relevant_projects(
        company.get("sector", ""),
        company.get("sub_sector", ""),
        company.get("description", ""),
        company.get("tags", ""),
    )
    projects_text = "\n".join(
        f"  - {p['title']}: {p['description']}" for p in relevant
    )

    return f"""
Generate a highly personalized B2B cold email for this company from BeamData.

Company data:
  Company name: {company.get('company_name','')}
  Arabic name: {company.get('arabic_name','')}
  Sector: {company.get('sector','')}
  Sub-sector: {company.get('sub_sector','')}
  City: {company.get('city','')}, {company.get('country','')}
  Employees: {company.get('employees','')}
  Founded: {company.get('founded_year','')}
  Website: {company.get('website','')}
  Description: {company.get('description','')}
  Is startup: {company.get('is_startup','')}
  Is listed: {company.get('is_listed','')}
  Tags: {company.get('tags','')}

BeamData's most relevant PAST PROJECTS for this company:
{projects_text}

BeamData Services:
{BEAMDATA_SERVICES}

Instructions:
- Write in English. Be professional, concise, and warm.
- The email MUST reference 1-2 of the past projects listed above that are most relevant to this specific company's sector and needs. Mention them naturally.
- Identify what this company likely NEEDS based on their sector, description, and tags. Address that specific pain point.
- Suggest 1-2 BeamData services that would best solve their problem.
- Keep the email under 200 words. Goal: schedule a short meeting.
- Do NOT invent fake facts. Do NOT use generic filler.
- Sender: BeamData Team | beamdata.ai

Return JSON only:
{{
  "email_subject": "subject here",
  "email_body": "email body here",
  "campaign_goal": "goal here",
  "suggested_service": "service here",
  "matched_projects": ["project title 1", "project title 2"]
}}
"""


def _fallback_email(company: dict) -> dict:
    relevant = _get_relevant_projects(
        company.get("sector", ""), company.get("sub_sector", ""),
        company.get("description", ""), company.get("tags", ""), top_k=1,
    )
    mention = ""
    if relevant:
        p = relevant[0]
        mention = f"\nWe recently delivered a '{p['title']}' for a client in a similar space — {p['description'][:100]}...\n"

    subject = f"AI Solutions for {company.get('company_name','')} — BeamData"
    body = f"""Dear {company.get('company_name','')} Team,

We've been following the growth of organizations in the {company.get('sector','')} sector in Saudi Arabia, and we believe BeamData can add real value to your operations.
{mention}
At BeamData, we specialize in AI automation, data analytics, and intelligent workflow solutions — purpose-built for companies like yours.

We'd love to schedule a quick 20-minute call to explore how we can support your goals.

Best regards,
BeamData Team
www.beamdata.ai
""".strip()

    return {
        "email_subject": subject,
        "email_body": body,
        "campaign_goal": "Schedule a short meeting",
        "suggested_service": "AI automation and data analytics",
        "matched_projects": [],
    }


# ─────────────────────────────────────────────────────────────────────
# Graph nodes
# ─────────────────────────────────────────────────────────────────────

def _node_validate_input(state: CampaignState) -> CampaignState:
    company = state.get("company", {})
    require_fields(company, ["company_name"], where="company")
    return {"prompt": _build_prompt(company), "attempts": 0}


def _node_call_llm(state: CampaignState) -> CampaignState:
    if not OPENAI_API_KEY:
        raise GuardrailError("Missing OPENAI_API_KEY. Add it inside .env file.")
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=MODEL,
            temperature=0.4,
            messages=[
                {"role": "system",
                 "content": ("You are a professional B2B sales campaign assistant for BeamData, "
                             "an AI and data solutions company. Return valid JSON only. No markdown.")},
                {"role": "user", "content": state["prompt"]},
            ],
        )
        text = response.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        return {"raw_response": text,
                "attempts": state.get("attempts", 0) + 1,
                "error": ""}
    except Exception as exc:
        logger.warning("Campaign LLM call failed: %s", exc)
        return {"error": f"openai: {exc}",
                "raw_response": "",
                "attempts": state.get("attempts", 0) + 1}


def _node_validate_output(state: CampaignState) -> CampaignState:
    raw = state.get("raw_response", "")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {"error": f"json parse: {exc}", "parsed": {}}
    try:
        validated = validate_output(CampaignEmailOutput, parsed)
    except GuardrailError as exc:
        return {"error": str(exc), "parsed": parsed}
    return {"parsed": parsed, "validated": validated, "error": ""}


def _node_assemble(state: CampaignState) -> CampaignState:
    company = state["company"]
    v: CampaignEmailOutput = state["validated"]
    out = {
        "company_name": company.get("company_name", ""),
        "arabic_name": company.get("arabic_name", ""),
        "sector": company.get("sector", ""),
        "sub_sector": company.get("sub_sector", ""),
        "city": company.get("city", ""),
        "country": company.get("country", ""),
        "website": company.get("website", ""),
        "email": company.get("email", ""),
        "phone": company.get("phone", ""),
        "linkedin_url": company.get("linkedin_url", ""),
        "employees": company.get("employees", ""),
        "founded_year": company.get("founded_year", ""),
        "description": company.get("description", ""),
        "tags": company.get("tags", ""),
        "email_subject": v.email_subject,
        "email_body": v.email_body,
        "campaign_goal": v.campaign_goal or "Schedule a short meeting",
        "suggested_service": v.suggested_service or "AI automation and data analytics",
        "matched_projects": ", ".join(v.matched_projects),
    }
    return {"final_email": out}


def _node_fallback(state: CampaignState) -> CampaignState:
    company = state["company"]
    fb = _fallback_email(company)
    final = {
        "company_name": company.get("company_name", ""),
        "arabic_name": company.get("arabic_name", ""),
        "sector": company.get("sector", ""),
        "sub_sector": company.get("sub_sector", ""),
        "city": company.get("city", ""),
        "country": company.get("country", ""),
        "website": company.get("website", ""),
        "email": company.get("email", ""),
        "phone": company.get("phone", ""),
        "linkedin_url": company.get("linkedin_url", ""),
        "employees": company.get("employees", ""),
        "founded_year": company.get("founded_year", ""),
        "description": company.get("description", ""),
        "tags": company.get("tags", ""),
        **fb,
        "matched_projects": "",
    }
    return {"final_email": final}


def _route_after_validate(state: CampaignState) -> str:
    if state.get("validated"):
        return "assemble"
    if state.get("attempts", 0) >= MAX_RETRIES:
        return "fallback"
    return "call_llm"


# ─────────────────────────────────────────────────────────────────────
# Build & cache the compiled graph
# ─────────────────────────────────────────────────────────────────────

def _build_graph():
    g = StateGraph(CampaignState)
    g.add_node("validate_input", _node_validate_input)
    g.add_node("call_llm", _node_call_llm)
    g.add_node("validate_output", _node_validate_output)
    g.add_node("assemble", _node_assemble)
    g.add_node("fallback", _node_fallback)

    g.add_edge(START, "validate_input")
    g.add_edge("validate_input", "call_llm")
    g.add_edge("call_llm", "validate_output")
    g.add_conditional_edges(
        "validate_output", _route_after_validate,
        {"assemble": "assemble", "fallback": "fallback", "call_llm": "call_llm"},
    )
    g.add_edge("assemble", END)
    g.add_edge("fallback", END)
    return g.compile()


_GRAPH = None


def get_campaign_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = _build_graph()
    return _GRAPH


# ─────────────────────────────────────────────────────────────────────
# Public API — preserved for app.py
# ─────────────────────────────────────────────────────────────────────

class CampaignAgent:
    """Backwards-compatible wrapper around the LangGraph implementation.
    `app.py` keeps calling `CampaignAgent().run_on_companies(...)` unchanged."""

    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("Missing OPENAI_API_KEY. Add it inside .env file.")

    def normalize_company(self, company: dict) -> dict:
        return {
            "company_name": company.get("company_name") or company.get("name", ""),
            "arabic_name": company.get("arabic_name", ""),
            "sector": company.get("sector", ""),
            "sub_sector": company.get("sub_sector", ""),
            "city": company.get("city") or company.get("city_clean", ""),
            "country": company.get("country", "Saudi Arabia"),
            "website": company.get("website", ""),
            "email": company.get("email", ""),
            "phone": company.get("phone", ""),
            "linkedin_url": company.get("linkedin_url", ""),
            "employees": company.get("employees", ""),
            "founded_year": company.get("founded_year", ""),
            "description": company.get("description", ""),
            "is_startup": company.get("is_startup", ""),
            "is_listed": company.get("is_listed", ""),
            "tags": company.get("tags", ""),
        }

    def generate_email_for_company(self, company: dict) -> dict:
        graph = get_campaign_graph()
        try:
            final_state = graph.invoke({"company": company})
            return final_state.get("final_email") or {}
        except GuardrailError as exc:
            logger.warning("Campaign guardrail rejection for '%s': %s",
                           company.get("company_name", ""), exc)
            # Surface a fallback so the UI still gets a row back
            return _node_fallback({"company": company})["final_email"]

    def read_companies(self, input_file: str = "saudi_companies_500.csv", limit: int = 20):
        companies = []
        with open(input_file, mode="r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row.get("company_name", "").strip():
                    companies.append(row)
                if limit and len(companies) >= limit:
                    break
        return companies

    def save_campaign_emails(self, emails, output_file: str = "campaign_emails.csv"):
        fields = [
            "company_name", "arabic_name", "sector", "sub_sector", "city", "country",
            "website", "email", "phone", "linkedin_url", "employees", "founded_year",
            "description", "tags", "email_subject", "email_body", "campaign_goal",
            "suggested_service", "matched_projects",
        ]
        with open(output_file, mode="w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for e in emails:
                writer.writerow(e)
        return output_file

    def run(self, input_file: str = "saudi_companies_500.csv", limit: int = 20):
        companies = self.read_companies(input_file=input_file, limit=limit)
        emails = []
        for c in companies:
            emails.append(self.generate_email_for_company(self.normalize_company(c)))
        return {"total_companies": len(emails),
                "output_file": self.save_campaign_emails(emails)}

    def run_on_companies(
        self,
        companies: list,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> list:
        out = []
        for i, raw in enumerate(companies):
            c = self.normalize_company(raw)
            if progress_callback:
                progress_callback(i + 1, len(companies), c.get("company_name", ""))
            out.append(self.generate_email_for_company(c))
        return out
