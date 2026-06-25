"""
Opportunity Qualification Agent — LangGraph implementation.

Two graphs are exposed, one per assessment mode:

  Auto BANT:  validate_input → call_llm → validate_output → finalize → END
                                                │
                                                ▼ (invalid + retries left)
                                            call_llm
                                                │
                                                ▼ (retries exhausted)
                                             error → END

  Custom criteria: identical shape, different prompt / output schema.

Guardrails:
  - Input validation : company_name + sector required.
  - Output schema    : Pydantic OpportunityAutoOutput / OpportunityCustomOutput.
  - Retries          : bounded retries on JSON / schema / API errors.
  - Status fallback  : if model omits status, derived from threshold.
"""

from __future__ import annotations

import os
import json
import logging
import argparse
import time
from typing import Optional, TypedDict

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

from langgraph.graph import StateGraph, START, END

from utils.guardrails import (
    GuardrailError,
    require_fields,
    validate_output,
    OpportunityAutoOutput,
    OpportunityCustomOutput,
)

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", "2.0"))
QUALIFY_THRESHOLD = int(os.getenv("QUALIFY_THRESHOLD", "70"))

if not API_KEY:
    raise EnvironmentError(
        "Missing OPENAI_API_KEY. Add it to your .env file or environment variables."
    )

_client = OpenAI(api_key=API_KEY)


SYSTEM_PROMPT = """You are an Opportunity Qualification Agent for an AI Sales CRM.
Your role is to objectively score and qualify sales opportunities after the campaign stage.
Always respond with valid JSON only — no explanation, no markdown, no preamble."""


# ─────────────────────────────────────────────────────────────────────
# Prompts (unchanged content)
# ─────────────────────────────────────────────────────────────────────

AUTO_BANT_PROMPT = """You are an expert sales qualification agent for BeamData, an AI and data solutions company in Saudi Arabia.

Analyze this potential client and evaluate the sales opportunity using the BANT framework.
Infer each criterion intelligently from the company profile and their email reply (if any).

Company Profile:
  Name: {company_name}
  Sector: {sector}
  Sub-sector: {sub_sector}
  Employees: {employees}
  City: {city}
  Description: {description}
  Tags: {tags}
  Founded: {founded_year}

Campaign Email Subject Sent: {email_subject}

Their Reply:
{email_reply}

Evaluate using BANT (each scored 0-25):
- Budget (0-25)
- Authority (0-25)
- Need (0-25)
- Timeline (0-25)

Return ONLY valid JSON:
{{
  "bant": {{
    "budget_score": <0-25>, "budget_reason": "<one sentence>",
    "authority_score": <0-25>, "authority_reason": "<one sentence>",
    "need_score": <0-25>, "need_reason": "<one sentence>",
    "timeline_score": <0-25>, "timeline_reason": "<one sentence>"
  }},
  "opportunity_score": <integer 0-100, sum of BANT scores>,
  "status": "<Qualified | Not Qualified>",
  "reason": "<concise one-sentence overall assessment>",
  "recommended_next_step": "<concrete next action>"
}}"""


CUSTOM_CRITERIA_PROMPT = """You are an expert sales qualification agent for BeamData, an AI and data solutions company in Saudi Arabia.

Evaluate this sales opportunity based on the custom scoring criteria defined below.

Company Profile:
  Name: {company_name}
  Sector: {sector}
  Sub-sector: {sub_sector}
  Employees: {employees}
  City: {city}
  Description: {description}
  Tags: {tags}
  Founded: {founded_year}

Their Reply:
{email_reply}

Custom Scoring Criteria:
{custom_criteria}

Instructions:
- Evaluate the company against EACH criterion listed.
- Score each criterion out of 100.
- Calculate a weighted average for the total opportunity_score.
- Be objective and base scores on evidence from the company profile and reply.

Return ONLY valid JSON:
{{
  "criteria_scores": [
    {{"criterion": "<name>", "score": <0-100>, "reason": "<one sentence>"}}
  ],
  "opportunity_score": <integer 0-100, weighted average>,
  "status": "<Qualified | Not Qualified>",
  "reason": "<concise one-sentence overall assessment>",
  "recommended_next_step": "<concrete next action>"
}}"""


# ─────────────────────────────────────────────────────────────────────
# Shared state shape
# ─────────────────────────────────────────────────────────────────────

class OppState(TypedDict, total=False):
    company: dict
    email_reply: str
    custom_criteria: str
    prompt: str
    raw_response: str
    parsed: dict
    error: str
    attempts: int
    result: dict           # final dict shown to the UI
    mode: str              # "auto_bant" or "custom_criteria"


def _llm_json_call(prompt: str, max_tokens: int) -> str:
    response = _client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────────────────────────────
# Auto BANT graph
# ─────────────────────────────────────────────────────────────────────

def _auto_validate_input(state: OppState) -> OppState:
    company = state.get("company", {})
    require_fields(company, ["company_name"], where="company")
    prompt = AUTO_BANT_PROMPT.format(
        company_name=company.get("company_name", "N/A"),
        sector=company.get("sector", "N/A"),
        sub_sector=company.get("sub_sector", "N/A"),
        employees=company.get("employees", "N/A"),
        city=company.get("city", "N/A"),
        description=company.get("description", "N/A"),
        tags=company.get("tags", "N/A"),
        founded_year=company.get("founded_year", "N/A"),
        email_subject=company.get("email_subject", "BeamData Introduction"),
        email_reply=(state.get("email_reply", "").strip()
                     or "(No reply received yet — evaluate based on company profile only)"),
    )
    return {"prompt": prompt, "attempts": 0, "mode": "auto_bant"}


def _auto_call_llm(state: OppState) -> OppState:
    try:
        raw = _llm_json_call(state["prompt"], max_tokens=500)
        return {"raw_response": raw, "attempts": state.get("attempts", 0) + 1, "error": ""}
    except OpenAIError as exc:
        time.sleep(RETRY_DELAY)
        return {"error": f"openai: {exc}", "attempts": state.get("attempts", 0) + 1}


def _auto_validate_output(state: OppState) -> OppState:
    if state.get("error") and not state.get("raw_response"):
        return {}
    try:
        parsed = json.loads(state.get("raw_response", ""))
    except json.JSONDecodeError as exc:
        return {"error": f"json: {exc}", "parsed": {}}

    # Recompute total from BANT so the schema's invariant always holds
    bant = parsed.get("bant") or {}
    if isinstance(bant, dict):
        for k in ("budget_score", "authority_score", "need_score", "timeline_score"):
            try:
                bant[k] = max(0, min(25, int(bant.get(k, 0))))
            except (TypeError, ValueError):
                bant[k] = 0
        parsed["bant"] = bant
        parsed["opportunity_score"] = sum(bant[k] for k in
                                          ("budget_score", "authority_score",
                                           "need_score", "timeline_score"))

    if parsed.get("status") not in ("Qualified", "Not Qualified"):
        parsed["status"] = ("Qualified" if parsed.get("opportunity_score", 0) >= QUALIFY_THRESHOLD
                            else "Not Qualified")

    try:
        validate_output(OpportunityAutoOutput, parsed)
    except GuardrailError as exc:
        return {"error": str(exc), "parsed": parsed}
    return {"parsed": parsed, "error": ""}


def _auto_finalize(state: OppState) -> OppState:
    parsed = state["parsed"]
    parsed["assessment_mode"] = "auto_bant"
    return {"result": parsed}


def _auto_error(state: OppState) -> OppState:
    return {
        "result": {
            "opportunity_score": 0,
            "status": "Error",
            "reason": f"AI evaluation failed: {state.get('error', 'unknown error')}",
            "recommended_next_step": "Manual review required.",
            "assessment_mode": "auto_bant",
        }
    }


def _auto_route(state: OppState) -> str:
    if state.get("parsed") and not state.get("error"):
        return "finalize"
    if state.get("attempts", 0) >= MAX_RETRIES:
        return "handle_error"
    return "call_llm"


def _build_auto_graph():
    g = StateGraph(OppState)
    g.add_node("validate_input", _auto_validate_input)
    g.add_node("call_llm", _auto_call_llm)
    g.add_node("validate_output", _auto_validate_output)
    g.add_node("finalize", _auto_finalize)
    g.add_node("handle_error", _auto_error)

    g.add_edge(START, "validate_input")
    g.add_edge("validate_input", "call_llm")
    g.add_edge("call_llm", "validate_output")
    g.add_conditional_edges("validate_output", _auto_route,
                            {"finalize": "finalize", "call_llm": "call_llm", "handle_error": "handle_error"})
    g.add_edge("finalize", END)
    g.add_edge("handle_error", END)
    return g.compile()


# ─────────────────────────────────────────────────────────────────────
# Custom criteria graph
# ─────────────────────────────────────────────────────────────────────

def _custom_validate_input(state: OppState) -> OppState:
    company = state.get("company", {})
    require_fields(company, ["company_name"], where="company")
    custom = (state.get("custom_criteria") or "").strip()
    if not custom:
        raise GuardrailError("custom_criteria must not be empty")
    prompt = CUSTOM_CRITERIA_PROMPT.format(
        company_name=company.get("company_name", "N/A"),
        sector=company.get("sector", "N/A"),
        sub_sector=company.get("sub_sector", "N/A"),
        employees=company.get("employees", "N/A"),
        city=company.get("city", "N/A"),
        description=company.get("description", "N/A"),
        tags=company.get("tags", "N/A"),
        founded_year=company.get("founded_year", "N/A"),
        email_reply=(state.get("email_reply", "").strip()
                     or "(No reply received yet — evaluate based on company profile only)"),
        custom_criteria=custom,
    )
    return {"prompt": prompt, "attempts": 0, "mode": "custom_criteria"}


def _custom_call_llm(state: OppState) -> OppState:
    try:
        raw = _llm_json_call(state["prompt"], max_tokens=600)
        return {"raw_response": raw, "attempts": state.get("attempts", 0) + 1, "error": ""}
    except OpenAIError as exc:
        time.sleep(RETRY_DELAY)
        return {"error": f"openai: {exc}", "attempts": state.get("attempts", 0) + 1}


def _custom_validate_output(state: OppState) -> OppState:
    if state.get("error") and not state.get("raw_response"):
        return {}
    try:
        parsed = json.loads(state.get("raw_response", ""))
    except json.JSONDecodeError as exc:
        return {"error": f"json: {exc}", "parsed": {}}

    try:
        parsed["opportunity_score"] = max(0, min(100, int(parsed.get("opportunity_score", 0))))
    except (TypeError, ValueError):
        parsed["opportunity_score"] = 0

    if parsed.get("status") not in ("Qualified", "Not Qualified"):
        parsed["status"] = ("Qualified" if parsed["opportunity_score"] >= QUALIFY_THRESHOLD
                            else "Not Qualified")

    try:
        validate_output(OpportunityCustomOutput, parsed)
    except GuardrailError as exc:
        return {"error": str(exc), "parsed": parsed}
    return {"parsed": parsed, "error": ""}


def _custom_finalize(state: OppState) -> OppState:
    parsed = state["parsed"]
    parsed["assessment_mode"] = "custom_criteria"
    return {"result": parsed}


def _custom_error(state: OppState) -> OppState:
    return {
        "result": {
            "opportunity_score": 0,
            "status": "Error",
            "reason": f"AI evaluation failed: {state.get('error', 'unknown error')}",
            "recommended_next_step": "Manual review required.",
            "assessment_mode": "custom_criteria",
        }
    }


def _custom_route(state: OppState) -> str:
    if state.get("parsed") and not state.get("error"):
        return "finalize"
    if state.get("attempts", 0) >= MAX_RETRIES:
        return "handle_error"
    return "call_llm"


def _build_custom_graph():
    g = StateGraph(OppState)
    g.add_node("validate_input", _custom_validate_input)
    g.add_node("call_llm", _custom_call_llm)
    g.add_node("validate_output", _custom_validate_output)
    g.add_node("finalize", _custom_finalize)
    g.add_node("handle_error", _custom_error)

    g.add_edge(START, "validate_input")
    g.add_edge("validate_input", "call_llm")
    g.add_edge("call_llm", "validate_output")
    g.add_conditional_edges("validate_output", _custom_route,
                            {"finalize": "finalize", "call_llm": "call_llm", "handle_error": "handle_error"})
    g.add_edge("finalize", END)
    g.add_edge("handle_error", END)
    return g.compile()


_AUTO_GRAPH = None
_CUSTOM_GRAPH = None


def get_auto_bant_graph():
    global _AUTO_GRAPH
    if _AUTO_GRAPH is None:
        _AUTO_GRAPH = _build_auto_graph()
    return _AUTO_GRAPH


def get_custom_graph():
    global _CUSTOM_GRAPH
    if _CUSTOM_GRAPH is None:
        _CUSTOM_GRAPH = _build_custom_graph()
    return _CUSTOM_GRAPH


# ─────────────────────────────────────────────────────────────────────
# Public API — preserved for app.py
# ─────────────────────────────────────────────────────────────────────

def evaluate_opportunity_auto(company_data: dict, email_reply: str = "") -> dict:
    try:
        state = get_auto_bant_graph().invoke(
            {"company": company_data, "email_reply": email_reply or ""}
        )
        return state.get("result") or {}
    except GuardrailError as exc:
        return {
            "opportunity_score": 0,
            "status": "Error",
            "reason": f"Input guardrail: {exc}",
            "recommended_next_step": "Fix input and retry.",
            "assessment_mode": "auto_bant",
        }


def evaluate_opportunity_custom(company_data: dict, custom_criteria: str, email_reply: str = "") -> dict:
    try:
        state = get_custom_graph().invoke({
            "company": company_data,
            "email_reply": email_reply or "",
            "custom_criteria": custom_criteria or "",
        })
        return state.get("result") or {}
    except GuardrailError as exc:
        return {
            "opportunity_score": 0,
            "status": "Error",
            "reason": f"Input guardrail: {exc}",
            "recommended_next_step": "Fix input and retry.",
            "assessment_mode": "custom_criteria",
        }


def evaluate_single_opportunity(data: dict) -> dict:
    """Backwards-compatible single-row evaluator used by tests / CLI.
    Treats the dict as both 'company profile' and 'email response' fields."""
    # Keep the legacy flat-dict scoring rubric available by routing through
    # the auto-BANT graph with what info we have.
    company_ctx = {
        "company_name": data.get("company_name", ""),
        "sector": data.get("industry", "") or data.get("sector", ""),
        "sub_sector": data.get("sub_sector", ""),
        "employees": data.get("employees", "") or data.get("decision_maker", ""),
        "city": data.get("city", ""),
        "description": data.get("need", "") or data.get("description", ""),
        "tags": data.get("tags", ""),
        "founded_year": data.get("founded_year", ""),
        "email_subject": data.get("email_subject", "BeamData Introduction"),
    }
    return evaluate_opportunity_auto(company_ctx, data.get("email_response", ""))


# ─────────────────────────────────────────────────────────────────────
# CLI (legacy support)
# ─────────────────────────────────────────────────────────────────────

REQUIRED_COLUMNS = [
    "company_name", "industry", "budget", "email_response", "need",
    "decision_maker", "timeline", "agreement_level", "potential_value",
]


def validate_input(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")
    logger.info("Input validation passed — %d rows, %d columns.", len(df), len(df.columns))


def run_opportunity_agent(
    input_file: str,
    output_file: str,
    proposal_output_file: str,
    human_loop: bool = False,
) -> None:
    df = pd.read_csv(input_file)
    validate_input(df)
    results: list[dict] = []
    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        ctx = row.to_dict()
        r = evaluate_single_opportunity(ctx)
        results.append({**ctx,
                        "ai_opportunity_score": r.get("opportunity_score", 0),
                        "ai_status": r.get("status", ""),
                        "ai_reason": r.get("reason", ""),
                        "ai_recommended_next_step": r.get("recommended_next_step", "")})
    out_df = pd.DataFrame(results).sort_values(by="ai_opportunity_score", ascending=False)
    out_df.to_csv(output_file, index=False)
    out_df[out_df["ai_status"] == "Qualified"].to_csv(proposal_output_file, index=False)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="campaign_responses.csv")
    parser.add_argument("--output", default="qualified_opportunities.csv")
    parser.add_argument("--proposal-output", default="proposal_ready_opportunities.csv")
    parser.add_argument("--human-loop", action="store_true")
    parser.add_argument("--threshold", type=int, default=QUALIFY_THRESHOLD)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    global QUALIFY_THRESHOLD
    QUALIFY_THRESHOLD = args.threshold
    run_opportunity_agent(args.input, args.output, args.proposal_output, args.human_loop)


if __name__ == "__main__":
    main()
