"""
Lead Scorer — basic mode plus a LangGraph-powered agent mode.

Two scoring paths are exposed via `score_in_batches(..., use_agent=...)`:

  - Basic (use_agent=False):
        One LLM call per batch of 15 companies. Cheap and fast.

  - Agent (use_agent=True):
        Per-company LangGraph:
            validate_input → web_search → score → validate_output
                                            │
                                            ▼ (invalid + retries left)
                                          score
                                            │
                                            ▼ (retries exhausted)
                                      handle_error → END

        Guardrails:
          - Input validation : company name + sector required.
          - Output schema    : Pydantic ScoreOutput (score in [0,100], grade).
          - Retries          : bounded retries on JSON / schema / API errors.
          - Web tool         : Anthropic web_search; failure degrades gracefully.

The public entry point `score_in_batches(...)` keeps its previous signature
so `app.py` and the existing UI don't change.
"""

from __future__ import annotations

import os
import json
import logging
import urllib.request
from typing import Any, Callable, Optional, TypedDict

from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END

from utils.guardrails import (
    GuardrailError,
    require_fields,
    validate_output,
    ScoreOutput,
)

load_dotenv()

logger = logging.getLogger(__name__)

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MAX_RETRIES = int(os.getenv("SCORER_MAX_RETRIES", "3"))

BEAMDATA_CRITERIA = """
- Sector priority (High): Information Technology, Financials/Fintech, Telecom, Health Care
- Sector priority (Medium): Consumer Discretionary, Industrials, Energy
- Sector priority (Low): Consumer Staples, Materials
- Company size: prefer 200+ employees (larger = bigger AI budget)
- Location: Riyadh is top priority, other Saudi cities are acceptable
- Digital presence: having a website increases accessibility
- Established companies preferred over early-stage startups for enterprise AI deals
- BeamData sells: AI Hub Platform, Data & AI Strategy, POC development, Deployment, AI Governance
"""


# ─────────────────────────────────────────────────────────────────────
# Low-level HTTP helpers
# ─────────────────────────────────────────────────────────────────────

def _web_search(query: str) -> str:
    """Anthropic web_search tool call. Raises on transport error so the
    graph's retry/handle_error nodes see it."""
    if not ANTHROPIC_API_KEY:
        return f"(no ANTHROPIC_API_KEY configured — skipping web research for '{query}')"

    payload = json.dumps({
        "model": "claude-sonnet-4-6",
        "max_tokens": 1024,
        "tools": [{"type": "web_search_20250305", "name": "web_search"}],
        "messages": [{"role": "user",
                      "content": f"Search for: {query}. Return a brief summary in 2-3 sentences."}],
    }).encode("utf-8")

    req = urllib.request.Request(
        ANTHROPIC_API_URL, data=payload, method="POST",
        headers={
            "Content-Type": "application/json",
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    for block in data.get("content", []):
        if block.get("type") == "text":
            return block["text"]
    return "No results found."


def _openai_chat_json(prompt: str) -> dict:
    payload = json.dumps({
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }).encode("utf-8")
    req = urllib.request.Request(
        OPENAI_API_URL, data=payload, method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    raw = data["choices"][0]["message"]["content"].strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    return json.loads(raw)


# ─────────────────────────────────────────────────────────────────────
# Basic (non-agent) batch scoring — preserved
# ─────────────────────────────────────────────────────────────────────

def score_companies_basic(companies: list, criteria: str,
                          use_beamdata_defaults: bool = False) -> list:
    final_criteria = BEAMDATA_CRITERIA if use_beamdata_defaults else criteria
    companies_text = ""
    for i, c in enumerate(companies):
        companies_text += f"""
Company {i+1}:
- Name: {c.get('name', 'N/A')}
- Sector: {c.get('sector', 'N/A')}
- City: {c.get('city_clean', 'N/A')}
- Employees: {c.get('employees', 'N/A')}
- Website: {c.get('website', 'N/A')}
- Description: {c.get('description', 'N/A')}
- Is Startup: {c.get('is_startup', False)}
"""

    prompt = f"""You are a B2B sales qualification agent.

SCORING CRITERIA:
{final_criteria}

COMPANIES TO SCORE:
{companies_text}

Score each company 0-100. Respond ONLY with JSON array:
[{{"index": 1, "score": 85, "grade": "High", "reason": "One sentence"}}]"""

    payload = json.dumps({
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }).encode("utf-8")
    req = urllib.request.Request(
        OPENAI_API_URL, data=payload, method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {OPENAI_API_KEY}"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    raw = data["choices"][0]["message"]["content"].strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    scores = json.loads(raw)

    results = []
    for item in scores:
        idx = item["index"] - 1
        if 0 <= idx < len(companies):
            company = dict(companies[idx])
            company["score"] = max(0, min(100, int(item.get("score", 0))))
            company["grade"] = item.get("grade", "Low")
            company["reason"] = item.get("reason", "")
            company["research"] = ""
            results.append(company)
    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    return results


# ─────────────────────────────────────────────────────────────────────
# LangGraph agent-mode scorer (one graph instance, reused per company)
# ─────────────────────────────────────────────────────────────────────

class ScoreState(TypedDict, total=False):
    company: dict
    criteria: str
    use_beamdata_defaults: bool
    research: str
    prompt: str
    parsed: dict
    attempts: int
    error: str
    result: dict


def _sc_validate_input(state: ScoreState) -> ScoreState:
    company = state.get("company") or {}
    require_fields(company, ["name"], where="company")
    return {"attempts": 0}


def _sc_web_search(state: ScoreState) -> ScoreState:
    company = state["company"]
    name = company.get("name", "")
    query = (f"{name} Saudi Arabia AI technology latest news 2024 2025 "
             "stock exchange Tadawul")
    try:
        research = _web_search(query)
    except Exception as exc:
        research = f"Could not research: {exc}"
    return {"research": research}


def _sc_score(state: ScoreState) -> ScoreState:
    company = state["company"]
    final_criteria = (BEAMDATA_CRITERIA
                      if state.get("use_beamdata_defaults")
                      else state.get("criteria", ""))
    prompt = f"""You are a B2B sales qualification agent for BeamData AI.

SCORING CRITERIA:
{final_criteria}

COMPANY DATA (from our database):
- Name: {company.get('name', 'N/A')}
- Sector: {company.get('sector', 'N/A')}
- Sub-sector: {company.get('sub_sector', 'N/A')}
- City: {company.get('city_clean', 'N/A')}
- Employees: {company.get('employees', 'N/A')}
- Website: {company.get('website', 'N/A')}
- Description: {company.get('description', 'N/A')}
- Is Startup: {company.get('is_startup', False)}

FRESH RESEARCH FROM WEB:
{state.get('research','')}

Based on BOTH the database info AND the fresh research, score this company.
- 70-100 = High
- 40-69 = Medium
- 0-39 = Low

Respond ONLY with JSON, no markdown:
{{"score": 85, "grade": "High", "reason": "One sentence"}}"""
    try:
        parsed = _openai_chat_json(prompt)
        return {"parsed": parsed,
                "attempts": state.get("attempts", 0) + 1, "error": ""}
    except Exception as exc:
        return {"error": str(exc),
                "attempts": state.get("attempts", 0) + 1}


def _sc_validate_output(state: ScoreState) -> ScoreState:
    parsed = state.get("parsed") or {}
    if not parsed:
        return {}
    try:
        # Coerce to schema-acceptable types before validation
        if "score" in parsed:
            try:
                parsed["score"] = max(0, min(100, int(parsed["score"])))
            except (TypeError, ValueError):
                parsed["score"] = 0
        if parsed.get("grade") not in ("High", "Medium", "Low"):
            s = parsed.get("score", 0)
            parsed["grade"] = "High" if s >= 70 else ("Medium" if s >= 40 else "Low")
        validate_output(ScoreOutput, parsed)
    except GuardrailError as exc:
        return {"error": str(exc), "parsed": parsed}
    return {"parsed": parsed, "error": ""}


def _sc_finalize(state: ScoreState) -> ScoreState:
    parsed = state["parsed"]
    company = dict(state["company"])
    company.update({
        "score": parsed["score"],
        "grade": parsed["grade"],
        "reason": parsed.get("reason", ""),
        "research": state.get("research", ""),
    })
    return {"result": company}


def _sc_handle_error(state: ScoreState) -> ScoreState:
    company = dict(state["company"])
    company.update({
        "score": 0, "grade": "Low",
        "reason": f"Agent scoring failed: {state.get('error', 'unknown error')}",
        "research": state.get("research", ""),
    })
    return {"result": company}


def _sc_route(state: ScoreState) -> str:
    parsed = state.get("parsed") or {}
    if parsed and not state.get("error"):
        return "finalize"
    if state.get("attempts", 0) >= MAX_RETRIES:
        return "handle_error"
    return "score"


def _build_scorer_graph():
    g = StateGraph(ScoreState)
    g.add_node("validate_input", _sc_validate_input)
    g.add_node("web_search", _sc_web_search)
    g.add_node("score", _sc_score)
    g.add_node("validate_output", _sc_validate_output)
    g.add_node("finalize", _sc_finalize)
    g.add_node("handle_error", _sc_handle_error)

    g.add_edge(START, "validate_input")
    g.add_edge("validate_input", "web_search")
    g.add_edge("web_search", "score")
    g.add_edge("score", "validate_output")
    g.add_conditional_edges("validate_output", _sc_route, {
        "finalize": "finalize",
        "score": "score",
        "handle_error": "handle_error",
    })
    g.add_edge("finalize", END)
    g.add_edge("handle_error", END)
    return g.compile()


_SCORER_GRAPH = None


def get_scorer_graph():
    global _SCORER_GRAPH
    if _SCORER_GRAPH is None:
        _SCORER_GRAPH = _build_scorer_graph()
    return _SCORER_GRAPH


# ─────────────────────────────────────────────────────────────────────
# Public API — unchanged signature
# ─────────────────────────────────────────────────────────────────────

def score_in_batches(
    companies: list,
    criteria: str,
    use_beamdata_defaults: bool = False,
    batch_size: int = 15,
    use_agent: bool = False,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> list:
    if use_agent:
        graph = get_scorer_graph()
        results: list[dict] = []
        for i, company in enumerate(companies):
            if progress_callback:
                progress_callback(i + 1, len(companies), company.get("name", ""))
            try:
                state = graph.invoke({
                    "company": company,
                    "criteria": criteria,
                    "use_beamdata_defaults": use_beamdata_defaults,
                })
                if state.get("result"):
                    results.append(state["result"])
            except GuardrailError as exc:
                logger.warning("Scorer guardrail: %s", exc)
                fallback = dict(company)
                fallback.update({"score": 0, "grade": "Low",
                                 "reason": f"guardrail: {exc}", "research": ""})
                results.append(fallback)
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return results

    # Basic batch path
    all_results: list[dict] = []
    for i in range(0, len(companies), batch_size):
        batch = companies[i:i + batch_size]
        all_results.extend(score_companies_basic(batch, criteria, use_beamdata_defaults))
    all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
    return all_results
