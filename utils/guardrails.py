"""
Shared guardrails for every LangGraph agent in the project.

Three concerns are covered here:
  1. Input validation         — required-field checks that raise GuardrailError early.
  2. Output schema validation — Pydantic models the LLM JSON must satisfy.
  3. Content safety           — prompt-injection / language / PII heuristics
                                applied to free-text reaching or leaving the LLM.

The agents import these helpers and wire them as nodes inside their StateGraphs.
"""

from __future__ import annotations

import re
import time
import logging
from typing import Any, Callable, List, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Errors
# ─────────────────────────────────────────────────────────────────────

class GuardrailError(ValueError):
    """Raised when a guardrail rejects an input or output."""


# ─────────────────────────────────────────────────────────────────────
# Input validation
# ─────────────────────────────────────────────────────────────────────

def require_fields(data: dict, fields: List[str], where: str = "input") -> None:
    """Raise GuardrailError if any of the required keys are missing or blank."""
    missing = [f for f in fields if not str(data.get(f, "")).strip()]
    if missing:
        raise GuardrailError(
            f"{where} is missing required field(s): {', '.join(missing)}"
        )


def clamp_int(value: Any, lo: int, hi: int, default: int = 0) -> int:
    """Best-effort coercion of value to an int inside [lo, hi]."""
    try:
        v = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


# ─────────────────────────────────────────────────────────────────────
# Output schemas (Pydantic)
# ─────────────────────────────────────────────────────────────────────

class CampaignEmailOutput(BaseModel):
    email_subject: str = Field(min_length=5, max_length=200)
    email_body: str = Field(min_length=30)
    campaign_goal: str = ""
    suggested_service: str = ""
    matched_projects: List[str] = Field(default_factory=list)


class BantBreakdown(BaseModel):
    budget_score: int = Field(ge=0, le=25)
    budget_reason: str = ""
    authority_score: int = Field(ge=0, le=25)
    authority_reason: str = ""
    need_score: int = Field(ge=0, le=25)
    need_reason: str = ""
    timeline_score: int = Field(ge=0, le=25)
    timeline_reason: str = ""


class OpportunityAutoOutput(BaseModel):
    bant: BantBreakdown
    opportunity_score: int = Field(ge=0, le=100)
    status: str
    reason: str
    recommended_next_step: str

    @field_validator("status")
    @classmethod
    def _status_known(cls, v: str) -> str:
        if v not in ("Qualified", "Not Qualified"):
            raise ValueError("status must be 'Qualified' or 'Not Qualified'")
        return v


class CriterionScore(BaseModel):
    criterion: str
    score: int = Field(ge=0, le=100)
    reason: str = ""


class OpportunityCustomOutput(BaseModel):
    criteria_scores: List[CriterionScore]
    opportunity_score: int = Field(ge=0, le=100)
    status: str
    reason: str
    recommended_next_step: str

    @field_validator("status")
    @classmethod
    def _status_known(cls, v: str) -> str:
        if v not in ("Qualified", "Not Qualified"):
            raise ValueError("status must be 'Qualified' or 'Not Qualified'")
        return v


class EmailClassifyOutput(BaseModel):
    classification: str
    signals: str = ""
    priority: str
    draft_subject: str
    draft_body: str = Field(min_length=10)

    @field_validator("priority")
    @classmethod
    def _priority_known(cls, v: str) -> str:
        if v not in ("High", "Medium", "Low"):
            raise ValueError("priority must be High / Medium / Low")
        return v


class ScoreOutput(BaseModel):
    score: int = Field(ge=0, le=100)
    grade: str
    reason: str = ""

    @field_validator("grade")
    @classmethod
    def _grade_known(cls, v: str) -> str:
        if v not in ("High", "Medium", "Low"):
            raise ValueError("grade must be High / Medium / Low")
        return v


def validate_output(model: type[BaseModel], data: dict) -> BaseModel:
    """Validate an LLM JSON dict against a Pydantic schema. Raises GuardrailError."""
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise GuardrailError(f"Output schema violation: {exc.errors()}") from exc


# ─────────────────────────────────────────────────────────────────────
# Content safety / prompt-injection heuristics
# ─────────────────────────────────────────────────────────────────────

_INJECTION_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"ignore (all |the )?(previous|prior|above) (instructions|rules|messages)", re.I),
    re.compile(r"disregard (all |the )?(previous|prior|above)", re.I),
    re.compile(r"forget (everything|all (you|prior) ?(know|instructions)?)", re.I),
    re.compile(r"you are (now|actually) (?!a B2B|a sales|a senior)", re.I),
    re.compile(r"act as (a |an )?(jailbroken|unrestricted|do anything)", re.I),
    re.compile(r"system prompt[:\s]", re.I),
    re.compile(r"</?\s*system\s*>", re.I),
    re.compile(r"\bDAN\b|\bdo anything now\b", re.I),
    re.compile(r"reveal (your|the) (system|hidden) prompt", re.I),
    re.compile(r"print (your|the) (instructions|system prompt|rules)", re.I),
    re.compile(r"override (your|the) (instructions|rules|safety)", re.I),
]

_ARABIC_RANGE = re.compile(r"[؀-ۿ]")


def detect_prompt_injection(text: str) -> Optional[str]:
    """Return the first pattern hit, or None if the text looks clean."""
    if not text:
        return None
    for pat in _INJECTION_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(0)
    return None


def contains_arabic(text: str) -> bool:
    """True if any Arabic codepoints appear in text. Used by the proposal agent
    which is contractually required to output English only."""
    return bool(_ARABIC_RANGE.search(text or ""))


def assert_no_injection(text: str, where: str = "input") -> None:
    """Raise GuardrailError if text matches a known injection pattern."""
    hit = detect_prompt_injection(text)
    if hit:
        raise GuardrailError(
            f"Possible prompt injection detected in {where}: {hit!r}"
        )


def assert_english_only(text: str, where: str = "output") -> None:
    """Raise GuardrailError if text contains Arabic characters."""
    if contains_arabic(text):
        raise GuardrailError(
            f"{where} contains Arabic characters but must be English only."
        )


# ─────────────────────────────────────────────────────────────────────
# Retry helper
# ─────────────────────────────────────────────────────────────────────

def with_retries(
    fn: Callable[[], Any],
    *,
    attempts: int = 3,
    delay: float = 1.5,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
    label: str = "call",
):
    """Run fn() up to `attempts` times with linear backoff. Re-raises the
    last exception if every attempt fails."""
    last_exc: BaseException | None = None
    for i in range(1, attempts + 1):
        try:
            return fn()
        except retry_on as exc:
            last_exc = exc
            logger.warning("%s attempt %d/%d failed: %s", label, i, attempts, exc)
            if i < attempts:
                time.sleep(delay * i)
    assert last_exc is not None
    raise last_exc
