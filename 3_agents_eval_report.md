# 3-Agent Evaluation Report

Consolidated outcome of the three sub-agents launched against the
Just Sell It — AI Sales & CRM Agent project after the LangGraph + guardrails
refactor.

Sibling artefacts referenced below:
- `guardrails_testing.md` — full per-sample guardrail test results.
- `agent_evaluation.md`   — full per-agent performance evaluation.

---

## Agent 1 — Runtime testing & debug

**Scope:** end-to-end run of `app.py` via `streamlit.testing.v1.AppTest`,
direct import of every agent module, graph-compilation check, and exercise
of every guardrail / HITL path.

**Findings**

| # | File:line | Root cause | Fix |
|---|-----------|------------|-----|
| 1 | `agents/campaign_agent.py:_node_call_llm` | OpenAI call was not wrapped in try/except, so any LLM transport error (401, rate-limit, network blip) propagated out of `graph.invoke()` past `generate_email_for_company`'s `GuardrailError`-only catch and crashed the Campaign tab. | Wrapped the call in try/except, recording `error` + empty `raw_response` + bumping `attempts`, so the existing `_route_after_validate` routes through retries and lands in the deterministic `fallback` node — matching the documented "fallback so the pipeline never returns nothing" guarantee. |

**Final verdict:** 1 bug fixed. Re-run is clean — `AppTest` reports
0 exceptions / 0 errors / 0 warnings, and every graph + guardrail + HITL
path behaves as designed.

---

## Agent 2 — Guardrails testing

Full results live in `guardrails_testing.md`.

**Sample volume**

| Metric | Value |
|---|---|
| Total samples | **81** (floor was 50) |
| Categories covered | 6 / 6 |
| Real OpenAI / Anthropic / Gmail calls | 0 |

**Overall result**

| Metric | Value |
|---|---|
| Pass rate | **100.00 %** |
| Failed samples | 0 |
| Failing categories | 0 |

**Detector confusion matrices**

| Detector | TP | FP | TN | FN |
|---|---:|---:|---:|---:|
| Prompt-injection (`detect_prompt_injection`) | 15 | 0 | 13 | 0 |
| Arabic (`contains_arabic`) | 7 | 0 | 7 | 0 |

**Pydantic-schema validation pass counts**

| Model | Passing samples |
|---|---:|
| `CampaignEmailOutput` | covered |
| `OpportunityAutoOutput` (incl. `BantBreakdown`) | covered |
| `OpportunityCustomOutput` (incl. `CriterionScore`) | covered |
| `EmailClassifyOutput` | covered |
| `ScoreOutput` | covered |
| Boundary scores tested | 100, 101, −1, BANT sub-score 26 |

**HITL guardrail paths** — `start_send_email` + `resume_send_email` and
`ProposalExecutor.start` were exercised: the graphs pause as expected,
inputs are rejected upstream of the pause when invalid, and
`resume(approved=False)` never reaches Gmail.

---

## Agent 3 — Performance / hallucination evaluation

Full results live in `agent_evaluation.md`.

**Per-agent overall scores (0–100, higher is better)**

| Agent | Score |
|---|---:|
| Campaign | 64 |
| Opportunity | 75 |
| Email | 77 |
| Proposal | 57 |
| Scorer | 66 |
| **Full pipeline (weighted)** | **67** |

**Methodology notes**

- Static analysis was completed for every agent (prompt review, schema
  coverage, retry/fallback presence, hallucination vectors).
- **Dynamic testing was skipped** because `OPENAI_API_KEY` is not set
  in the shell or in a `.env` file, and `agents/opportunity_agent.py:62`
  raises on import without it.

**Biggest single hallucination risk**

`agents/campaign_agent.py:217` — the prompt asks the model for a
`matched_projects` list, but `CampaignEmailOutput.matched_projects`
at `utils/guardrails.py:64` accepts any free-form strings. There is
no intersection check against the 22 hardcoded
`BEAMDATA_PAST_PROJECTS` titles, so a fabricated project name can
leak into the outbound email and reach the prospect.

---

## Suggested next steps (not auto-applied)

1. **Kill the top hallucination vector.** Add a post-validation step in
   the campaign graph that filters `matched_projects` to the
   intersection with `[p["title"] for p in BEAMDATA_PAST_PROJECTS]`.
2. **Enable dynamic evaluation.** Either set `OPENAI_API_KEY` and re-run
   the Agent 3 evaluation with real samples, or make
   `agents/opportunity_agent.py` lazy-load the key so static imports
   work without it.
