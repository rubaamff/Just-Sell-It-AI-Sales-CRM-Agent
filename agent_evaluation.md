# Agent Evaluation Report

Project: `saudi-companies-directory`
Date: 2026-06-23
Evaluator: static analysis only (dynamic skipped — see Section 3)

---

## 1. Methodology

- **Static analysis (performed):** every agent file was read end-to-end. Prompts, output schemas (Pydantic models in `utils/guardrails.py`), retry budgets, fallback paths, and known hallucination vectors were inspected.
- **Dynamic analysis (skipped):** the prompt requires `OPENAI_API_KEY` to be present in the live environment to run real samples. `OPENAI_API_KEY` is **NOT** set in the shell, no `.env` file exists in the project root, and the CampaignAgent constructor raises `ValueError` without it. Per the instructions ("If OPENAI_API_KEY is NOT set, say so explicitly and skip dynamic — do not invent results."), no dynamic table is reported. The corresponding section below is left as a stub.
- **Scoring scale:** 0–100 on every dimension. For **Hallucination risk** lower = better (0 = none, 100 = severe). For the other four dimensions higher = better.

---

## 2. Per-agent scores

| Agent | Output accuracy potential | Hallucination risk (lower=better) | Schema robustness | Adversarial robustness | Fallback / recovery quality |
|---|---:|---:|---:|---:|---:|
| `agents/campaign_agent.py` — `CampaignAgent` | 78 | 55 | 72 | 35 | 88 |
| `agents/opportunity_agent.py` — `evaluate_opportunity_auto` (+ custom) | 82 | 30 | 88 | 55 | 80 |
| `agents/email_agent.py` — `classify_and_draft` | 80 | 35 | 78 | 78 | 85 |
| `agents/proposal_agent.py` — `ProposalExecutor` (ReAct) | 70 | 60 | 40 | 65 | 70 |
| `utils/scorer.py` — `score_in_batches(use_agent=True)` | 75 | 50 | 80 | 45 | 78 |
| **Full pipeline (Scorer → Campaign → Opportunity → Proposal)** | **68** | **62** | **65** | **50** | **75** |

> Single-number per-agent overall (mean of accuracy, schema, adversarial, fallback minus hallucination, normalized back to 0–100):
> Campaign **64**, Opportunity **75**, Email **77**, Proposal **57**, Scorer **66**, Pipeline **59**.

---

## 3. Dynamic test results

**NOT RUN.** Reason: `OPENAI_API_KEY` is not present in the shell environment and no `.env` file exists at the project root. `agents/opportunity_agent.py:62` raises `EnvironmentError` at import time without it, and the campaign/proposal/scorer agents all need it for any real call. No fabricated latencies, JSON-validity, or hallucination flags are reported.

| sample # | agent | latency_ms | schema_pass | hallucination_flag |
|---|---|---|---|---|
| — | — | — (skipped) | — | — |

---

## 4. Hallucination vectors per agent

### 4.1 `agents/campaign_agent.py` — count: **4**
1. **`matched_projects` is not constrained to the BeamData catalog.** `_build_prompt` (lines 168–219) asks the model to return `"matched_projects": [...]` but Pydantic `CampaignEmailOutput.matched_projects: List[str]` (`utils/guardrails.py:64`) is a free-string list. The agent does **not** intersect returned titles with `BEAMDATA_PAST_PROJECTS[i]["title"]`. The model can invent project titles that BeamData never delivered. **This is the single most exploitable hallucination vector in the system** (see §6).
2. **Numerical claims in past-project descriptions are passed verbatim** ("Reduced support tickets by 60%", "Increased revenue by 18%", `campaign_agent.py:52–116`). The prompt invites the LLM to "mention them naturally," which encourages paraphrase/elaboration → fabricated metrics.
3. **`suggested_service` is free text**, not validated against the 5+3 items in `BEAMDATA_SERVICES`. Hallucinated service names are accepted by the schema (`guardrails.py:63`).
4. **"Do NOT invent fake facts" is a soft, single-sentence guardrail** with no programmatic enforcement (`campaign_agent.py:208`). Mitigates risk but does not eliminate it. Positive signal.

### 4.2 `agents/opportunity_agent.py` — count: **2**
1. **Reason fields are free text** (`budget_reason`, `authority_reason`, etc., `guardrails.py:68–75`) with no length cap or evidence requirement. Model can fabricate budget/authority claims since the company JSON usually has no such data.
2. **`custom_criteria` mode is fully open-ended** (`opportunity_agent.py:120–155`). Whatever the user types becomes the rubric; no taxonomy check. Low risk operationally but a hallucination/manipulation vector.

Positives: scores are clamped (`opportunity_agent.py:234,341`), totals are recomputed from components (`:238`), `status` is re-derived from threshold (`:242`, `:345`), `response_format={"type": "json_object"}` (`:184`).

### 4.3 `agents/email_agent.py` — count: **2**
1. **`signals` field is unbounded free text** (`guardrails.py:116`) and the prompt explicitly asks the model to "extract key signals (budget hints, timeline)" (`email_agent.py:270`) — encouraging invented quantities when none exist in the source email.
2. **`draft_body` is not checked against the source email** for citation or factual consistency — the model can introduce details the sender never mentioned.

Strong positives: `detect_prompt_injection` runs on incoming body and subject (`email_agent.py:240`), the LLM is told via a system note to treat injection text as data (`:253`), outbound send-graph re-asserts no-injection on subject/body (`:411`), and HITL approval is required for the actual send (`:416`).

### 4.4 `agents/proposal_agent.py` — count: **5**
1. **No Pydantic output schema.** Unlike the other agents, the final draft is a single unstructured string (`ProposalState.draft: str`, `proposal_agent.py:104`). Section presence ("Introduction / Discovered Challenges / …") is **not** checked anywhere — the docstring claims it is (line 19) but `_node_content_safety` only checks Arabic + injection. Pipeline can ship a draft missing entire sections.
2. **DuckDuckGo tool output is fed back as truth.** Web results are scrubbed for injection (`:127`) but never for factual correctness — the agent literally builds "Discovered Challenges" from search snippets, which is the textbook RAG-hallucination shape.
3. **RAG output is similarly trusted** (`:138`); no source-citation requirement, so the LLM can blend retrieved Beam Data content with invented capabilities.
4. **Pricing rule is "give a brief, reasonable estimate range"** (`:85–89`) — this is an explicit instruction to invent a number when none is supplied.
5. **`MAX_TOOL_ITERATIONS=6`** (`:92`) — when exhausted the agent is forced to produce a final answer even with weak evidence, raising hallucination probability under sparse research.

Positives: very strong English-only enforcement (system prompt + `contains_arabic` post-check at `:211`), HITL gate before output, tool outputs filtered.

### 4.5 `utils/scorer.py` — count: **3**
1. **Web-research text is concatenated raw into the scoring prompt** (`scorer.py:248`) and the LLM is asked to score "based on BOTH" — there is no factuality check on the research blob, so a noisy or wrong Anthropic web_search summary becomes a "fact."
2. **`reason` is unbounded** in `ScoreOutput` (`guardrails.py:132`) and the prompt asks for a "one sentence" justification — the LLM can fabricate sector size / employee count claims.
3. **Falls back silently to `score=0, grade=Low` on agent failure** (`scorer.py:298–305`) — not a hallucination per se, but downstream agents may treat a hard-zero as ground truth, masking the failure.

Positives: score clamp `[0,100]` (`:274`), grade derived from score when missing (`:277`), `ScoreOutput` Pydantic validation.

---

## 5. Per-agent narrative

### Campaign Agent — overall 64
LangGraph wiring is clean (`validate_input → call_llm → validate_output → assemble | fallback`) and the deterministic fallback (`_fallback_email`, `campaign_agent.py:222`) guarantees the UI always gets a row. The schema is enforced but too permissive on `matched_projects` and `suggested_service` (see vector #1). No prompt-injection guardrail on incoming company fields — a hostile description field would be passed straight to the prompt.

### Opportunity Agent — overall 75
The best-engineered agent in the project. JSON-mode is used (`response_format={"type":"json_object"}`), BANT sub-scores are clamped, totals are recomputed server-side, and status is re-derived from threshold — so the model can't game the final qualification by reporting a high score with low BANT components. Retries + deterministic error result complete the picture.

### Email Agent — overall 77
Best adversarial posture: it is the only agent that runs `detect_prompt_injection` on incoming model-facing content **and** on outbound drafts before sending. The HITL send graph (`_node_require_approval` via `interrupt()`) is the correct pattern. Schema is tight, fallback is deterministic. Loses points only because `signals` and `draft_body` aren't grounded in the source email.

### Proposal Agent — overall 57 (weakest)
Largest surface area, weakest guardrails. No structured output schema — section presence, pricing format, length caps are all unenforced. ReAct loop with DuckDuckGo + RAG and a 6-iteration ceiling is fertile ground for hallucinated facts. English-only enforcement is strong but that is one narrow dimension. The HITL gate is good but the human reviewer is the *only* substantive line of defense against fabricated client claims.

### Scorer (agent mode) — overall 66
Solid graph and validation. The hallucination risk is moderate because the scoring prompt blends web-search output with database fields without source tagging. The `_web_search` Anthropic tool uses model id `claude-sonnet-4-6` (`scorer.py:82`) — looks like a typo for a real model id; if rejected by the API the agent silently degrades with `"Could not research: …"` and still scores, which is a graceful but information-poor fallback.

---

## 6. Full-pipeline assessment

**Compounding risk path:** Scorer fabricates a "Tadawul-listed AI-investment-ready" claim in `reason` → Campaign reads only company JSON (not the scorer's reason) so this specific claim does not propagate, **but** the Campaign agent can independently fabricate `matched_projects` titles → Opportunity reads `email_reply` and base company data, not the campaign's matched projects, so again the claim is not auto-propagated **but** the email body containing the fake project is what the prospect replies to, biasing their reply → Proposal then receives only `company_name` + `agreed_price` and reconstructs everything from web search + RAG, repeating the entire hallucination surface from scratch.

So the agents are mostly **isolated by data flow** (good), but the **email body the prospect sees** is the leaking surface — fabricated past-project titles or metrics in a Campaign email are visible to the client and damage trust irrecoverably.

### Biggest hallucination risk in the system
**`agents/campaign_agent.py:217`** — the prompt instructs the LLM to return `"matched_projects": ["project title 1", "project title 2"]` and the validating schema `CampaignEmailOutput.matched_projects` at `utils/guardrails.py:64` accepts any list of strings. There is **no intersection** with the 22 hardcoded `BEAMDATA_PAST_PROJECTS` titles before this list is written to `campaign_emails.csv` and embedded in the email body. A fabricated past-project name leaves the system and reaches the prospect.

**Recommended fix (one-liner):** after `validate_output(...)` succeeds in `_node_validate_output` (`campaign_agent.py:284`), filter `validated.matched_projects` against `{p["title"] for p in BEAMDATA_PAST_PROJECTS}` and drop unknown titles (or trigger a retry).

---

## 7. Overall system score

Weighted average across the five agents (weights chosen by blast-radius — agents whose output reaches the client weigh more):

| Agent | Overall | Weight |
|---|---:|---:|
| Campaign | 64 | 0.30 |
| Proposal | 57 | 0.25 |
| Email | 77 | 0.20 |
| Opportunity | 75 | 0.15 |
| Scorer | 66 | 0.10 |

**Weighted overall system score: `0.30·64 + 0.25·57 + 0.20·77 + 0.15·75 + 0.10·66 = 67.2 / 100`.**

Interpretation: the system is solidly engineered at the orchestration layer (LangGraph wiring, Pydantic validation, retries, fallbacks, HITL gates on send + proposal) but is held back by under-constrained free-text outputs in the two agents whose results reach the outside world (Campaign + Proposal). Fixing the `matched_projects` constraint and adding a section-presence schema for proposals would lift the overall to roughly 75.
