# Guardrails Testing Report

Test target: `utils/guardrails.py` and its LangGraph wiring in
`agents/email_agent.py` and `agents/proposal_agent.py`.

External APIs (OpenAI, Anthropic, Gmail) were NOT hit. `OPENAI_API_KEY`
was set to a local dummy placeholder only so client objects could be constructed.

## Headline numbers

| Metric | Value |
|---|---|
| Total samples | 81 |
| Passed | 81 |
| Failed | 0 |
| Overall pass rate | 100.00% |

The floor required was 50 samples; 81 were executed.

## Per-category breakdown

| Category | Pass | Fail |
|---|---|---|
| A. Prompt-injection positives | 15 | 0 |
| B. Prompt-injection negatives | 13 | 0 |
| C. Arabic detector positives  | 7  | 0 |
| C. Arabic detector negatives  | 7  | 0 |
| D. Input validation           | 12 | 0 |
| E. Schema: CampaignEmailOutput      | 3 | 0 |
| E. Schema: OpportunityAutoOutput    | 6 | 0 |
| E. Schema: OpportunityCustomOutput  | 2 | 0 |
| E. Schema: EmailClassifyOutput      | 3 | 0 |
| E. Schema: ScoreOutput              | 5 | 0 |
| F. HITL / end-to-end          | 8  | 0 |

(Category A includes the 14 injection-positive strings plus one
`assert_no_injection` round-trip. Category B includes the 12
benign-prose negatives plus one `assert_no_injection` clean check.
Category C-arabic-neg/pos include the 6 strings each plus one
`assert_english_only` round-trip.)

## Confusion matrices

### Prompt-injection detector (`detect_prompt_injection`)

| | Predicted: injection | Predicted: clean |
|---|---|---|
| **Actual: injection** (positives) | TP = 15 | FN = 0 |
| **Actual: clean** (negatives)     | FP = 0  | TN = 13 |

Precision = 100% Recall = 100% Accuracy = 28/28 = 100%.

### Arabic detector (`contains_arabic`)

| | Predicted: arabic | Predicted: not arabic |
|---|---|---|
| **Actual: contains arabic** (positives) | TP = 7 | FN = 0 |
| **Actual: english only** (negatives)    | FP = 0 | TN = 7 |

Precision = 100% Recall = 100% Accuracy = 14/14 = 100%.

Notable negatives that were correctly NOT flagged:
- "Marhaba and welcome to Beam Data!"  (transliteration, Latin only)
- "InshAllah we will deliver on time."  (transliteration, Latin only)
- "Riyadh and Jeddah are the major cities."

Notable negatives that the injection detector correctly let through
(despite containing trigger-adjacent vocabulary):
- "Our previous quarter revenue grew 12% across Saudi Arabia."
- "Could you please share the architecture diagram of the system?"
- "The previous email I sent had the wrong attachment, apologies."
- "We need a system architecture review of our current data warehouse."

## Schema validation, per Pydantic model

| Model | Cases | Pass |
|---|---|---|
| `CampaignEmailOutput`     | 3 | 3 |
| `OpportunityAutoOutput`   | 6 | 6 |
| `OpportunityCustomOutput` | 2 | 2 |
| `EmailClassifyOutput`     | 3 | 3 |
| `ScoreOutput`             | 5 | 5 |

Boundary checks covered:
- `score = 100` accepted (both `ScoreOutput` and `OpportunityAutoOutput`)
- `score = 101` rejected
- `score = -1`  rejected
- BANT subscore `26` rejected (limit is 0..25)
- `status` outside {Qualified, Not Qualified} rejected
- `priority` outside {High, Medium, Low} rejected
- `grade` outside {High, Medium, Low} rejected
- `draft_body` shorter than 10 chars rejected
- `email_subject` shorter than 5 chars rejected
- `email_body` shorter than 30 chars rejected

## Input validation, key results

| Sample | Expected | Actual |
|---|---|---|
| `require_fields({"a":"x","b":"y"}, ["a","b"])` | ok | ok |
| `require_fields({"a":"x"}, ["a","b"])`         | raises | raises |
| `require_fields({"a":"  "}, ["a"])`            | raises | raises |
| `clamp_int(150, 0, 100)` | 100 | 100 |
| `clamp_int(-5, 0, 100)`  | 0   | 0   |
| `clamp_int("abc", 0, 100, default=7)` | 7 | 7 |
| `clamp_int("42", 0, 100)` | 42 | 42 |
| Email graph: empty recipient   | GuardrailError | GuardrailError |
| Email graph: malformed address | GuardrailError | GuardrailError |
| Email graph: well-formed address | ok | ok |
| Email graph: blank subject | GuardrailError | GuardrailError |
| Email graph: blank body    | GuardrailError | GuardrailError |

## HITL / end-to-end (`start_send_email` + `resume_send_email`, `ProposalExecutor.start`)

| # | Scenario | Expected | Actual |
|---|---|---|---|
| F1 | `start_send_email` with malformed recipient | `state=error`  | `state=error` |
| F2 | `start_send_email` with empty subject       | `state=error`  | `state=error` |
| F3 | `start_send_email` body contains "ignore previous instructions" | `state=error` | `state=error` |
| F4 | Clean send pauses at HITL interrupt          | `state=awaiting_approval` | `state=awaiting_approval` |
| F5 | `resume_send_email(approved=False)` → cancelled, Gmail NOT called | `state=cancelled` | `state=cancelled` |
| F6 | `ProposalExecutor.start` with empty `company_name` | `state=error` | `state=error` |
| F7 | `with_retries` succeeds on 2nd attempt       | "ok"   | "ok" |
| F8 | `with_retries` exhausts retries and re-raises | raises | raises |

F4 + F5 jointly confirm the pause-then-reject path is intact: the graph
halts at `_send_require_approval` (the LangGraph `interrupt(...)` node);
when the harness resumes with `approved=False`, the conditional edge
`_route_after_approval` routes to `END` instead of `gmail_send`, so the
Gmail transport is never invoked. No real Gmail credentials were
present in the test environment and no `send` call was attempted.

F1, F2, F3 confirm that bad inputs are rejected upstream of the HITL
pause — the graph never reaches `require_approval`, so the user is
never asked to approve a malformed payload.

## Failure analysis

No failures. Section omitted intentionally.
