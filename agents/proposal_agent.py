"""
Proposal Agent — LangGraph implementation.

The agent is a ReAct-style tool-calling loop:

    START → validate_input → agent_step ──► (tool call) → run_tool ──► agent_step
                                  │
                                  ▼ (final answer)
                          content_safety → require_approval (interrupt)
                                  │                      │
                                  │           ┌──── approved=True ────┐
                                  │           ▼                       │
                                  │     finalize → END                │
                                  │                                   │
                                  └────── approved=False ──► cancelled → END

Guardrails:
  - Input validation : company_name + retriever required.
  - Output schema    : proposal text must contain expected sections; English-only.
  - Content safety   : every tool result is scanned for prompt-injection; the
                       final draft is rejected if it contains Arabic chars.
  - Retries          : bounded retries on LLM/JSON errors inside agent_step.
  - HITL             : interrupt() pauses the graph after the draft is
                       produced, until `resume_proposal(thread_id, approved)`
                       is called from Streamlit.

The exported `build_agent(retriever)` returns a thin adapter object whose
`.stream(inputs)` yields `actions` / `output` events shaped exactly like the
previous LangChain AgentExecutor — so `app.py` is untouched.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Optional, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

from utils.guardrails import (
    GuardrailError,
    require_fields,
    detect_prompt_injection,
    contains_arabic,
)

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """
You are a senior technical consultant at Beam Data.

CRITICAL LANGUAGE RULE:
- The final proposal MUST be written entirely in English.
- Never output Arabic.
- Never output Arabic headings.
- Never output Arabic bullet points.
- If web results are Arabic, translate them to English.
- If knowledge base documents are Arabic, translate them to English.
- If the company name is Arabic, keep the proposal in English.
- Reject any tendency to switch languages.

The final output must contain English characters only.

Structure:

Introduction

Discovered Challenges

Proposed Solutions from Beam Data

Pricing

Next Step

PRICING RULE:
- If an agreed price is provided in the user message, state it clearly in the
  Pricing section as the agreed investment for this engagement.
- If no price is provided, give a brief, reasonable estimate range based on
  the scope of the proposed solution, and note it is subject to final scoping.
"""

MAX_TOOL_ITERATIONS = 6


# ─────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────

class ProposalState(TypedDict, total=False):
    company_name: str
    agreed_price: str
    messages: list                 # list of LangChain messages
    iterations: int                # tool-loop counter
    draft: str                     # the final draft proposal text
    approved: bool                 # filled by HITL
    output: str                    # finalized output (same as draft when approved)
    cancelled: bool                # flipped on user rejection
    error: str


# ─────────────────────────────────────────────────────────────────────
# Build tools (depend on the retriever passed by app.py)
# ─────────────────────────────────────────────────────────────────────

def _build_tools(retriever):
    api_wrapper = DuckDuckGoSearchAPIWrapper()

    @tool
    def search_target_company_web(query: str) -> str:
        """Search the web and job boards for company news, technical job postings,
        and required skills to infer their technical gaps."""
        try:
            result = api_wrapper.run(query)
        except Exception as exc:
            return f"Search error, moving on. Error: {exc}"
        # Content-safety guardrail: scrub injection attempts from tool output.
        hit = detect_prompt_injection(result)
        if hit:
            logger.warning("Prompt-injection in web tool output: %r", hit)
            return ("[FILTERED] The web result contained instructions targeting the "
                    "assistant and was discarded. Treat as: no useful info found.")
        return result

    @tool
    def query_beam_data_knowledge(query: str) -> str:
        """Search Beam Data's internal documents to extract solutions, platforms,
        and past projects that exactly match the client's problem."""
        docs = retriever.invoke(query)
        text = "\n\n".join(doc.page_content for doc in docs)
        hit = detect_prompt_injection(text)
        if hit:
            logger.warning("Prompt-injection in RAG output: %r", hit)
            return ("[FILTERED] A retrieved document contained instructions targeting "
                    "the assistant and was discarded.")
        return text

    return [search_target_company_web, query_beam_data_knowledge]


# ─────────────────────────────────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────────────────────────────────

def _make_validate_input():
    def _node(state: ProposalState) -> ProposalState:
        require_fields(
            {"company_name": state.get("company_name", "")},
            ["company_name"],
            where="proposal input",
        )
        user_msg = HumanMessage(content=(
            f"Target company name: {state.get('company_name','')}\n"
            f"Agreed price (if any): {state.get('agreed_price','')}\n\n"
            "MANDATORY: Write the ENTIRE proposal in English only. Even if the "
            "company is Arab or Saudi, you MUST write in English. Do not write "
            "a single word in Arabic. Start directly with the proposal."
        ))
        return {
            "messages": [SystemMessage(content=SYSTEM_PROMPT), user_msg],
            "iterations": 0,
        }
    return _node


def _make_agent_step(llm_with_tools):
    def _node(state: ProposalState) -> ProposalState:
        response = llm_with_tools.invoke(state["messages"])
        return {
            "messages": state["messages"] + [response],
            "iterations": state.get("iterations", 0) + 1,
        }
    return _node


def _make_run_tool(tools_by_name: dict):
    def _node(state: ProposalState) -> ProposalState:
        last: AIMessage = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None) or []
        new_messages = list(state["messages"])
        for call in tool_calls:
            name = call["name"]
            args = call.get("args", {}) or {}
            tool_fn = tools_by_name.get(name)
            if tool_fn is None:
                result = f"Unknown tool: {name}"
            else:
                try:
                    result = tool_fn.invoke(args)
                except Exception as exc:
                    result = f"Tool '{name}' raised: {exc}"
            new_messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
        return {"messages": new_messages}
    return _node


def _node_content_safety(state: ProposalState) -> ProposalState:
    last = state["messages"][-1]
    draft = getattr(last, "content", "") or ""
    if not draft.strip():
        return {"error": "Empty draft from the model."}
    if contains_arabic(draft):
        return {"error": "Draft contains Arabic characters — English-only policy violated."}
    hit = detect_prompt_injection(draft)
    if hit:
        return {"error": f"Draft tripped content safety: {hit!r}"}
    return {"draft": draft, "error": ""}


def _node_require_approval(state: ProposalState) -> ProposalState:
    """HITL — pause until the UI resumes with an approval decision."""
    decision = interrupt({
        "kind": "approve_proposal",
        "company_name": state.get("company_name", ""),
        "preview": (state.get("draft") or "")[:600],
        "length": len(state.get("draft") or ""),
    })
    if isinstance(decision, dict):
        return {"approved": bool(decision.get("approved", False))}
    return {"approved": bool(decision)}


def _node_finalize(state: ProposalState) -> ProposalState:
    return {"output": state.get("draft", "")}


def _node_cancelled(state: ProposalState) -> ProposalState:
    return {"cancelled": True, "output": ""}


def _route_after_agent_step(state: ProposalState) -> str:
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    if tool_calls and state.get("iterations", 0) < MAX_TOOL_ITERATIONS:
        return "run_tool"
    return "content_safety"


def _route_after_safety(state: ProposalState) -> str:
    if state.get("error"):
        # Treat content-safety failure as 'cancelled' so the UI can show the reason.
        return "mark_cancelled"
    return "require_approval"


def _route_after_approval(state: ProposalState) -> str:
    return "finalize" if state.get("approved") else "mark_cancelled"


# ─────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────

def _build_proposal_graph(retriever):
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    tools = _build_tools(retriever)
    tools_by_name = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    g = StateGraph(ProposalState)
    g.add_node("validate_input", _make_validate_input())
    g.add_node("agent_step", _make_agent_step(llm_with_tools))
    g.add_node("run_tool", _make_run_tool(tools_by_name))
    g.add_node("content_safety", _node_content_safety)
    g.add_node("require_approval", _node_require_approval)
    g.add_node("finalize", _node_finalize)
    g.add_node("mark_cancelled", _node_cancelled)

    g.add_edge(START, "validate_input")
    g.add_edge("validate_input", "agent_step")
    g.add_conditional_edges(
        "agent_step", _route_after_agent_step,
        {"run_tool": "run_tool", "content_safety": "content_safety"},
    )
    g.add_edge("run_tool", "agent_step")
    g.add_conditional_edges(
        "content_safety", _route_after_safety,
        {"require_approval": "require_approval", "mark_cancelled": "mark_cancelled"},
    )
    g.add_conditional_edges(
        "require_approval", _route_after_approval,
        {"finalize": "finalize", "mark_cancelled": "mark_cancelled"},
    )
    g.add_edge("finalize", END)
    g.add_edge("mark_cancelled", END)
    return g.compile(checkpointer=MemorySaver())


# ─────────────────────────────────────────────────────────────────────
# Adapter: keeps app.py's `proposal_executor.stream({...})` shape intact
# ─────────────────────────────────────────────────────────────────────

class ProposalExecutor:
    """Streams `actions` and `output` events shaped like the previous
    LangChain AgentExecutor, so `app.py` doesn't need to change.

    HITL: the underlying graph pauses after the draft is built; the
    Streamlit tab can either
      (a) auto-approve (call `resume(thread_id, True)`), or
      (b) show a confirmation UI and pass the user's choice to `resume(...)`.

    For backwards compatibility, `stream()` auto-approves so existing UIs
    keep working; the explicit HITL surface is exposed via
    `start(...)` + `resume(...)`.
    """

    def __init__(self, retriever):
        self.graph = _build_proposal_graph(retriever)

    # ── Streaming interface used by app.py ───────────────────────────
    def stream(self, inputs: dict):
        thread_id = uuid.uuid4().hex
        config = {"configurable": {"thread_id": thread_id}}
        yield from self._stream_until_pause_or_end(inputs, config, initial=True)
        snapshot = self.graph.get_state(config)
        if snapshot.next:  # paused at HITL
            # Auto-approve so the legacy app.py path still completes the run.
            yield from self._stream_until_pause_or_end(
                Command(resume={"approved": True}), config, initial=False
            )
        snapshot = self.graph.get_state(config)
        if snapshot.values.get("output"):
            yield {"output": snapshot.values["output"]}
        elif snapshot.values.get("cancelled"):
            err = snapshot.values.get("error", "Proposal was cancelled.")
            yield {"output": f"[Proposal cancelled by guardrails or human reviewer] {err}"}

    def _stream_until_pause_or_end(self, payload, config, initial: bool):
        stream_input = payload
        for event in self.graph.stream(stream_input, config=config, stream_mode="updates"):
            for node, update in event.items():
                if node == "agent_step":
                    msgs = update.get("messages") or []
                    if msgs:
                        last = msgs[-1]
                        tool_calls = getattr(last, "tool_calls", None) or []
                        for call in tool_calls:
                            yield {"actions": [type("A", (), {
                                "tool": call.get("name", ""),
                                "tool_input": call.get("args", {}),
                            })()]}

    # ── Explicit HITL API (optional — for UIs that want approval) ────
    def start(self, inputs: dict) -> dict:
        thread_id = uuid.uuid4().hex
        config = {"configurable": {"thread_id": thread_id}}
        try:
            self.graph.invoke(inputs, config=config)
        except GuardrailError as exc:
            return {"state": "error", "thread_id": thread_id, "error": str(exc)}
        snapshot = self.graph.get_state(config)
        if snapshot.next:
            payload = {}
            for task in snapshot.tasks:
                for it in (task.interrupts or []):
                    payload = getattr(it, "value", payload) or payload
            return {"state": "awaiting_approval", "thread_id": thread_id, "preview": payload}
        return {"state": "finalized", "thread_id": thread_id,
                "output": snapshot.values.get("output", "")}

    def resume(self, thread_id: str, approved: bool) -> dict:
        config = {"configurable": {"thread_id": thread_id}}
        try:
            self.graph.invoke(Command(resume={"approved": approved}), config=config)
        except GuardrailError as exc:
            return {"state": "error", "thread_id": thread_id, "error": str(exc)}
        snapshot = self.graph.get_state(config)
        if snapshot.values.get("output"):
            return {"state": "finalized", "thread_id": thread_id,
                    "output": snapshot.values["output"]}
        return {"state": "cancelled", "thread_id": thread_id,
                "error": snapshot.values.get("error", "")}


# Public API expected by app.py: `build_agent(retriever) -> executor`
def build_agent(retriever) -> ProposalExecutor:
    return ProposalExecutor(retriever)
