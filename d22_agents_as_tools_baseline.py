from __future__ import annotations

import json
import os
import time
from typing import Any, Literal

from dotenv import load_dotenv  
from pydantic import BaseModel, Field  

from shared.config import DEFAULT_MODEL
from shared.logging_utils import generate_session_id, generate_trace_id
from langchain_openai import ChatOpenAI  
load_dotenv()
SubAgentName = Literal[
    "orders_agent_tool",
    "billing_agent_tool",
    "technical_agent_tool",
    "policy_agent_tool",
    "rag_agent_tool",
]


class SubAgentResult(BaseModel):
    agent: SubAgentName
    summary: str
    details: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    requires_human: bool = False


class OrchestratorPlan(BaseModel):
    tools_to_call: list[SubAgentName]
    reason: str


def print_section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78 + "\n")


def jlog(
    event: str,
    trace_id: str,
    session_id: str,
    agent_or_skill: str,
    meta: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "event": event,
        "trace_id": trace_id,
        "session_id": session_id,
        "agent_or_skill": agent_or_skill,
        "ts_ms": int(time.time() * 1000),
    }
    if meta:
        payload["meta"] = meta
    print(json.dumps(payload, ensure_ascii=False))


def _get_llm():
    # keep the demo runnable without network/API keys.
    if not os.getenv("OPENAI_API_KEY"):
        return None
    if ChatOpenAI is None:
        return None
    try:
        return ChatOpenAI(model=DEFAULT_MODEL, temperature=0)
    except Exception:
        return None


def _safe_structured_invoke(llm, prompt: str, schema_model: type[BaseModel]) -> BaseModel | None:
    """
    Structured outputs reduce "string parsing" brittleness.
    If the model/runtime can’t do it (version mismatch / missing key), we fall back.
    """
    if llm is None:
        return None
    try:
        structured = llm.with_structured_output(schema_model, method="function_calling")  # type: ignore[call-arg,attr-defined]
        return structured.invoke(prompt)
    except Exception:
        return None


def orders_agent_tool(user_request: str) -> SubAgentResult:
    llm = _get_llm()
    prompt = (
        "You are an Orders Support subagent.\n"
        "Return structured JSON with shipping/order investigation steps.\n"
        f"User request: {user_request}"
    )
    result = _safe_structured_invoke(llm, prompt, SubAgentResult)
    if isinstance(result, SubAgentResult):
        return result
    # Fallback: deterministic simulated response (teaching-friendly)
    return SubAgentResult(
        agent="orders_agent_tool",
        summary="Checked order status: likely delayed in transit. Collected next steps for carrier escalation.",
        details={"suggested_order_id": "ord_1234", "next_steps": ["confirm address", "open carrier ticket"]},
        confidence=0.72,
        requires_human=False,
    )


def billing_agent_tool(user_request: str) -> SubAgentResult:
    llm = _get_llm()
    prompt = (
        "You are a Billing Support subagent.\n"
        "Return structured JSON with double-charge investigation steps.\n"
        f"User request: {user_request}"
    )
    result = _safe_structured_invoke(llm, prompt, SubAgentResult)
    if isinstance(result, SubAgentResult):
        return result
    return SubAgentResult(
        agent="billing_agent_tool",
        summary="Detected possible duplicate charge. Recommended transaction lookup and refund request workflow.",
        details={"suspected_transactions": ["txn_7781", "txn_7782"], "refund_path": "policy check → refund request"},
        confidence=0.78,
        requires_human=False,
    )


def technical_agent_tool(user_request: str) -> SubAgentResult:
    llm = _get_llm()
    prompt = (
        "You are a Technical Support subagent.\n"
        "Return structured JSON with debugging/triage steps.\n"
        f"User request: {user_request}"
    )
    result = _safe_structured_invoke(llm, prompt, SubAgentResult)
    if isinstance(result, SubAgentResult):
        return result
    return SubAgentResult(
        agent="technical_agent_tool",
        summary="No direct technical issue detected from the request; kept as standby.",
        details={"action": "none"},
        confidence=0.5,
        requires_human=False,
    )


def policy_agent_tool(user_request: str) -> SubAgentResult:
    llm = _get_llm()
    prompt = (
        "You are a Policy subagent.\n"
        "Return structured JSON with refund/escalation policy guidance.\n"
        f"User request: {user_request}"
    )
    result = _safe_structured_invoke(llm, prompt, SubAgentResult)
    if isinstance(result, SubAgentResult):
        return result
    return SubAgentResult(
        agent="policy_agent_tool",
        summary="Refunds are typically eligible within policy window; double-charge is handled by billing ops.",
        details={"refund_window_days": 14, "escalation_team": "billing_ops"},
        confidence=0.74,
        requires_human=False,
    )


def rag_agent_tool(user_request: str) -> SubAgentResult:
    llm = _get_llm()
    prompt = (
        "You are a Knowledge Base (RAG) subagent.\n"
        "Return structured JSON with relevant KB snippets.\n"
        f"User request: {user_request}"
    )
    result = _safe_structured_invoke(llm, prompt, SubAgentResult)
    if isinstance(result, SubAgentResult):
        return result
    return SubAgentResult(
        agent="rag_agent_tool",
        summary="Found KB articles related to shipping delays and refunds.",
        details={"kb_hits": ["Shipping delays", "Refund policy overview"]},
        confidence=0.7,
        requires_human=False,
    )


TOOL_MAP: dict[SubAgentName, callable[[str], SubAgentResult]] = {
    "orders_agent_tool": orders_agent_tool,
    "billing_agent_tool": billing_agent_tool,
    "technical_agent_tool": technical_agent_tool,
    "policy_agent_tool": policy_agent_tool,
    "rag_agent_tool": rag_agent_tool,
}


def orchestrator(user_request: str, trace_id: str, session_id: str) -> str:
    """
    A minimal orchestrator: decides which subagent-tools to call, then synthesizes.

    In production, "agent chooses tools" is *not enough* — we’ll add validation + runtime enforcement later.
    """
    llm = _get_llm()
    plan_prompt = (
        "You are an orchestrator. Choose which subagent tools to call.\n"
        "Return JSON matching OrchestratorPlan.\n"
        f"Available tools: {list(TOOL_MAP.keys())}\n"
        f"User request: {user_request}"
    )
    plan = _safe_structured_invoke(llm, plan_prompt, OrchestratorPlan)

    # Fallback plan: simple, explainable heuristic (so the demo always runs).
    if not isinstance(plan, OrchestratorPlan):
        tools: list[SubAgentName] = []
        if "order" in user_request.lower() or "arrived" in user_request.lower():
            tools.append("orders_agent_tool")
        if "charged" in user_request.lower() or "billing" in user_request.lower():
            tools.append("billing_agent_tool")
            tools.append("policy_agent_tool")
        plan = OrchestratorPlan(tools_to_call=tools or ["rag_agent_tool"], reason="Heuristic routing for demo.")

    print_section("SECTION: Orchestrator plan (choose subagent tools)")
    jlog("orchestrator_plan", trace_id, session_id, agent_or_skill="orchestrator", meta=plan.model_dump())

    results: list[SubAgentResult] = []
    for tool_name in plan.tools_to_call:
        print_section(f"SECTION: Tool call → {tool_name}")
        jlog("tool_call_start", trace_id, session_id, agent_or_skill="orchestrator", meta={"tool": tool_name})
        out = TOOL_MAP[tool_name](user_request)
        results.append(out)
        jlog("tool_call_end", trace_id, session_id, agent_or_skill="orchestrator", meta={"tool": tool_name, "result": out.model_dump()})

    # Very simple synthesis (we’ll do a cleaner synthesis schema in later demos)
    combined = "\n".join([f"- {r.agent}: {r.summary}" for r in results])
    final = (
        "Here’s what I checked:\n"
        f"{combined}\n\n"
        "Next steps:\n"
        "- Please share your order ID and the two transaction IDs so we can confirm delivery status and resolve the duplicate charge."
    )
    print_section("SECTION: Synthesis (combine structured tool results)")
    jlog(
        "final_answer",
        trace_id,
        session_id,
        agent_or_skill="orchestrator",
        meta={"tools_called": plan.tools_to_call, "note": "Baseline: subagents exposed as tools."},
    )
    return final


if __name__ == "__main__":
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print(
            "Missing OPENAI_API_KEY. Copy .env.example → .env and set your key.\n"
            "This demo will still run with simulated outputs (so you can teach the architecture)."
        )

    trace_id = generate_trace_id()
    session_id = generate_session_id()

    user_test = "My order has not arrived and I was charged twice. Please check both."
    print_section("Agents as Tools (Baseline)")
    print(f"User: {user_test}\n")

    answer = orchestrator(user_test, trace_id=trace_id, session_id=session_id)
    print_section("FINAL: Synthesized answer (what the user sees)")
    print(answer)

