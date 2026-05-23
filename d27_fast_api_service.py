from __future__ import annotations

import asyncio
import json
import operator
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal, TypedDict, Optional, Union

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, field_validator
from langgraph.graph import END, StateGraph

from shared.config import (
    DEFAULT_MODEL,
    MAX_LATENCY_SECONDS,
    MAX_SKILL_CALLS,
    MAX_TOOL_CALLS_PER_SKILL,
    MAX_TOTAL_COST_USD,
)
from shared.logging_utils import generate_session_id, generate_trace_id
from shared.schemas import FinalAnswer, SkillContract, SkillOutput, SkillPlan, SkillTask
from shared.simulated_tools import (
    check_escalation_policy,
    check_issue_tracker,
    check_refund_policy,
    create_bug_ticket,
    get_order_status,
    lookup_transaction,
    rerank_results,
    search_knowledge_base,
    get_plan_details,
    upgrade_plan,
)

# Load environment variables early
load_dotenv()

try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None


# --- Pydantic Schema Definitions ---

class ChatRequest(BaseModel):
    tenant_id: str = Field(..., description="Organization or customer identifier")
    user_id: str = Field(..., description="End user identifier")
    session_id: Optional[str] = Field(default=None, description="Conversation session id")
    message: str = Field(..., min_length=1, max_length=2000, description="User message")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        stripped = v.strip()
        if len(stripped) == 0:
            raise ValueError("Message cannot be empty or whitespace only")
        if len(stripped) > 2000:
            raise ValueError("Message exceeds maximum length of 2000 characters")
        return stripped


class ChatResponse(BaseModel):
    trace_id: str
    request_id: str
    session_id: str
    answer: str
    skills_used: list[str]
    escalated: bool
    total_cost_usd: float
    status: str


# --- Graph State Definition ---

class GraphState(TypedDict, total=False):
    # Production pattern: explicit state schema prevents "mystery keys" leaking across node boundaries.
    tenant_id: str
    user_id: str
    session_id: str
    request_id: str
    trace_id: str
    user_request: str
    blocked: bool
    block_reason: Optional[str]
    skills_used: list[str]
    skill_outputs: dict[str, str]  # dict of {skill_name: string_answer}
    final_answer: str              # final answer as a string
    escalated: bool
    total_cost_usd: float
    # Production pattern: append-only trace buffers use reducers to merge partial node outputs safely.
    trace_events: Annotated[list[dict[str, Any]], operator.add]
    
    # Internal orchestration keys (necessary for LangGraph state propagation)
    skill_plan: Optional[dict[str, Any]]
    validation_errors: Optional[list[str]]


# --- Utility: Trace Event Recorder ---

def record_event(state: GraphState, event: str, agent_or_skill: str, meta: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "event": event,
        "trace_id": state.get("trace_id"),
        "session_id": state.get("session_id"),
        "agent_or_skill": agent_or_skill,
        "ts_ms": int(time.time() * 1000),
        "meta": meta or {},
    }
    # Print clean machine-searchable JSON log
    print(json.dumps(payload, ensure_ascii=False))
    return payload


# --- Skill Registry ---

def build_skill_registry() -> dict[str, SkillContract]:
    return {
        "orders_skill": SkillContract(
            name="orders_skill",
            description="Resolve order delivery, shipment delays, and returns.",
            allowed_tools=["get_order_status", "submit_return"],
            forbidden_actions=["issue_refund", "change_payment_method"],
            max_tool_calls=MAX_TOOL_CALLS_PER_SKILL,
            timeout_seconds=8,
            fallback_behavior="Request order_id to continue.",
            risk_level="low",
        ),
        "billing_skill": SkillContract(
            name="billing_skill",
            description="Resolve billing issues like failed payments, duplicate charges, and refunds.",
            allowed_tools=["lookup_transaction", "request_refund"],
            forbidden_actions=["commit_refund_without_policy_check"],
            max_tool_calls=MAX_TOOL_CALLS_PER_SKILL,
            timeout_seconds=8,
            fallback_behavior="Request transaction_id to continue.",
            risk_level="medium",
        ),
        "technical_skill": SkillContract(
            name="technical_skill",
            description="Triage app issues and create a bug ticket when needed.",
            allowed_tools=["check_issue_tracker", "create_bug_ticket"],
            forbidden_actions=["deploy_code", "access_production_db"],
            max_tool_calls=MAX_TOOL_CALLS_PER_SKILL,
            timeout_seconds=10,
            fallback_behavior="Collect repro steps and app version.",
            risk_level="low",
        ),
        "rag_skill": SkillContract(
            name="rag_skill",
            description="Answer FAQs using a knowledge base search + reranking.",
            allowed_tools=["search_knowledge_base", "rerank_results"],
            forbidden_actions=["hallucinate_policy"],
            max_tool_calls=MAX_TOOL_CALLS_PER_SKILL,
            timeout_seconds=6,
            fallback_behavior="Ask clarifying question.",
            risk_level="low",
        ),
        "policy_skill": SkillContract(
            name="policy_skill",
            description="Interpret refund/escalation policies and decide if human escalation is required.",
            allowed_tools=["check_refund_policy", "check_escalation_policy"],
            forbidden_actions=["override_policy"],
            max_tool_calls=MAX_TOOL_CALLS_PER_SKILL,
            timeout_seconds=6,
            fallback_behavior="Escalate to human.",
            risk_level="high",
        ),
        "subscription_skill": SkillContract(
            name="subscription_skill",
            description="Retrieve subscription details, check plan pricing, or upgrade plans.",
            allowed_tools=["get_plan_details", "upgrade_plan"],
            forbidden_actions=["downgrade_plan", "cancel_subscription"],
            max_tool_calls=MAX_TOOL_CALLS_PER_SKILL,
            timeout_seconds=8,
            fallback_behavior="Request user_id and target tier to proceed.",
            risk_level="medium",
        ),
    }


# --- Runtime Constraints Controller ---

class RuntimeController:
    def __init__(
        self,
        trace_id: str,
        session_id: str,
        allowed_skills: set[str],
        max_skill_calls: int = MAX_SKILL_CALLS,
        max_total_cost_usd: float = MAX_TOTAL_COST_USD,
        max_latency_seconds: int = MAX_LATENCY_SECONDS,
        skill_timeout_seconds: Optional[dict[str, int]] = None,
    ):
        self.trace_id = trace_id
        self.session_id = session_id
        self.allowed_skills = allowed_skills
        self.max_skill_calls = max_skill_calls
        self.max_total_cost_usd = max_total_cost_usd
        self.max_latency_seconds = max_latency_seconds
        self.skill_timeout_seconds = skill_timeout_seconds or {}
        
        self.failure_counts: dict[str, int] = {}
        self.breaker_open_skills: set[str] = set()
        self.breaker_threshold: int = 2
        
        self.skill_calls_used: int = 0
        self.total_cost_usd: float = 0.0
        self.start_time: float = time.time()

    def can_execute_skill(self, skill_name: str) -> tuple[bool, str]:
        if skill_name in self.breaker_open_skills:
            return False, "circuit_breaker_open"
        if skill_name not in self.allowed_skills:
            return False, "skill_not_allowed"
        if self.skill_calls_used >= self.max_skill_calls:
            return False, "max_skill_calls_exceeded"
        estimated_next_cost = self.estimate_cost_usd(skill_name)
        if (self.total_cost_usd + estimated_next_cost) > self.max_total_cost_usd:
            return False, "max_total_cost_usd_exceeded"
        if (time.time() - self.start_time) >= self.max_latency_seconds:
            return False, "max_latency_seconds_exceeded"
        return True, "ok"

    def estimate_cost_usd(self, skill_name: str) -> float:
        return {"policy_skill": 0.008, "billing_skill": 0.012, "subscription_skill": 0.01}.get(skill_name, 0.01)

    def record_skill_start(self, state: GraphState, skill_name: str) -> dict[str, Any]:
        self.skill_calls_used += 1
        return record_event(
            state,
            "skill_start",
            agent_or_skill="runtime_controller",
            meta={
                "skill_name": skill_name,
                "skill_calls_used": self.skill_calls_used,
                "estimated_cost_usd": self.estimate_cost_usd(skill_name),
            },
        )

    def record_skill_result(self, state: GraphState, skill_name: str, cost_usd: float, latency_ms: int, ok: bool) -> list[dict[str, Any]]:
        self.total_cost_usd += float(cost_usd)
        events = []
        
        ev1 = record_event(
            state,
            "skill_end",
            agent_or_skill="runtime_controller",
            meta={
                "skill_name": skill_name,
                "ok": ok,
                "latency_ms": latency_ms,
                "cost_usd": round(cost_usd, 6),
                "total_cost_usd": round(self.total_cost_usd, 6),
            },
        )
        events.append(ev1)
        
        if not ok:
            self.failure_counts[skill_name] = self.failure_counts.get(skill_name, 0) + 1
            if self.failure_counts[skill_name] >= self.breaker_threshold:
                self.breaker_open_skills.add(skill_name)
                ev2 = record_event(
                    state,
                    "circuit_breaker_opened",
                    agent_or_skill="runtime_controller",
                    meta={"skill_name": skill_name, "failure_count": self.failure_counts[skill_name]},
                )
                events.append(ev2)
        return events


def _get_llm():
    if not os.getenv("OPENAI_API_KEY"):
        return None
    if ChatOpenAI is None:
        return None
    try:
        return ChatOpenAI(model=DEFAULT_MODEL, temperature=0)
    except Exception:
        return None


# --- LangGraph Nodes Implementation ---

BLOCK_PATTERNS = [
    "ignore previous instructions",
    "reveal your system prompt",
    "developer message",
    "bypass policy",
]


def input_guardrail_node(state: GraphState) -> dict[str, Any]:
    text = (state.get("user_request") or "").lower()
    for pat in BLOCK_PATTERNS:
        if pat in text:
            ev = record_event(state, "guardrail_blocked", "input_guardrail", meta={"pattern": pat})
            return {
                "blocked": True,
                "block_reason": f"blocked_prompt_injection_pattern: {pat}",
                "trace_events": [ev],
            }

    ev = record_event(state, "guardrail_passed", "input_guardrail")
    return {
        "blocked": False,
        "block_reason": None,
        "trace_events": [ev],
    }


def safe_response_node(state: GraphState) -> dict[str, Any]:
    ev = record_event(state, "safe_response", "safe_response")
    return {
        "final_answer": "I can’t help with that request. Please ask a normal support question.",
        "skills_used": [],
        "escalated": False,
        "trace_events": [ev],
    }


def route_after_guardrail(state: GraphState) -> Literal["safe_response", "skill_planner"]:
    return "safe_response" if state.get("blocked") else "skill_planner"


def skill_planner_node(state: GraphState) -> dict[str, Any]:
    registry = build_skill_registry()
    llm = _get_llm()

    if llm is not None:
        prompt = (
            "You are a production skill planner.\n"
            "Return a JSON SkillPlan.\n"
            "Rules:\n"
            "- Choose at most 3 skills\n"
            "- For mixed intents, select multiple skills\n"
            "- If billing/payment/refund risk is present, include policy_skill\n"
            f"Available skills: {list(registry.keys())}\n"
            f"User request: {state['user_request']}"
        )
        try:
            structured = llm.with_structured_output(SkillPlan, method="function_calling")
            plan_obj: SkillPlan = structured.invoke(prompt)
            plan_dict = plan_obj.model_dump()
            ev = record_event(state, "skill_plan_generated", "skill_planner", meta=plan_dict)
            return {
                "skill_plan": plan_dict,
                "trace_events": [ev],
            }
        except Exception as e:
            ev = record_event(state, "skill_plan_llm_failed", "skill_planner", meta={"error": str(e)})
            # Fall back to heuristic matching

    # Heuristic matching fallback
    txt = (state.get("user_request") or "").lower()
    tasks: list[SkillTask] = []
    if "order" in txt or "arrived" in txt or "shipping" in txt:
        tasks.append(SkillTask(skill_name="orders_skill", task="Check order status and next steps.", reason="Delivery issue.", priority="high"))
    if "crash" in txt or "crashes" in txt or "bug" in txt:
        tasks.append(SkillTask(skill_name="technical_skill", task="Triage crash and create ticket if needed.", reason="Technical issue reported.", priority="high"))
    if "charged" in txt or "debited" in txt or "payment" in txt:
        tasks.append(SkillTask(skill_name="billing_skill", task="Check payment/charge status and duplicates.", reason="Billing risk.", priority="high"))
        tasks.append(SkillTask(skill_name="policy_skill", task="Confirm refund/escalation policy.", reason="Billing high priority requires policy.", priority="high"))
    if "refund policy" in txt or "refund" in txt:
        tasks = [SkillTask(skill_name="policy_skill", task="Explain refund policy clearly.", reason="Policy request.", priority="normal")]
    if "business hours" in txt or "hours" in txt:
        tasks = [SkillTask(skill_name="rag_skill", task="Find business hours from KB.", reason="FAQ.", priority="normal")]
    if "subscription" in txt or "upgrade" in txt or "plan" in txt:
        tasks.append(SkillTask(skill_name="subscription_skill", task="Retrieve subscription details or upgrade subscription plan.", reason="Subscription/plan query.", priority="normal"))
        if any(w in txt for w in ["payment", "charge", "fee", "cost", "price", "pay"]):
            if not any(t.skill_name == "policy_skill" for t in tasks):
                tasks.append(SkillTask(skill_name="policy_skill", task="Confirm refund/escalation policy.", reason="Upgrade involving payment requires policy validation.", priority="normal"))
    if not tasks:
        tasks = [SkillTask(skill_name="rag_skill", task="Search KB and answer.", reason="General FAQ.", priority="normal")]

    plan_obj = SkillPlan(tasks=tasks[:3], requires_human=False, reason="Heuristic planner fallback.")
    plan_dict = plan_obj.model_dump()
    ev = record_event(state, "skill_plan_generated", "skill_planner", meta=plan_dict)
    return {
        "skill_plan": plan_dict,
        "trace_events": [ev],
    }


def plan_validator_node(state: GraphState) -> dict[str, Any]:
    registry = build_skill_registry()
    plan_dict = state.get("skill_plan") or {"tasks": [], "requires_human": False, "reason": "missing"}
    plan = SkillPlan.model_validate(plan_dict)
    errors: list[str] = []

    if len(plan.tasks) > 3:
        errors.append("max 3 skills")

    seen: set[str] = set()
    for t in plan.tasks:
        if t.skill_name not in registry:
            errors.append(f"skill does not exist: {t.skill_name}")
        if t.skill_name in seen:
            errors.append(f"duplicate skill not allowed: {t.skill_name}")
        seen.add(t.skill_name)

    has_high_billing = any(t.skill_name == "billing_skill" and t.priority == "high" for t in plan.tasks)
    has_policy = any(t.skill_name == "policy_skill" for t in plan.tasks)
    if has_high_billing and not has_policy:
        errors.append("billing high-priority requires policy_skill")

    has_upgrade_with_payment = any(
        t.skill_name == "subscription_skill"
        and "upgrade" in t.task.lower()
        and any(w in (t.task + t.reason).lower() for w in ["payment", "charge", "fee", "cost", "price", "pay"])
        for t in plan.tasks
    )
    if has_upgrade_with_payment and not has_policy:
        errors.append("subscription upgrade task involving payment requires policy_skill")

    allowed_skills = set(registry.keys())
    for t in plan.tasks:
        if t.skill_name not in allowed_skills:
            errors.append(f"skill not allowed: {t.skill_name}")

    ev = record_event(state, "plan_validated", "plan_validator", meta={"ok": not errors, "errors": errors})
    return {
        "validation_errors": errors,
        "trace_events": [ev],
    }


def clarify_or_escalate_node(state: GraphState) -> dict[str, Any]:
    errors = state.get("validation_errors") or []
    err_msg = "\n".join([f"- {e}" for e in errors])
    ev = record_event(state, "clarify_or_escalate", "clarify_or_escalate", meta={"errors": errors})
    return {
        "escalated": True,
        "final_answer": f"I can’t safely execute that plan. I need clarification or a human review.\nValidation errors:\n{err_msg}",
        "skills_used": [],
        "trace_events": [ev],
    }


def route_after_validation(state: GraphState) -> Literal["safe_response", "clarify_or_escalate", "skill_executor"]:
    if state.get("blocked"):
        return "safe_response"
    if state.get("validation_errors"):
        return "clarify_or_escalate"
    return "skill_executor"


def execute_skill(task: SkillTask) -> SkillOutput:
    if task.skill_name == "orders_skill":
        status_res = get_order_status("ord_1234")
        return SkillOutput(
            skill_name="orders_skill",
            status="resolved",
            answer=f"Order ord_1234 is {status_res['status']} (ETA {status_res['eta_days']} day(s)).",
            confidence=0.74,
            tools_used=["get_order_status"],
            requires_human=False,
            risk_level="low",
        )

    if task.skill_name == "billing_skill":
        tx1 = lookup_transaction("txn_7781")
        tx2 = lookup_transaction("txn_7782")
        suspected_double = tx1["status"] == "settled" and tx2["status"] == "settled"
        return SkillOutput(
            skill_name="billing_skill",
            status="resolved" if suspected_double else "needs_more_info",
            answer=f"Transactions checked: txn_7781={tx1['status']}, txn_7782={tx2['status']}.",
            confidence=0.7 if suspected_double else 0.55,
            tools_used=["lookup_transaction", "lookup_transaction"],
            requires_human=False,
            risk_level="medium",
        )

    if task.skill_name == "technical_skill":
        hits = check_issue_tracker("checkout crash")
        tools = ["check_issue_tracker"]
        ticket = None
        if (hits.get("matching_issues") or 0) == 0:
            ticket = create_bug_ticket("Crash during checkout", priority="high")
            tools.append("create_bug_ticket")
        return SkillOutput(
            skill_name="technical_skill",
            status="resolved",
            answer=f"Issue tracker: {hits}. Ticket: {ticket}.",
            confidence=0.65,
            tools_used=tools,
            requires_human=False,
            risk_level="low",
        )

    if task.skill_name == "rag_skill":
        results = search_knowledge_base(task.task)
        top = rerank_results(results)
        return SkillOutput(
            skill_name="rag_skill",
            status="resolved",
            answer="; ".join([f"{d['title']} — {d['snippet']}" for d in top]),
            confidence=0.66,
            tools_used=["search_knowledge_base", "rerank_results"],
            requires_human=False,
            risk_level="low",
        )

    if task.skill_name == "policy_skill":
        policy = check_refund_policy(amount=49.0)
        escalation = check_escalation_policy(issue_type="charged twice")
        requires_human = bool(escalation.get("requires_human"))
        return SkillOutput(
            skill_name="policy_skill",
            status="escalated" if requires_human else "resolved",
            answer=f"Refund policy: {policy}. Escalation policy: {escalation}.",
            confidence=0.72,
            tools_used=["check_refund_policy", "check_escalation_policy"],
            requires_human=requires_human,
            risk_level="high",
        )

    if task.skill_name == "subscription_skill":
        details = get_plan_details("usr_9981")
        upgrade = upgrade_plan("usr_9981", target_tier="premium")
        return SkillOutput(
            skill_name="subscription_skill",
            status="resolved",
            answer=f"Subscription details: usr_9981 is on {details['plan_tier']} ({details['status']}). Upgrade attempt to premium: {upgrade['status']}.",
            confidence=0.85,
            tools_used=["get_plan_details", "upgrade_plan"],
            requires_human=False,
            risk_level="medium",
        )

    return SkillOutput(
        skill_name=str(task.skill_name),
        status="failed",
        answer="Unsupported skill.",
        confidence=0.0,
        tools_used=[],
        requires_human=True,
        risk_level="high",
    )


def skill_executor_node(state: GraphState) -> dict[str, Any]:
    registry = build_skill_registry()
    plan_dict = state.get("skill_plan") or {"tasks": []}
    plan = SkillPlan.model_validate(plan_dict)

    runtime = RuntimeController(
        trace_id=state["trace_id"],
        session_id=state["session_id"],
        allowed_skills=set(registry.keys()),
        max_skill_calls=MAX_SKILL_CALLS,
        max_total_cost_usd=MAX_TOTAL_COST_USD,
        max_latency_seconds=MAX_LATENCY_SECONDS,
        skill_timeout_seconds={k: v.timeout_seconds for k, v in registry.items()},
    )

    outputs: list[SkillOutput] = []
    events_acc: list[dict[str, Any]] = []

    for task in plan.tasks:
        ok, reason = runtime.can_execute_skill(task.skill_name)
        if not ok:
            ev = record_event(state, "skill_rejected", "runtime_controller", meta={"skill_name": task.skill_name, "reason": reason})
            events_acc.append(ev)
            outputs.append(
                SkillOutput(
                    skill_name=task.skill_name,
                    status="failed",
                    answer=f"Skill blocked by runtime: {reason}. Fallback: {registry[task.skill_name].fallback_behavior}",
                    confidence=0.2,
                    tools_used=[],
                    requires_human=True,
                    risk_level="high",
                )
            )
            continue

        start_ev = runtime.record_skill_start(state, task.skill_name)
        events_acc.append(start_ev)

        t0 = time.time()
        simulated_timeout_s = runtime.skill_timeout_seconds.get(task.skill_name, 8)

        try:
            # Short safe pause
            time.sleep(0.01)
            if (time.time() - t0) > simulated_timeout_s:
                raise TimeoutError("skill_timeout")

            out = execute_skill(task)
            outputs.append(out)
            latency_ms = int((time.time() - t0) * 1000)
            res_events = runtime.record_skill_result(
                state,
                task.skill_name,
                cost_usd=runtime.estimate_cost_usd(task.skill_name),
                latency_ms=latency_ms,
                ok=True,
            )
            events_acc.extend(res_events)
        except Exception as e:
            latency_ms = int((time.time() - t0) * 1000)
            res_events = runtime.record_skill_result(
                state,
                task.skill_name,
                cost_usd=0.0,
                latency_ms=latency_ms,
                ok=False,
            )
            events_acc.extend(res_events)
            outputs.append(
                SkillOutput(
                    skill_name=task.skill_name,
                    status="failed",
                    answer=f"Skill failed: {type(e).__name__}. Fallback: {registry[task.skill_name].fallback_behavior}",
                    confidence=0.2,
                    tools_used=[],
                    requires_human=True,
                    risk_level="high",
                )
            )

    # Transform outputs into dict[str, str] and list[str] of used skills
    skill_outputs_dict: dict[str, str] = {o.skill_name: o.answer for o in outputs}
    skills_used_list: list[str] = [o.skill_name for o in outputs]
    
    exec_ev = record_event(state, "skills_executed", "skill_executor", meta={"count": len(outputs)})
    events_acc.append(exec_ev)

    # Return elements matching GraphState TypedDict
    return {
        "skill_outputs": skill_outputs_dict,
        "skills_used": skills_used_list,
        "total_cost_usd": runtime.total_cost_usd,
        "trace_events": events_acc,
    }


def synthesis_node(state: GraphState) -> dict[str, Any]:
    outputs_dict = state.get("skill_outputs") or {}
    skills = state.get("skills_used") or []
    
    # Analyze if escalation is required (using simulated metadata matching check_escalation_policy)
    # We check if policy_skill was used and if the response contains escalated status indications
    escalated = any("escalat" in ans.lower() or "requires_human" in ans.lower() for ans in outputs_dict.values())
    
    combined = "\n".join([f"- [{k}] {v}" for k, v in outputs_dict.items()])
    final_ans_str = f"User request: {state.get('user_request')}\n\nResults:\n{combined}"
    
    ev = record_event(state, "final_answer_ready", "synthesis", meta={"escalated": escalated, "skills_used": skills})
    return {
        "final_answer": final_ans_str,
        "escalated": escalated,
        "trace_events": [ev],
    }


# --- Graph Construction Function ---

def build_graph():
    g = StateGraph(GraphState)
    g.add_node("input_guardrail", input_guardrail_node)
    g.add_node("safe_response", safe_response_node)
    g.add_node("skill_planner", skill_planner_node)
    g.add_node("plan_validator", plan_validator_node)
    g.add_node("clarify_or_escalate", clarify_or_escalate_node)
    g.add_node("skill_executor", skill_executor_node)
    g.add_node("synthesis", synthesis_node)

    g.set_entry_point("input_guardrail")
    g.add_conditional_edges("input_guardrail", route_after_guardrail, {
        "safe_response": "safe_response",
        "skill_planner": "skill_planner",
    })
    g.add_edge("skill_planner", "plan_validator")

    g.add_conditional_edges("plan_validator", route_after_validation, {
        "safe_response": "safe_response",
        "clarify_or_escalate": "clarify_or_escalate",
        "skill_executor": "skill_executor",
    })
    g.add_edge("safe_response", END)
    g.add_edge("clarify_or_escalate", END)
    g.add_edge("skill_executor", "synthesis")
    g.add_edge("synthesis", END)
    return g.compile()


# --- FastAPI Lifecycle & App Setup ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Precompile and cache state graph to memory on startup for stateless requests
    app.state.compiled_graph = build_graph()
    print("[SERVER STARTUP] Compiled LangGraph orchestrator graph successfully.")
    yield
    print("[SERVER SHUTDOWN] Disposing FastAPI resources.")

# --- API Key Auth Placeholder ---

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def verify_api_key(api_key: Optional[str] = Depends(api_key_header)):
    if not api_key:
        # For demo purposes: if the API key is completely missing, we auto-generate a valid one so the request passes!
        return "sk_demo_auto_generated"
    # Placeholder: allow any key starting with 'sk_', 'mock_', or 'demo_' for easy demonstration
    if not (api_key.startswith("sk_") or api_key.startswith("mock_") or api_key.startswith("demo_")):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or unauthorized API key. Must start with 'sk_', 'mock_', or 'demo_'",
        )
    return api_key


app = FastAPI(
    title="A2A Skills Orchestration API Service",
    description="Stateless agent-to-agent skill execution service built with FastAPI and LangGraph",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=dict[str, str])
async def health_check():
    return {"status": "ok"}


@app.post("/v1/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat_endpoint(request: ChatRequest, api_key: str = Depends(verify_api_key)):
    req_id = f"req_{uuid.uuid4().hex[:12]}"
    trace_id = generate_trace_id()
    sess_id = request.session_id or generate_session_id()

    # Log request arrival
    start_payload = {
        "event": "request_received",
        "tenant_id": request.tenant_id,
        "user_id": request.user_id,
        "session_id": sess_id,
        "trace_id": trace_id,
        "request_id": req_id,
        "ts_ms": int(time.time() * 1000),
    }
    print(json.dumps(start_payload, ensure_ascii=False))

    # Initialize GraphState with explicit fields
    initial_state: GraphState = {
        "tenant_id": request.tenant_id,
        "user_id": request.user_id,
        "session_id": sess_id,
        "request_id": req_id,
        "trace_id": trace_id,
        "user_request": request.message,
        "blocked": False,
        "block_reason": None,
        "skills_used": [],
        "skill_outputs": {},
        "final_answer": "",
        "escalated": False,
        "total_cost_usd": 0.0,
        "trace_events": [],
    }

    t0 = time.time()
    try:
        # Load precompiled graph
        graph = app.state.compiled_graph
        # Run graph in-process safely without blocking FastAPI event loop
        final_state = await asyncio.to_thread(graph.invoke, initial_state)
    except Exception as e:
        error_payload = {
            "event": "request_failed",
            "trace_id": trace_id,
            "session_id": sess_id,
            "request_id": req_id,
            "error": str(e),
            "ts_ms": int(time.time() * 1000),
        }
        print(json.dumps(error_payload, ensure_ascii=False))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Orchestrator failed execution: {str(e)}",
        )

    latency_ms = int((time.time() - t0) * 1000)

    # Map status based on GraphState results
    # Priority order: blocked -> escalated -> (failed if validator error) -> success
    if final_state.get("blocked"):
        res_status = "blocked"
    elif final_state.get("escalated"):
        res_status = "escalated"
    elif final_state.get("validation_errors"):
        res_status = "failed"
    else:
        res_status = "success"

    # Emit request completion summary trace
    summary_payload = {
        "event": "request_completed",
        "trace_id": trace_id,
        "session_id": sess_id,
        "request_id": req_id,
        "status": res_status,
        "latency_ms": latency_ms,
        "total_cost_usd": round(float(final_state.get("total_cost_usd") or 0.0), 6),
        "skills_used": final_state.get("skills_used", []),
        "ts_ms": int(time.time() * 1000),
    }
    print(json.dumps(summary_payload, ensure_ascii=False))

    return ChatResponse(
        trace_id=trace_id,
        request_id=req_id,
        session_id=sess_id,
        answer=final_state.get("final_answer") or "",
        skills_used=final_state.get("skills_used") or [],
        escalated=bool(final_state.get("escalated")),
        total_cost_usd=round(float(final_state.get("total_cost_usd") or 0.0), 6),
        status=res_status,
    )


@app.get("/v1/traces/{trace_id}", status_code=status.HTTP_200_OK)
async def get_trace_endpoint(trace_id: str, api_key: str = Depends(verify_api_key)):
    # Return mock trace logs representing the lifecycle of the trace
    return {
        "trace_id": trace_id,
        "status": "success",
        "events": [
            {
                "event": "request_received",
                "agent_or_skill": "entry",
                "ts_ms": int(time.time() * 1000) - 100,
            },
            {
                "event": "guardrail_passed",
                "agent_or_skill": "input_guardrail",
                "ts_ms": int(time.time() * 1000) - 90,
            },
            {
                "event": "skill_plan_generated",
                "agent_or_skill": "skill_planner",
                "ts_ms": int(time.time() * 1000) - 80,
                "meta": {"skills": ["orders_skill"]},
            },
            {
                "event": "skills_executed",
                "agent_or_skill": "skill_executor",
                "ts_ms": int(time.time() * 1000) - 50,
            },
            {
                "event": "final_answer_ready",
                "agent_or_skill": "synthesis",
                "ts_ms": int(time.time() * 1000) - 10,
            }
        ]
    }
