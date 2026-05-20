
from __future__ import annotations

import json
import os
import time
from typing import Any

from dotenv import load_dotenv

from shared.config import DEFAULT_MODEL
from shared.schemas import FinalAnswer, SkillOutput, SkillPlan, SkillTask, SkillContract
from shared.simulated_tools import (
    check_escalation_policy,
    check_issue_tracker,
    check_refund_policy,
    create_bug_ticket,
    get_order_status,
    lookup_transaction,
    request_refund,
    rerank_results,
    search_knowledge_base,
    get_plan_details,
    upgrade_plan,
)
from shared.logging_utils import generate_session_id, generate_trace_id


try:
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover
    ChatOpenAI = None  # type: ignore


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


def build_skill_registry() -> dict[str, SkillContract]:
    # Same idea as Demo 23, kept local so this file is independently runnable.
    return {
        "orders_skill": SkillContract(
            name="orders_skill",
            description="Resolve order delivery, shipment delays, and returns.",
            allowed_tools=["get_order_status", "submit_return"],
            forbidden_actions=["issue_refund", "change_payment_method"],
            max_tool_calls=2,
            timeout_seconds=8,
            fallback_behavior="Request order_id to continue.",
            risk_level="low",
        ),
        "billing_skill": SkillContract(
            name="billing_skill",
            description="Resolve billing issues like failed payments, duplicate charges, and refunds.",
            allowed_tools=["lookup_transaction", "request_refund"],
            forbidden_actions=["commit_refund_without_policy_check"],
            max_tool_calls=2,
            timeout_seconds=8,
            fallback_behavior="Request transaction_id to continue.",
            risk_level="medium",
        ),
        "technical_skill": SkillContract(
            name="technical_skill",
            description="Triage app issues and create a bug ticket when needed.",
            allowed_tools=["check_issue_tracker", "create_bug_ticket"],
            forbidden_actions=["deploy_code", "access_production_db"],
            max_tool_calls=2,
            timeout_seconds=10,
            fallback_behavior="Collect repro steps and app version.",
            risk_level="low",
        ),
        "rag_skill": SkillContract(
            name="rag_skill",
            description="Answer FAQs using a knowledge base search + reranking.",
            allowed_tools=["search_knowledge_base", "rerank_results"],
            forbidden_actions=["hallucinate_policy"],
            max_tool_calls=2,
            timeout_seconds=6,
            fallback_behavior="Ask clarifying question.",
            risk_level="low",
        ),
        "policy_skill": SkillContract(
            name="policy_skill",
            description="Interpret refund/escalation policies and decide if human escalation is required.",
            allowed_tools=["check_refund_policy", "check_escalation_policy"],
            forbidden_actions=["override_policy"],
            max_tool_calls=2,
            timeout_seconds=6,
            fallback_behavior="Escalate to human.",
            risk_level="high",
        ),
        "subscription_skill": SkillContract(
            name="subscription_skill",
            description="Retrieve subscription details, check plan pricing, or upgrade plans.",
            allowed_tools=["get_plan_details", "upgrade_plan"],
            forbidden_actions=["downgrade_plan", "cancel_subscription"],
            max_tool_calls=2,
            timeout_seconds=8,
            fallback_behavior="Request user_id and target tier to proceed.",
            risk_level="medium",
        ),
    }


def _get_llm():
    # demos should run offline (no key) using deterministic fallbacks.
    if not os.getenv("OPENAI_API_KEY"):
        return None
    if ChatOpenAI is None:
        return None
    try:
        return ChatOpenAI(model=DEFAULT_MODEL, temperature=0)
    except Exception:
        return None


def plan_with_llm(user_request: str, registry: dict[str, SkillContract]) -> SkillPlan | None:
    # Production pattern: structured output for plans (avoid brittle string parsing).
    llm = _get_llm()
    if llm is None:
        return None

    prompt = (
        "You are a skill planner.\n"
        "Return a JSON object matching SkillPlan.\n"
        "Constraints:\n"
        "- Choose at most 3 skills\n"
        "- If billing/payment/refund risk is present, include policy_skill\n"
        f"Available skills: {list(registry.keys())}\n"
        f"User request: {user_request}"
    )
    try:
        structured = llm.with_structured_output(SkillPlan, method="function_calling")  
        return structured.invoke(prompt)
    except Exception:
        return None


def plan_fallback(user_request: str) -> SkillPlan:
    text = user_request.lower()
    tasks: list[SkillTask] = []

    if "order" in text or "arrived" in text or "shipping" in text:
        tasks.append(
            SkillTask(
                skill_name="orders_skill",
                task="Check delivery status and identify next action.",
                reason="User reports order has not arrived.",
                priority="high",
            )
        )
    if "charged twice" in text or "charged" in text or "payment" in text or "debited" in text:
        tasks.append(
            SkillTask(
                skill_name="billing_skill",
                task="Check duplicate charge / payment status and determine refund eligibility path.",
                reason="User reports billing issue (possible double charge).",
                priority="high",
            )
        )
        tasks.append(
            SkillTask(
                skill_name="policy_skill",
                task="Confirm refund/escalation policy for this billing scenario.",
                reason="Billing high-priority requires policy guidance.",
                priority="high",
            )
        )

    if "subscription" in text or "upgrade" in text or "plan" in text:
        tasks.append(
            SkillTask(
                skill_name="subscription_skill",
                task="Retrieve plan details or upgrade plan.",
                reason="User query regarding subscription/plan.",
                priority="normal",
            )
        )
        if any(w in text for w in ["payment", "charge", "fee", "cost", "price", "pay"]):
            if not any(t.skill_name == "policy_skill" for t in tasks):
                tasks.append(
                    SkillTask(
                        skill_name="policy_skill",
                        task="Confirm subscription upgrade policy.",
                        reason="Upgrade involving payment requires policy validation.",
                        priority="normal",
                    )
                )

    if not tasks:
        tasks.append(
            SkillTask(
                skill_name="rag_skill",
                task="Search KB for the answer and summarize.",
                reason="No clear operational skill needed; FAQ style request.",
                priority="normal",
            )
        )

    return SkillPlan(tasks=tasks[:3], requires_human=False, reason="Heuristic planner fallback.")


def validate_plan(plan: SkillPlan, registry: dict[str, SkillContract]) -> list[str]:
    # Production pattern: deterministic validation (policy/enforcement lives in code, not in prompts).
    errors: list[str] = []

    if len(plan.tasks) > 3:
        errors.append("max skills <= 3")

    seen = set()
    for t in plan.tasks:
        if t.skill_name not in registry:
            errors.append(f"skill does not exist: {t.skill_name}")
        if t.skill_name in seen:
            errors.append(f"duplicate skill not allowed: {t.skill_name}")
        seen.add(t.skill_name)

    # rule: high priority billing task should include policy_skill
    has_high_billing = any(t.skill_name == "billing_skill" and t.priority == "high" for t in plan.tasks)
    has_policy = any(t.skill_name == "policy_skill" for t in plan.tasks)
    if has_high_billing and not has_policy:
        errors.append("high priority billing task requires policy_skill")

    # rule: upgrade_plan requires policy_skill if payment is involved
    has_upgrade_with_payment = any(
        t.skill_name == "subscription_skill"
        and "upgrade" in t.task.lower()
        and any(w in (t.task + t.reason).lower() for w in ["payment", "charge", "fee", "cost", "price", "pay"])
        for t in plan.tasks
    )
    if has_upgrade_with_payment and not has_policy:
        errors.append("subscription upgrade task involving payment requires policy_skill")

    return errors


def execute_skill(task: SkillTask) -> SkillOutput:
    """
    The executor does NOT "invent tools".
    It runs a *small, controlled* implementation per skill.
    """
    print(f"[TRACE ENGINE] execute_skill: {task.skill_name} | Task: {task.task} | Priority: {task.priority}")
    if task.skill_name == "orders_skill":
        status = get_order_status(order_id="ord_1234")
        return SkillOutput(
            skill_name="orders_skill",
            status="resolved",
            answer=f"Order `ord_1234` is `{status['status']}` with ETA {status['eta_days']} day(s).",
            confidence=0.74,
            tools_used=["get_order_status"],
            requires_human=False,
            risk_level="low",
        )

    if task.skill_name == "billing_skill":
        tx1 = lookup_transaction("txn_7781")
        tx2 = lookup_transaction("txn_7782")
        suspected_double = tx1["status"] == "settled" and tx2["status"] == "settled"
        answer = (
            f"Checked transactions txn_7781({tx1['status']}) and txn_7782({tx2['status']}). "
            + ("Looks like a possible duplicate settled charge." if suspected_double else "One of the charges may be pending/failed.")
        )
        return SkillOutput(
            skill_name="billing_skill",
            status="needs_more_info" if not suspected_double else "resolved",
            answer=answer,
            confidence=0.7 if suspected_double else 0.55,
            tools_used=["lookup_transaction", "lookup_transaction"],
            requires_human=False,
            risk_level="medium",
        )

    if task.skill_name == "technical_skill":
        hits = check_issue_tracker(keyword="checkout crash")
        ticket = None
        tools = ["check_issue_tracker"]
        if (hits.get("matching_issues") or 0) == 0:
            ticket = create_bug_ticket(summary="Crash during checkout", priority="high")
            tools.append("create_bug_ticket")
        return SkillOutput(
            skill_name="technical_skill",
            status="resolved",
            answer=f"Issue tracker check: {hits}. Ticket: {ticket}",
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
            answer="Top KB results: " + "; ".join([f"{d['title']}: {d['snippet']}" for d in top]),
            confidence=0.66,
            tools_used=["search_knowledge_base", "rerank_results"],
            requires_human=False,
            risk_level="low",
        )

    if task.skill_name == "policy_skill":
        policy = check_refund_policy(amount=49.0)
        escalation = check_escalation_policy(issue_type="charged twice")
        requires_human = bool(escalation.get("requires_human"))
        status = "escalated" if requires_human else "resolved"
        return SkillOutput(
            skill_name="policy_skill",
            status=status,
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
            answer=f"Subscription plan details: usr_9981 is on the {details['plan_tier']} plan ({details['status']}). Upgrade attempt to premium plan: {upgrade['status']}.",
            confidence=0.85,
            tools_used=["get_plan_details", "upgrade_plan"],
            requires_human=False,
            risk_level="medium",
        )

    # should never happen due to validation
    return SkillOutput(
        skill_name=str(task.skill_name),
        status="failed",
        answer="Unsupported skill.",
        confidence=0.0,
        tools_used=[],
        requires_human=True,
        risk_level="high",
    )


def synthesize(user_request: str, outputs: list[SkillOutput]) -> FinalAnswer:
    # Production pattern: synthesis as a separate step (keeps execution outputs auditable).
    skills_used = [o.skill_name for o in outputs]
    escalated = any(o.requires_human or o.risk_level == "high" for o in outputs)
    confidence = round(sum(o.confidence for o in outputs) / max(1, len(outputs)), 2)
    combined = "\n".join([f"- [{o.skill_name}] {o.answer}" for o in outputs])
    return FinalAnswer(
        answer=f"Request: {user_request}\n\nFindings:\n{combined}",
        skills_used=skills_used,
        confidence=float(confidence),
        escalated=escalated,
        reason="Escalated due to policy/high-risk output." if escalated else "All skills resolved without escalation.",
    )


if __name__ == "__main__":
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print(
            "Missing OPENAI_API_KEY. Copy .env.example → .env and set your key.\n"
            "Planner will fall back to a heuristic plan (teaching-friendly)."
        )

    print_section("DEMO 24: Planner → Validator → Executor")

    trace_id = generate_trace_id()
    session_id = generate_session_id()
    user_request = "My order has not arrived and I was charged twice. Please check both."

    registry = build_skill_registry()

    print_section("SECTION: Planner (structured SkillPlan)")
    plan = plan_with_llm(user_request, registry) or plan_fallback(user_request)
    jlog("skill_plan_generated", trace_id, session_id, agent_or_skill="planner", meta=plan.model_dump())

    print_section("SECTION: Validator (deterministic checks)")
    errors = validate_plan(plan, registry)
    if errors:
        jlog("plan_validation_failed", trace_id, session_id, agent_or_skill="validator", meta={"errors": errors})
        print("\nValidation errors:")
        for e in errors:
            print(f"- {e}")
        raise SystemExit(1)

    jlog(
        "plan_validation_passed",
        trace_id,
        session_id,
        agent_or_skill="validator",
        meta={"tasks": [t.model_dump() for t in plan.tasks]},
    )

    print_section("SECTION: Executor (run only validated skills)")
    outputs: list[SkillOutput] = []
    for task in plan.tasks:
        jlog(
            "skill_execute_start",
            trace_id,
            session_id,
            agent_or_skill="executor",
            meta={"skill": task.skill_name, "task": task.task},
        )
        out = execute_skill(task)
        outputs.append(out)
        jlog("skill_execute_end", trace_id, session_id, agent_or_skill="executor", meta=out.model_dump())

    print_section("SECTION: Synthesis (combine skill outputs)")
    final = synthesize(user_request, outputs)
    jlog("final_answer", trace_id, session_id, agent_or_skill="synthesizer", meta=final.model_dump())

    print_section("FINAL: Synthesized answer (what the user sees)")
    print(final.answer)

