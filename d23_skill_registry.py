from __future__ import annotations

import json

from shared.schemas import SkillContract


def print_section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78 + "\n")


def jprint(title: str, obj) -> None:
    print(f"--- {title} ---")
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def build_skill_registry() -> dict[str, SkillContract]:
    # Internal tools are not shown to the top-level planner. Only the skill contract is.
    return {
        "orders_skill": SkillContract(
            name="orders_skill",
            description="Resolve order delivery, shipment delays, and returns.",
            allowed_tools=["get_order_status", "submit_return"],
            forbidden_actions=["issue_refund", "change_payment_method"],
            max_tool_calls=2,
            timeout_seconds=8,
            fallback_behavior="Apologize and request order_id to continue.",
            risk_level="low",
        ),
        "billing_skill": SkillContract(
            name="billing_skill",
            description="Resolve billing issues like failed payments, duplicate charges, and refunds.",
            allowed_tools=["lookup_transaction", "request_refund"],
            forbidden_actions=["commit_refund_without_policy_check"],
            max_tool_calls=2,
            timeout_seconds=8,
            fallback_behavior="Ask for transaction_id and last 4 digits (if needed).",
            risk_level="medium",
        ),
        "technical_skill": SkillContract(
            name="technical_skill",
            description="Triage app issues and create a bug ticket when needed.",
            allowed_tools=["check_issue_tracker", "create_bug_ticket"],
            forbidden_actions=["deploy_code", "access_production_db"],
            max_tool_calls=2,
            timeout_seconds=10,
            fallback_behavior="Collect repro steps and device/app version.",
            risk_level="low",
        ),
        "rag_skill": SkillContract(
            name="rag_skill",
            description="Answer FAQs using a knowledge base search and reranking.",
            allowed_tools=["search_knowledge_base", "rerank_results"],
            forbidden_actions=["hallucinate_policy"],
            max_tool_calls=2,
            timeout_seconds=6,
            fallback_behavior="Ask a clarifying question or route to policy_skill.",
            risk_level="low",
        ),
        "policy_skill": SkillContract(
            name="policy_skill",
            description="Interpret refund/escalation policies and decide if human escalation is required.",
            allowed_tools=["check_refund_policy", "check_escalation_policy"],
            forbidden_actions=["override_policy"],
            max_tool_calls=2,
            timeout_seconds=6,
            fallback_behavior="Escalate to human for review.",
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


if __name__ == "__main__":
    print_section("Skill Registry (Reducing Tool Bloat)")

    registry = build_skill_registry()

    # What the top-level planner sees:
    print_section("SECTION: Planner view (skills only)")
    planner_view = {
        name: {
            "description": c.description,
            "risk_level": c.risk_level,
            "timeout_seconds": c.timeout_seconds,
        }
        for name, c in registry.items()
    }
    jprint("Available skills (planner view)", planner_view)

    # What the platform hides inside each skill:
    print_section("SECTION: Platform view (internal tools hidden behind skills)")
    internals_view = {name: c.allowed_tools for name, c in registry.items()}
    jprint("Internal tools (hidden behind each skill)", internals_view)

    # Tool bloat demonstration:
    print_section("SECTION: Tool bloat (context pollution) comparison")
    flattened_tools = sorted({t for c in registry.values() for t in c.allowed_tools})
    jprint(
        "Tool bloat comparison",
        {
            "skills_count": len(registry),
            "internal_tools_count": len(flattened_tools),
            "internal_tools_flat_list": flattened_tools,
            "note": "If you expose every internal tool directly to the planner, your prompt/context fills up fast.",
        },
    )

