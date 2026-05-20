from __future__ import annotations

import random
import time
from typing import Any


# These tools intentionally do NOT call real services.
# They simulate realistic "business tools" and return small, explainable outputs.


# --- Orders tools ---
def get_order_status(order_id: str) -> dict[str, Any]:
    status = random.choice(["processing", "shipped", "out_for_delivery", "delivered", "delayed"])
    return {"order_id": order_id, "status": status, "eta_days": random.choice([0, 1, 2, 3, 5])}


def submit_return(order_id: str, reason: str) -> dict[str, Any]:
    return {"order_id": order_id, "return_id": f"ret_{order_id[-4:]}_{random.randint(100,999)}", "reason": reason}


# --- Billing tools ---
def lookup_transaction(transaction_id: str) -> dict[str, Any]:
    return {
        "transaction_id": transaction_id,
        "status": random.choice(["settled", "pending", "failed"]),
        "amount_usd": round(random.choice([19.0, 29.0, 49.0, 99.0]), 2),
        "currency": "USD",
    }


def request_refund(transaction_id: str, amount: float) -> dict[str, Any]:
    return {"transaction_id": transaction_id, "refund_id": f"rf_{transaction_id[-4:]}_{random.randint(100,999)}", "amount": amount}


# --- Technical tools ---
def check_issue_tracker(keyword: str) -> dict[str, Any]:
    hits = random.choice([0, 1, 2, 3])
    return {"keyword": keyword, "matching_issues": hits, "top_issue": "CHK-1021: crash on checkout" if hits else None}


def create_bug_ticket(summary: str, priority: str) -> dict[str, Any]:
    return {"ticket_id": f"BUG-{random.randint(1000,9999)}", "summary": summary, "priority": priority}


# --- RAG tools ---
def search_knowledge_base(query: str) -> list[dict[str, Any]]:
    # small, deterministic-ish corpus
    docs = [
        {"doc_id": "kb_001", "title": "Refund policy overview", "snippet": "Refunds are eligible within 14 days for most purchases."},
        {"doc_id": "kb_002", "title": "Business hours", "snippet": "Support is available Mon–Fri, 9am–6pm local time."},
        {"doc_id": "kb_003", "title": "Shipping delays", "snippet": "Some regions can experience 2–5 day carrier delays."},
    ]
    # naive scoring
    q = query.lower()
    scored = []
    for d in docs:
        score = 0
        if "refund" in q and "refund" in d["title"].lower():
            score += 2
        if "hours" in q and "hours" in d["title"].lower():
            score += 2
        if "order" in q or "shipping" in q:
            if "shipping" in d["title"].lower() or "delay" in d["title"].lower():
                score += 1
        scored.append({**d, "score": score})
    return sorted(scored, key=lambda x: x["score"], reverse=True)


def rerank_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # pretend "rerank" is more expensive but returns top-2 cleanly
    return results[:2]


# --- Policy tools ---
def check_refund_policy(amount: float) -> dict[str, Any]:
    if amount <= 50:
        return {"eligible": True, "route": "auto_refund", "max_days": 14}
    return {"eligible": True, "route": "manual_review", "max_days": 14}


def check_escalation_policy(issue_type: str) -> dict[str, Any]:
    issue_type = issue_type.lower()
    if "chargeback" in issue_type or "fraud" in issue_type:
        return {"requires_human": True, "team": "risk_ops"}
    if "double charge" in issue_type or "charged twice" in issue_type:
        return {"requires_human": False, "team": "billing_ops"}
    return {"requires_human": False, "team": "support"}


# --- Subscription tools ---
def get_plan_details(user_id: str) -> dict[str, Any]:
    # Simulates getting details of a user's subscription plan.
    plan_tier = random.choice(["free", "basic", "premium", "enterprise"])
    status = random.choice(["active", "past_due", "canceled"])
    price_usd = {"free": 0.0, "basic": 9.99, "premium": 29.99, "enterprise": 199.99}[plan_tier]
    return {
        "user_id": user_id,
        "plan_tier": plan_tier,
        "status": status,
        "price_usd": price_usd,
    }


def upgrade_plan(user_id: str, target_tier: str) -> dict[str, Any]:
    # Simulates upgrading a user's subscription plan.
    return {
        "user_id": user_id,
        "previous_tier": random.choice(["free", "basic", "premium"]),
        "new_tier": target_tier,
        "status": "success",
        "timestamp": int(time.time()),
    }

