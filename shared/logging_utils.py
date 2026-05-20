from __future__ import annotations

import json
import time
import uuid
from typing import Any


def generate_trace_id() -> str:
    return f"tr_{uuid.uuid4().hex[:12]}"


def generate_session_id() -> str:
    return f"ss_{uuid.uuid4().hex[:12]}"


def emit_event(
    state: dict[str, Any],
    event_name: str,
    agent_or_skill: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    - We log "events" as JSON objects so logs are machine-searchable.
    - We always attach trace_id + session_id to make debugging possible later.
    """
    payload: dict[str, Any] = {
        "event": event_name,
        "trace_id": state.get("trace_id"),
        "session_id": state.get("session_id"),
        "agent_or_skill": agent_or_skill,
        "ts_ms": int(time.time() * 1000),
    }
    if metadata:
        payload["meta"] = metadata
    print(json.dumps(payload, ensure_ascii=False))


def print_trace_summary(state: dict[str, Any]) -> None:
    """
    A trace summary is a compact end-of-request report.
    In production this could be shipped to observability systems.
    """
    events = state.get("trace_events", [])
    total_cost = state.get("total_cost_usd", 0.0)
    summary = {
        "event": "trace_summary",
        "trace_id": state.get("trace_id"),
        "session_id": state.get("session_id"),
        "blocked": state.get("blocked", False),
        "escalated": state.get("escalated", False),
        "skills_used": [e.get("meta", {}).get("skill_name") for e in events if e.get("event") == "skill_end"],
        "events_count": len(events),
        "total_cost_usd": round(float(total_cost or 0.0), 6),
    }
    print(json.dumps(summary, ensure_ascii=False))

