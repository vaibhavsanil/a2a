from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from shared.config import MAX_LATENCY_SECONDS, MAX_SKILL_CALLS, MAX_TOTAL_COST_USD
from shared.logging_utils import generate_session_id, generate_trace_id


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


@dataclass
class RuntimeController:
    # Production pattern: runtime controller tracks budgets + limits across the whole request/session.
    trace_id: str
    session_id: str
    max_skill_calls: int = MAX_SKILL_CALLS
    max_total_cost_usd: float = MAX_TOTAL_COST_USD
    max_latency_seconds: int = MAX_LATENCY_SECONDS
    allowed_skills: set[str] = field(default_factory=lambda: {"orders_skill", "billing_skill", "technical_skill", "rag_skill", "policy_skill", "subscription_skill"})
    per_skill_timeout_ms: int = 500

    # True per-skill timeouts map
    skill_timeout_ms: dict[str, int] = field(default_factory=dict)

    # circuit breaker fields
    failure_counts: dict[str, int] = field(default_factory=dict)
    breaker_open_skills: set[str] = field(default_factory=set)
    breaker_threshold: int = 2

    skill_calls_used: int = 0
    total_cost_usd: float = 0.0
    start_time: float = field(default_factory=time.time)

    def estimate_cost_usd(self, skill_name: str) -> float:
        # simple, explainable estimation (avoid billing complexity in class)
        return {"policy_skill": 0.008, "billing_skill": 0.012, "subscription_skill": 0.01}.get(skill_name, 0.01)

    def can_execute_skill(self, skill_name: str) -> tuple[bool, str]:
        # Production pattern: central enforcement gate (every execution path checks here).
        if skill_name in self.breaker_open_skills:
            return False, "circuit_breaker_open"
        if skill_name not in self.allowed_skills:
            return False, "skill not allowed"
        if self.skill_calls_used >= self.max_skill_calls:
            return False, "max_skill_calls exceeded"
        # Preemptive estimated cost check before execution
        estimated_next_cost = self.estimate_cost_usd(skill_name)
        if (self.total_cost_usd + estimated_next_cost) > self.max_total_cost_usd:
            return False, "max_total_cost_usd exceeded"
        if (time.time() - self.start_time) >= self.max_latency_seconds:
            return False, "max_latency_seconds exceeded"
        return True, "ok"

    def record_skill_start(self, skill_name: str) -> None:
        self.skill_calls_used += 1
        jlog(
            "skill_start",
            self.trace_id,
            self.session_id,
            agent_or_skill="runtime_controller",
            meta={
                "skill_name": skill_name, 
                "skill_calls_used": self.skill_calls_used,
                "estimated_cost_usd": self.estimate_cost_usd(skill_name)
            },
        )

    def record_skill_result(self, skill_name: str, cost_usd: float, latency_ms: int, ok: bool = True) -> None:
        self.total_cost_usd += float(cost_usd)
        jlog(
            "skill_end",
            self.trace_id,
            self.session_id,
            agent_or_skill="runtime_controller",
            meta={
                "skill_name": skill_name,
                "latency_ms": latency_ms,
                "cost_usd": round(cost_usd, 6),
                "total_cost_usd": round(self.total_cost_usd, 6),
                "ok": ok,
            },
        )
        if not ok:
            self.failure_counts[skill_name] = self.failure_counts.get(skill_name, 0) + 1
            if self.failure_counts[skill_name] >= self.breaker_threshold:
                self.breaker_open_skills.add(skill_name)
                jlog(
                    "circuit_breaker_opened",
                    self.trace_id,
                    self.session_id,
                    agent_or_skill="runtime_controller",
                    meta={"skill_name": skill_name, "failure_count": self.failure_counts[skill_name]},
                )

    def check_budget(self) -> bool:
        return self.total_cost_usd < self.max_total_cost_usd

    def check_latency(self) -> bool:
        return (time.time() - self.start_time) < self.max_latency_seconds

    def emit_runtime_summary(self) -> None:
        jlog(
            "runtime_summary",
            self.trace_id,
            self.session_id,
            agent_or_skill="runtime_controller",
            meta={
                "skill_calls_used": self.skill_calls_used,
                "max_skill_calls": self.max_skill_calls,
                "total_cost_usd": round(self.total_cost_usd, 6),
                "max_total_cost_usd": self.max_total_cost_usd,
                "elapsed_s": round(time.time() - self.start_time, 3),
                "max_latency_seconds": self.max_latency_seconds,
            },
        )


def simulate_skill_execution(runtime: RuntimeController, skill_name: str, simulated_cost: float, simulated_latency_ms: int, ok: bool = True) -> None:
    allowed, reason = runtime.can_execute_skill(skill_name)
    if not allowed:
        jlog(
            "skill_rejected",
            runtime.trace_id,
            runtime.session_id,
            agent_or_skill="runtime_controller",
            meta={"skill_name": skill_name, "reason": reason},
        )
        jlog(
            "fallback_returned",
            runtime.trace_id,
            runtime.session_id,
            agent_or_skill="runtime_controller",
            meta={"skill_name": skill_name, "fallback": "Unable to safely execute this step. Please try again later or escalate."},
        )
        return

    runtime.record_skill_start(skill_name)
    # simulate latency without slowing class too much (small sleep)
    time.sleep(min(simulated_latency_ms / 1000.0, 0.2))
    timeout_ms = runtime.skill_timeout_ms.get(skill_name, runtime.per_skill_timeout_ms)
    if simulated_latency_ms > timeout_ms:
        jlog(
            "skill_timeout",
            runtime.trace_id,
            runtime.session_id,
            agent_or_skill="runtime_controller",
            meta={"skill_name": skill_name, "timeout_ms": timeout_ms, "observed_latency_ms": simulated_latency_ms},
        )
        jlog(
            "fallback_returned",
            runtime.trace_id,
            runtime.session_id,
            agent_or_skill="runtime_controller",
            meta={"skill_name": skill_name, "fallback": "Step timed out. Returning a safe fallback response."},
        )
        # We intentionally do not add cost on timeout in this teaching demo.
        return
    runtime.record_skill_result(skill_name, cost_usd=simulated_cost, latency_ms=simulated_latency_ms, ok=ok)


if __name__ == "__main__":
    print_section("DEMO 25: Runtime Constraints (Enforcement beyond prompts)")

    # Scenario A: allowed execution succeeds
    trace_id = generate_trace_id()
    session_id = generate_session_id()
    runtime = RuntimeController(
        trace_id=trace_id,
        session_id=session_id,
        max_skill_calls=3,
        max_total_cost_usd=0.05,
        max_latency_seconds=20,
        per_skill_timeout_ms=500,
    )

    print_section("SCENARIO A: normal execution (allowed)")
    simulate_skill_execution(runtime, "orders_skill", simulated_cost=0.01, simulated_latency_ms=200)
    simulate_skill_execution(runtime, "billing_skill", simulated_cost=0.02, simulated_latency_ms=250)
    simulate_skill_execution(runtime, "policy_skill", simulated_cost=0.005, simulated_latency_ms=150)
    runtime.emit_runtime_summary()

    # Scenario D: per-skill timeout enforced (fallback fires)
    trace_id = generate_trace_id()
    session_id = generate_session_id()
    runtime = RuntimeController(trace_id=trace_id, session_id=session_id, max_skill_calls=5, max_total_cost_usd=0.20, max_latency_seconds=20, per_skill_timeout_ms=180)

    print_section("SCENARIO D: per-skill timeout (fallback fires)")
    simulate_skill_execution(runtime, "rag_skill", simulated_cost=0.01, simulated_latency_ms=250)  # timeout
    simulate_skill_execution(runtime, "orders_skill", simulated_cost=0.01, simulated_latency_ms=120)  # ok
    runtime.emit_runtime_summary()

    # Scenario B: max skill calls exceeded
    trace_id = generate_trace_id()
    session_id = generate_session_id()
    runtime = RuntimeController(
        trace_id=trace_id,
        session_id=session_id,
        max_skill_calls=3,
        max_total_cost_usd=0.20,
        max_latency_seconds=20,
        per_skill_timeout_ms=500,
    )

    print_section("SCENARIO B: max_skill_calls exceeded (rejected)")
    simulate_skill_execution(runtime, "orders_skill", simulated_cost=0.01, simulated_latency_ms=120)
    simulate_skill_execution(runtime, "billing_skill", simulated_cost=0.01, simulated_latency_ms=120)
    simulate_skill_execution(runtime, "policy_skill", simulated_cost=0.01, simulated_latency_ms=120)
    simulate_skill_execution(runtime, "subscription_skill", simulated_cost=0.01, simulated_latency_ms=120)  # rejected
    runtime.emit_runtime_summary()

    # Scenario C: budget exceeded (fallback simulation)
    trace_id = generate_trace_id()
    session_id = generate_session_id()
    runtime = RuntimeController(
        trace_id=trace_id,
        session_id=session_id,
        max_skill_calls=5,
        max_total_cost_usd=0.02,
        max_latency_seconds=20,
        per_skill_timeout_ms=500,
    )

    print_section("SCENARIO C: budget exceeded (rejected)")
    simulate_skill_execution(runtime, "billing_skill", simulated_cost=0.015, simulated_latency_ms=200)
    simulate_skill_execution(runtime, "policy_skill", simulated_cost=0.010, simulated_latency_ms=200)  # pushes over budget
    simulate_skill_execution(runtime, "orders_skill", simulated_cost=0.005, simulated_latency_ms=120)  # rejected (budget)
    runtime.emit_runtime_summary()

    # Scenario E: circuit breaker opens
    trace_id = generate_trace_id()
    session_id = generate_session_id()
    runtime = RuntimeController(
        trace_id=trace_id,
        session_id=session_id,
        max_skill_calls=5,
        max_total_cost_usd=0.20,
        max_latency_seconds=20,
        per_skill_timeout_ms=500,
    )

    print_section("SCENARIO E: circuit breaker opens (rejected)")
    simulate_skill_execution(runtime, "billing_skill", simulated_cost=0.01, simulated_latency_ms=120, ok=False) # 1st failure
    simulate_skill_execution(runtime, "billing_skill", simulated_cost=0.01, simulated_latency_ms=120, ok=False) # 2nd failure -> breaker opens
    simulate_skill_execution(runtime, "billing_skill", simulated_cost=0.01, simulated_latency_ms=120, ok=True) # 3rd call -> rejected by breaker
    runtime.emit_runtime_summary()

    # Scenario F: true per-skill timeouts
    trace_id = generate_trace_id()
    session_id = generate_session_id()
    runtime = RuntimeController(
        trace_id=trace_id,
        session_id=session_id,
        max_skill_calls=5,
        max_total_cost_usd=0.20,
        max_latency_seconds=20,
        per_skill_timeout_ms=500,
        skill_timeout_ms={"rag_skill": 100, "orders_skill": 400}
    )

    print_section("SCENARIO F: custom per-skill timeouts")
    simulate_skill_execution(runtime, "rag_skill", simulated_cost=0.01, simulated_latency_ms=150) # timeout (threshold 100ms)
    simulate_skill_execution(runtime, "orders_skill", simulated_cost=0.01, simulated_latency_ms=250) # success (threshold 400ms)
    runtime.emit_runtime_summary()

