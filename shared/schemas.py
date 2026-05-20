from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


SkillName = Literal["orders_skill", "billing_skill", "technical_skill", "rag_skill", "policy_skill", "subscription_skill"]
Priority = Literal["low", "normal", "high"]
SkillStatus = Literal["resolved", "needs_more_info", "failed", "escalated"]
RiskLevel = Literal["low", "medium", "high"]


class SkillTask(BaseModel):
    skill_name: SkillName
    task: str = Field(..., min_length=1)
    reason: str = Field(..., min_length=1)
    priority: Priority = "normal"


class SkillPlan(BaseModel):
    tasks: list[SkillTask]
    requires_human: bool = False
    reason: str


class SkillOutput(BaseModel):
    skill_name: str
    status: SkillStatus
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    tools_used: list[str] = Field(default_factory=list)
    requires_human: bool = False
    risk_level: RiskLevel = "low"


class FinalAnswer(BaseModel):
    answer: str
    skills_used: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    escalated: bool
    reason: str


class TraceEvent(BaseModel):
    event: str
    trace_id: str
    session_id: str
    agent_or_skill: str
    ts_ms: int
    meta: dict[str, Any] = Field(default_factory=dict)


class SkillContract(BaseModel):
    name: SkillName
    description: str
    allowed_tools: list[str]
    forbidden_actions: list[str]
    max_tool_calls: int
    timeout_seconds: int
    fallback_behavior: str
    risk_level: RiskLevel

