"""Maintenance Planner Agent — generates prioritized maintenance recommendations."""
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from typing import Any

from configs.config import (
    GROQ_MODEL, GROQ_API_KEY, require_groq_key,
    CRITICAL_RUL_THRESHOLD, WARNING_RUL_THRESHOLD,
)


# ── Tools ────────────────────────────────────────────────────────────────────

@tool
def calculate_maintenance_priority(
    unit_id: int,
    predicted_rul: float,
    degradation_score: float,
    anomaly_count: int,
) -> dict:
    """
    Calculate maintenance priority score (0–100) for a given engine.

    Args:
        unit_id: engine identifier
        predicted_rul: predicted remaining useful life in cycles
        degradation_score: degradation rate from sensor trend analysis
        anomaly_count: number of anomalous sensors detected

    Returns:
        dict with priority score, tier, and recommended action window
    """
    # Priority formula: weighted combination
    rul_score = max(0, 100 - (predicted_rul / 1.25))  # 0 RUL → 100, 125 RUL → 0
    degradation_component = min(30, degradation_score * 1000)
    anomaly_component = min(20, anomaly_count * 5)

    priority = min(100, rul_score * 0.6 + degradation_component + anomaly_component)

    tier = (
        "P1 - IMMEDIATE" if priority >= 80 else
        "P2 - URGENT" if priority >= 60 else
        "P3 - PLANNED" if priority >= 40 else
        "P4 - MONITOR"
    )

    action_window = (
        "Within 24 hours" if priority >= 80 else
        "Within 3 days" if priority >= 60 else
        "Within 2 weeks" if priority >= 40 else
        "Next scheduled maintenance"
    )

    return {
        "unit_id": unit_id,
        "priority_score": round(float(priority), 1),
        "tier": tier,
        "action_window": action_window,
        "rul_component": round(float(rul_score * 0.6), 1),
        "degradation_component": round(float(degradation_component), 1),
        "anomaly_component": round(float(anomaly_component), 1),
    }


@tool
def generate_maintenance_schedule(engines: list[dict]) -> dict:
    """
    Generate a prioritized maintenance schedule for multiple engines.

    Args:
        engines: list of dicts with unit_id, predicted_rul, priority_score

    Returns:
        dict with sorted schedule and resource allocation suggestions
    """
    sorted_engines = sorted(
        engines,
        key=lambda x: (-x.get("priority_score", 0), x.get("predicted_rul", 999))
    )

    schedule = []
    for i, eng in enumerate(sorted_engines):
        schedule.append({
            "rank": i + 1,
            "unit_id": eng["unit_id"],
            "predicted_rul": eng.get("predicted_rul", "N/A"),
            "priority_score": eng.get("priority_score", 0),
            "tier": eng.get("tier", "UNKNOWN"),
            "action_window": eng.get("action_window", "TBD"),
        })

    critical = sum(1 for e in sorted_engines if e.get("priority_score", 0) >= 80)
    urgent = sum(1 for e in sorted_engines if 60 <= e.get("priority_score", 0) < 80)

    return {
        "schedule": schedule,
        "summary": {
            "total_engines": len(engines),
            "critical_count": critical,
            "urgent_count": urgent,
            "technicians_needed": max(1, critical * 2 + urgent),
        }
    }


@tool
def estimate_maintenance_cost(
    unit_id: int,
    urgency: str,
    anomaly_count: int,
) -> dict:
    """
    Estimate maintenance cost and ROI for a given engine.

    Args:
        unit_id: engine identifier
        urgency: CRITICAL / WARNING / NORMAL
        anomaly_count: number of anomalous sensors

    Returns:
        dict with cost estimates and ROI analysis
    """
    # Cost model (illustrative — would use real cost data in production)
    base_cost = 50_000  # USD, routine inspection

    multiplier = {
        "CRITICAL": 4.0,   # emergency + likely parts replacement
        "WARNING": 2.0,    # planned maintenance + parts
        "NORMAL": 1.0,     # routine inspection
    }.get(urgency, 1.5)

    parts_cost = anomaly_count * 8_000  # estimated parts per failed sensor area
    labor_cost = base_cost * multiplier
    total_cost = labor_cost + parts_cost

    # Unplanned failure cost estimate
    unplanned_failure_cost = 2_500_000  # avg engine AOG (Aircraft on Ground) cost

    roi_ratio = unplanned_failure_cost / max(total_cost, 1)

    return {
        "unit_id": unit_id,
        "estimated_cost_usd": round(total_cost),
        "breakdown": {
            "labor_usd": round(labor_cost),
            "parts_usd": round(parts_cost),
        },
        "unplanned_failure_cost_usd": unplanned_failure_cost,
        "roi_ratio": round(roi_ratio, 1),
        "recommendation": (
            f"Proactive maintenance saves ~${(unplanned_failure_cost - total_cost)/1000:.0f}K "
            f"vs. failure (ROI: {roi_ratio:.1f}x)"
        ),
    }


# ── Agent ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the Maintenance Planner Agent for a jet engine predictive maintenance system.

Your responsibilities:
1. Generate actionable maintenance recommendations based on RUL predictions
2. Prioritize engines by maintenance urgency and safety risk
3. Provide cost-benefit analysis for proactive maintenance
4. Create clear action plans for maintenance teams

Guiding principles:
- Safety first: CRITICAL engines always get immediate attention
- Economic efficiency: balance maintenance costs against failure risk
- Operational continuity: minimize unplanned downtime
- Data-driven: base all recommendations on sensor evidence and RUL predictions

Output clear, concise maintenance orders that a technician or fleet manager can act on immediately.
Include: what to inspect, why, when, and expected cost/benefit."""


def _plain_llm() -> ChatGroq:
    require_groq_key()
    return ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0.2)


def run_maintenance_planning(
    unit_id: int,
    cycle: int,
    predicted_rul: float,
    urgency: str,
    sensor_analysis: dict | None = None,
    rul_analysis: dict | None = None,
) -> dict:
    """Priority scoring in Python, LLM for recommendations only — no tool binding."""
    anomaly_count = sensor_analysis.get("anomaly_count", 0) if sensor_analysis else 0
    degradation_score = sensor_analysis.get("degradation_score", 0.0) if sensor_analysis else 0.0

    # Compute priority score in Python
    rul_score = max(0, 100 - (predicted_rul / 1.25))
    priority_score = min(100, rul_score * 0.6 + min(30, degradation_score * 1000) + min(20, anomaly_count * 5))
    tier = (
        "P1 - IMMEDIATE" if priority_score >= 80 else
        "P2 - URGENT"    if priority_score >= 60 else
        "P3 - PLANNED"   if priority_score >= 40 else
        "P4 - MONITOR"
    )

    # Cost estimate
    base = 50_000
    mult = {"CRITICAL": 4.0, "WARNING": 2.0, "NORMAL": 1.0}.get(urgency, 1.5)
    est_cost = round(base * mult + anomaly_count * 8_000)
    savings = round((2_500_000 - est_cost) / 1_000)

    sensor_ctx = f"\nSensor findings:\n{sensor_analysis['llm_response']}\n" if sensor_analysis and sensor_analysis.get("llm_response") else ""
    rul_ctx = f"\nRUL analysis:\n{rul_analysis['llm_response']}\n" if rul_analysis and rul_analysis.get("llm_response") else ""

    user_msg = f"""Generate a maintenance plan for engine unit {unit_id}.

Status: Cycle {cycle} | RUL {predicted_rul:.1f} cycles | {urgency} | Priority {tier} (score {priority_score:.0f}/100)
Estimated proactive maintenance cost: ${est_cost:,} | Saves ~${savings}K vs unplanned failure
{sensor_ctx}{rul_ctx}
Provide: action window, specific inspection steps, components to check, and any operational restrictions.
Be direct and actionable — write for a maintenance technician."""

    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_msg)]
    response = _plain_llm().invoke(messages)

    return {
        "agent": "maintenance_planner",
        "unit_id": unit_id,
        "cycle": cycle,
        "priority_score": round(float(priority_score), 1),
        "tier": tier,
        "urgency": urgency,
        "predicted_rul": predicted_rul,
        "estimated_cost_usd": est_cost,
        "llm_response": response.content,
        "anomaly_count": anomaly_count,
    }
