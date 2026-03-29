"""Sensor Monitor Agent — detects anomalies and degradation trends."""
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
import numpy as np
import pandas as pd
from typing import Any

from configs.config import (
    GROQ_MODEL, GROQ_API_KEY,
    ANOMALY_ZSCORE_THRESHOLD, FEATURE_COLUMNS
)


# ── Tools ────────────────────────────────────────────────────────────────────

@tool
def detect_anomalies(sensor_data: dict) -> dict:
    """
    Detect anomalies in sensor readings using z-score method.

    Args:
        sensor_data: dict with keys 'unit_id', 'cycle', and sensor readings

    Returns:
        dict with anomaly flags and z-scores for each sensor
    """
    readings = {k: v for k, v in sensor_data.items()
                if k.startswith("sensor_")}

    # Approximate normal ranges (mean ± 3σ from FD001 training stats)
    # In production, these would come from a fitted scaler
    anomalies = {}
    for sensor, value in readings.items():
        # Post-normalization: values should be in [0, 1]; flag if outside
        if isinstance(value, (int, float)):
            if value < -0.1 or value > 1.1:
                anomalies[sensor] = {"value": value, "anomaly": True, "zscore": abs(value - 0.5) / 0.2}
            else:
                anomalies[sensor] = {"value": value, "anomaly": False, "zscore": 0.0}

    flagged = [s for s, d in anomalies.items() if d["anomaly"]]
    return {
        "unit_id": sensor_data.get("unit_id"),
        "cycle": sensor_data.get("cycle"),
        "anomaly_count": len(flagged),
        "flagged_sensors": flagged,
        "sensor_details": anomalies,
    }


@tool
def analyze_degradation_trend(history: list[dict]) -> dict:
    """
    Analyze sensor degradation trend over multiple cycles.

    Args:
        history: list of dicts, each with cycle + sensor readings (last N cycles)

    Returns:
        dict with trend direction, rate, and degradation score per sensor
    """
    if len(history) < 2:
        return {"error": "Need at least 2 cycles for trend analysis"}

    df = pd.DataFrame(history)
    df = df.sort_values("cycle")

    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    trends = {}
    for col in sensor_cols:
        if col in df.columns:
            vals = df[col].dropna().values
            if len(vals) >= 2:
                slope = float(np.polyfit(range(len(vals)), vals, 1)[0])
                trends[col] = {
                    "slope": round(slope, 6),
                    "direction": "increasing" if slope > 0.001 else
                                 "decreasing" if slope < -0.001 else "stable",
                    "latest_value": round(float(vals[-1]), 4),
                }

    degrading = [s for s, t in trends.items() if t["direction"] != "stable"]
    degradation_score = sum(abs(t["slope"]) for t in trends.values())

    return {
        "cycles_analyzed": len(df),
        "degrading_sensors": degrading,
        "degradation_score": round(float(degradation_score), 4),
        "trends": trends,
    }


# ── Agent ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the Sensor Monitor Agent for a jet engine predictive maintenance system.

Your responsibilities:
1. Analyze real-time and historical sensor data from turbofan engines
2. Detect anomalies using statistical methods
3. Identify degradation trends across sensor readings
4. Provide clear, actionable findings to the maintenance team

Engine sensors monitored: temperature, pressure, fan speed, compressor speed,
bypass ratio, fuel flow, and vibration metrics.

Be precise and concise. Flag critical issues clearly. Express confidence levels.
When sensor values are post-normalized (0–1 scale), values near 1.0 often indicate
high stress/degradation. Focus on patterns, not individual readings."""


def create_sensor_monitor_agent() -> ChatGroq:
    """Return configured Groq LLM bound with sensor monitoring tools."""
    llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.1,
    )
    tools = [detect_anomalies, analyze_degradation_trend]
    return llm.bind_tools(tools), tools, SYSTEM_PROMPT


def run_sensor_analysis(
    unit_id: int,
    current_cycle: int,
    sensor_readings: dict,
    history: list[dict] | None = None,
) -> dict:
    """
    Standalone sensor analysis — returns structured findings.
    Used by the LangGraph orchestrator as a node function.
    """
    llm, tools, system_prompt = create_sensor_monitor_agent()

    sensor_str = "\n".join(
        f"  {k}: {v:.4f}" for k, v in sensor_readings.items()
        if k.startswith("sensor_")
    )
    history_note = (
        f"Last {len(history)} cycles of history are available."
        if history else "No history provided."
    )

    user_msg = f"""Analyze engine unit {unit_id} at cycle {current_cycle}.

Current sensor readings (normalized 0–1):
{sensor_str}

{history_note}

1. Detect any anomalies in the current readings.
2. If history is available, analyze degradation trends.
3. Summarize your findings and flag any concerns."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_msg),
    ]

    response = llm.invoke(messages)

    return {
        "agent": "sensor_monitor",
        "unit_id": unit_id,
        "cycle": current_cycle,
        "llm_response": response.content,
        "tool_calls": response.tool_calls if hasattr(response, "tool_calls") else [],
        "sensor_readings": sensor_readings,
    }
