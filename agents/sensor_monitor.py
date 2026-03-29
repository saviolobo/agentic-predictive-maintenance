"""Sensor Monitor Agent — detects anomalies and degradation trends."""
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
import numpy as np
import pandas as pd
from typing import Any

from configs.config import (
    GROQ_MODEL, GROQ_API_KEY, require_groq_key,
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


def _plain_llm() -> ChatGroq:
    require_groq_key()
    return ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0.1)


def _compute_anomalies(sensor_readings: dict) -> dict:
    """Run anomaly detection in Python (no LLM needed)."""
    flagged = []
    details = {}
    for k, v in sensor_readings.items():
        if k.startswith("sensor_") and isinstance(v, (int, float)):
            anomaly = v < -0.1 or v > 1.1
            details[k] = {"value": round(v, 4), "anomaly": anomaly}
            if anomaly:
                flagged.append(k)
    return {"flagged_sensors": flagged, "anomaly_count": len(flagged), "details": details}


def _compute_trends(history: list[dict]) -> dict:
    """Compute degradation trends in Python (no LLM needed)."""
    if not history or len(history) < 2:
        return {}
    df = pd.DataFrame(history).sort_values("cycle")
    trends = {}
    for col in [c for c in df.columns if c.startswith("sensor_")]:
        vals = df[col].dropna().values
        if len(vals) >= 2:
            slope = float(np.polyfit(range(len(vals)), vals, 1)[0])
            trends[col] = {
                "slope": round(slope, 6),
                "direction": "increasing" if slope > 0.001 else "decreasing" if slope < -0.001 else "stable",
                "latest": round(float(vals[-1]), 4),
            }
    degrading = [s for s, t in trends.items() if t["direction"] != "stable"]
    return {
        "degrading_sensors": degrading,
        "degradation_score": round(sum(abs(t["slope"]) for t in trends.values()), 4),
        "trends": trends,
    }


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
    # Pre-compute in Python — pass results to LLM for reasoning only
    anomaly_result = _compute_anomalies(sensor_readings)
    trend_result = _compute_trends(history) if history else {}

    sensor_str = "\n".join(
        f"  {k}: {v:.4f}" for k, v in sensor_readings.items()
        if k.startswith("sensor_")
    )
    anomaly_str = (
        f"Flagged sensors ({anomaly_result['anomaly_count']}): {anomaly_result['flagged_sensors']}"
        if anomaly_result["anomaly_count"] > 0 else "No out-of-range sensors detected."
    )
    trend_str = ""
    if trend_result:
        score = trend_result.get("degradation_score", 0)
        degrading = trend_result.get("degrading_sensors", [])
        trend_str = f"\nDegradation score: {score:.4f}. Trending sensors: {degrading or 'none'}."

    user_msg = f"""Analyze engine unit {unit_id} at cycle {current_cycle}.

Sensor readings (normalized 0–1):
{sensor_str}

Anomaly check: {anomaly_str}{trend_str}

Provide a concise health assessment: what the sensors indicate about engine condition,
any concerns, and confidence level. 2–3 short paragraphs max."""

    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_msg)]
    response = _plain_llm().invoke(messages)

    return {
        "agent": "sensor_monitor",
        "unit_id": unit_id,
        "cycle": current_cycle,
        "llm_response": response.content,
        "anomaly_count": anomaly_result["anomaly_count"],
        "flagged_sensors": anomaly_result["flagged_sensors"],
        "degradation_score": trend_result.get("degradation_score", 0.0),
        "sensor_readings": sensor_readings,
    }
