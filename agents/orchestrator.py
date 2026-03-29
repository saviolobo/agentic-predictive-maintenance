"""LangGraph orchestrator — wires the 3 agents into a multi-agent pipeline."""
from __future__ import annotations

import os
from typing import TypedDict, Annotated, Any
import operator

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage

from configs.config import (
    LANGFUSE_TRACING, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST,
    PHOENIX_TRACING, PHOENIX_HOST,
)
from agents.sensor_monitor import run_sensor_analysis
from agents.rul_predictor import run_rul_prediction
from agents.maintenance_planner import run_maintenance_planning


# ── Tracing setup (pick one backend) ─────────────────────────────────────────

def _setup_tracing():
    if LANGFUSE_TRACING and LANGFUSE_PUBLIC_KEY:
        # Langfuse integrates via LangChain callback — registered per-chain
        # Set env vars so LangChain auto-picks it up via langfuse-langchain
        os.environ["LANGFUSE_PUBLIC_KEY"] = LANGFUSE_PUBLIC_KEY
        os.environ["LANGFUSE_SECRET_KEY"] = LANGFUSE_SECRET_KEY
        os.environ["LANGFUSE_HOST"] = LANGFUSE_HOST
        # Enable via LangChain callback handler (imported lazily)
        print("[tracing] Langfuse enabled")

    elif PHOENIX_TRACING:
        try:
            import phoenix as px
            from openinference.instrumentation.langchain import LangChainInstrumentor
            px.launch_app(host="0.0.0.0", port=6006)
            LangChainInstrumentor().instrument()
            print(f"[tracing] Phoenix enabled at {PHOENIX_HOST}")
        except ImportError:
            print("[tracing] Phoenix not installed. Run: pip install arize-phoenix openinference-instrumentation-langchain")

_setup_tracing()


# ── State schema ─────────────────────────────────────────────────────────────

class MaintenanceState(TypedDict):
    # Input
    unit_id: int
    cycle: int
    sensor_readings: dict
    feature_vector: list[float]
    feature_names: list[str]
    history: list[dict] | None

    # Agent outputs (accumulated)
    messages: Annotated[list[BaseMessage], operator.add]
    sensor_analysis: dict | None
    rul_analysis: dict | None
    maintenance_plan: dict | None

    # Final
    error: str | None
    completed: bool


# ── Node functions ────────────────────────────────────────────────────────────

def sensor_monitor_node(state: MaintenanceState) -> dict:
    """Node 1: Run sensor anomaly detection and trend analysis."""
    print(f"[orchestrator] Running Sensor Monitor for unit {state['unit_id']}...")
    try:
        result = run_sensor_analysis(
            unit_id=state["unit_id"],
            current_cycle=state["cycle"],
            sensor_readings=state["sensor_readings"],
            history=state.get("history"),
        )
        return {
            "sensor_analysis": result,
            "messages": [HumanMessage(content=f"Sensor analysis complete: {result['llm_response'][:200]}...")],
        }
    except Exception as e:
        return {"error": f"Sensor monitor failed: {str(e)}", "sensor_analysis": None}


def rul_predictor_node(state: MaintenanceState) -> dict:
    """Node 2: Run RUL prediction and explanation."""
    print(f"[orchestrator] Running RUL Predictor for unit {state['unit_id']}...")
    try:
        result = run_rul_prediction(
            unit_id=state["unit_id"],
            cycle=state["cycle"],
            feature_vector=state["feature_vector"],
            feature_names=state["feature_names"],
            sensor_analysis=state.get("sensor_analysis"),
        )
        return {
            "rul_analysis": result,
            "messages": [HumanMessage(content=f"RUL prediction: {result['predicted_rul']:.1f} cycles ({result['urgency']})")],
        }
    except Exception as e:
        return {"error": f"RUL predictor failed: {str(e)}", "rul_analysis": None}


def maintenance_planner_node(state: MaintenanceState) -> dict:
    """Node 3: Generate maintenance recommendations."""
    print(f"[orchestrator] Running Maintenance Planner for unit {state['unit_id']}...")
    rul_analysis = state.get("rul_analysis") or {}
    predicted_rul = rul_analysis.get("predicted_rul", 60.0) or 60.0
    urgency = rul_analysis.get("urgency", "UNKNOWN")

    try:
        result = run_maintenance_planning(
            unit_id=state["unit_id"],
            cycle=state["cycle"],
            predicted_rul=predicted_rul,
            urgency=urgency,
            sensor_analysis=state.get("sensor_analysis"),
            rul_analysis=rul_analysis,
        )
        return {
            "maintenance_plan": result,
            "completed": True,
            "messages": [HumanMessage(content=f"Maintenance plan ready: {result['tier']}")],
        }
    except Exception as e:
        return {"error": f"Maintenance planner failed: {str(e)}", "maintenance_plan": None, "completed": True}


def should_continue(state: MaintenanceState) -> str:
    """Route: stop on error, otherwise continue."""
    if state.get("error"):
        return "end"
    return "continue"


# ── Graph construction ────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(MaintenanceState)

    graph.add_node("sensor_monitor", sensor_monitor_node)
    graph.add_node("rul_predictor", rul_predictor_node)
    graph.add_node("maintenance_planner", maintenance_planner_node)

    graph.set_entry_point("sensor_monitor")

    graph.add_conditional_edges(
        "sensor_monitor",
        should_continue,
        {"continue": "rul_predictor", "end": END},
    )
    graph.add_conditional_edges(
        "rul_predictor",
        should_continue,
        {"continue": "maintenance_planner", "end": END},
    )
    graph.add_edge("maintenance_planner", END)

    return graph.compile()


# ── Public API ────────────────────────────────────────────────────────────────

_app = None


def get_app():
    global _app
    if _app is None:
        _app = build_graph()
    return _app


def analyze_engine(
    unit_id: int,
    cycle: int,
    sensor_readings: dict,
    feature_vector: list[float],
    feature_names: list[str],
    history: list[dict] | None = None,
) -> dict:
    """
    Main entry point: run the full multi-agent pipeline for one engine.

    Returns the final state with all agent outputs.
    """
    app = get_app()

    initial_state: MaintenanceState = {
        "unit_id": unit_id,
        "cycle": cycle,
        "sensor_readings": sensor_readings,
        "feature_vector": feature_vector,
        "feature_names": feature_names,
        "history": history,
        "messages": [],
        "sensor_analysis": None,
        "rul_analysis": None,
        "maintenance_plan": None,
        "error": None,
        "completed": False,
    }

    final_state = app.invoke(initial_state)
    return final_state


if __name__ == "__main__":
    # Quick smoke test with dummy data
    import numpy as np
    n_features = 42  # approximate after rolling features

    result = analyze_engine(
        unit_id=1,
        cycle=150,
        sensor_readings={f"sensor_{i}": float(np.random.rand()) for i in range(1, 22)},
        feature_vector=list(np.random.rand(n_features)),
        feature_names=[f"feat_{i}" for i in range(n_features)],
    )
    print("\n=== Final State ===")
    print(f"Sensor analysis: {bool(result.get('sensor_analysis'))}")
    print(f"RUL prediction:  {result.get('rul_analysis', {}).get('predicted_rul')}")
    print(f"Maintenance tier: {result.get('maintenance_plan', {}).get('tier')}")
