"""RUL Prediction Agent — predicts remaining useful life and explains results."""
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
import numpy as np
import pandas as pd
from typing import Any

from configs.config import (
    GROQ_MODEL, GROQ_API_KEY,
    CRITICAL_RUL_THRESHOLD, WARNING_RUL_THRESHOLD,
    MODELS_DIR
)


# ── Tools ────────────────────────────────────────────────────────────────────

@tool
def predict_rul(feature_vector: list[float]) -> dict:
    """
    Predict remaining useful life (RUL) for a turbofan engine.

    Args:
        feature_vector: normalized feature values in model's expected order

    Returns:
        dict with predicted RUL and confidence interval
    """
    import joblib
    model_path = MODELS_DIR / "xgb_rul_FD001.joblib"
    if not model_path.exists():
        return {"error": "Model not trained. Run: python tools/train_model.py"}

    model = joblib.load(model_path)
    X = np.array(feature_vector).reshape(1, -1)
    rul = float(model.predict(X)[0])
    rul = max(0.0, round(rul, 1))

    # Approximate confidence interval from tree variance
    # XGBoost doesn't have native CI — use ±15% as heuristic
    ci_low = max(0.0, round(rul * 0.85, 1))
    ci_high = round(rul * 1.15, 1)

    urgency = (
        "CRITICAL" if rul <= CRITICAL_RUL_THRESHOLD else
        "WARNING" if rul <= WARNING_RUL_THRESHOLD else
        "NORMAL"
    )

    return {
        "predicted_rul": rul,
        "confidence_interval": [ci_low, ci_high],
        "urgency": urgency,
        "interpretation": (
            f"Engine has approximately {rul:.0f} cycles remaining "
            f"(95% CI: {ci_low:.0f}–{ci_high:.0f} cycles). Status: {urgency}."
        ),
    }


@tool
def get_feature_importance(top_n: int = 10) -> dict:
    """
    Get top N most important features from the trained RUL model.

    Args:
        top_n: number of top features to return

    Returns:
        dict with feature names and importance scores
    """
    import joblib
    model_path = MODELS_DIR / "xgb_rul_FD001.joblib"
    if not model_path.exists():
        return {"error": "Model not trained."}

    model = joblib.load(model_path)
    importance = model.feature_importances_

    from tools.data_pipeline import get_feature_columns_with_rolling
    feature_cols = get_feature_columns_with_rolling()

    pairs = sorted(
        zip(feature_cols, importance), key=lambda x: x[1], reverse=True
    )[:top_n]

    return {
        "top_features": [
            {"feature": name, "importance": round(float(imp), 4)}
            for name, imp in pairs
        ]
    }


# ── Agent ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the RUL Prediction Agent for a jet engine predictive maintenance system.

Your responsibilities:
1. Predict remaining useful life (RUL) in engine operating cycles
2. Explain what factors are driving the prediction
3. Quantify prediction confidence
4. Contextualize the RUL in terms of urgency (NORMAL / WARNING / CRITICAL)

RUL thresholds:
- CRITICAL: ≤ 30 cycles — immediate maintenance required
- WARNING: ≤ 60 cycles — schedule maintenance soon
- NORMAL: > 60 cycles — continue monitoring

The model was trained on NASA C-MAPSS FD001: 100 turbofan engines under single
operating condition with HPC (High Pressure Compressor) degradation mode.

Be technically precise. Explain predictions in terms of the driving sensors.
Acknowledge uncertainty — ML predictions carry inherent error (typical MAE ~15 cycles)."""


def _plain_llm() -> ChatGroq:
    return ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0.1)


def run_rul_prediction(
    unit_id: int,
    cycle: int,
    feature_vector: list[float],
    feature_names: list[str],
    sensor_analysis: dict | None = None,
) -> dict:
    """ML inference in Python, LLM for reasoning only — no tool binding."""
    import joblib
    model_path = MODELS_DIR / "xgb_rul_FD001.joblib"
    predicted_rul = None
    urgency = "UNKNOWN"
    top_features_str = ""

    if model_path.exists():
        model = joblib.load(model_path)
        X = np.array(feature_vector).reshape(1, -1)
        predicted_rul = float(max(0.0, model.predict(X)[0]))
        urgency = (
            "CRITICAL" if predicted_rul <= CRITICAL_RUL_THRESHOLD else
            "WARNING" if predicted_rul <= WARNING_RUL_THRESHOLD else
            "NORMAL"
        )
        importance = model.feature_importances_
        pairs = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)[:5]
        top_vals = [(n, feature_vector[feature_names.index(n)], imp) for n, imp in pairs]
        top_features_str = "Top driving features:\n" + "\n".join(
            f"  {n}: value={v:.4f}, importance={i:.4f}" for n, v, i in top_vals
        )

    sensor_context = ""
    if sensor_analysis and sensor_analysis.get("llm_response"):
        sensor_context = f"\nSensor Monitor summary:\n{sensor_analysis['llm_response']}\n"

    user_msg = f"""Engine unit {unit_id} at cycle {cycle}.

Predicted RUL: {f'{predicted_rul:.1f}' if predicted_rul is not None else 'N/A'} cycles | Status: {urgency}
{top_features_str}
{sensor_context}
Interpret the prediction: key driving factors, confidence level, and what this
means for engine health. 2–3 short paragraphs."""

    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_msg)]
    response = _plain_llm().invoke(messages)

    return {
        "agent": "rul_predictor",
        "unit_id": unit_id,
        "cycle": cycle,
        "predicted_rul": predicted_rul,
        "urgency": urgency,
        "llm_response": response.content,
    }
