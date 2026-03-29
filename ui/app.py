"""Streamlit UI — Jet Engine Predictive Maintenance Dashboard."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import (
    DATA_PROCESSED_DIR, MODELS_DIR,
    CRITICAL_RUL_THRESHOLD, WARNING_RUL_THRESHOLD,
)

st.set_page_config(
    page_title="Jet Engine Predictive Maintenance",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ───────────────────────────────────────────────────────────────────

st.markdown("""
<style>
.metric-critical { background: #fee2e2; border-left: 4px solid #ef4444; padding: 8px 12px; border-radius: 4px; }
.metric-warning  { background: #fef3c7; border-left: 4px solid #f59e0b; padding: 8px 12px; border-radius: 4px; }
.metric-normal   { background: #d1fae5; border-left: 4px solid #10b981; padding: 8px 12px; border-radius: 4px; }
.agent-box { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; margin-bottom: 12px; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ─────────────────────────────────────────────────────────────

@st.cache_data
def load_test_data():
    path = DATA_PROCESSED_DIR / "test_FD001_last.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


@st.cache_data
def load_full_test():
    path = DATA_PROCESSED_DIR / "test_FD001_full.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


@st.cache_resource
def load_model():
    import joblib
    path = MODELS_DIR / "xgb_rul_FD001.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


def get_rul_color(rul):
    if rul <= CRITICAL_RUL_THRESHOLD:
        return "#ef4444"
    elif rul <= WARNING_RUL_THRESHOLD:
        return "#f59e0b"
    return "#10b981"


def get_urgency(rul):
    if rul <= CRITICAL_RUL_THRESHOLD:
        return "CRITICAL"
    elif rul <= WARNING_RUL_THRESHOLD:
        return "WARNING"
    return "NORMAL"


# ── Header ───────────────────────────────────────────────────────────────────

st.title("✈️ Jet Engine Predictive Maintenance")
st.caption("Multi-agent AI system powered by LangGraph · Groq · NASA C-MAPSS FD001")

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Configuration")

    data = load_test_data()
    model = load_model()

    if data is None:
        st.error("Data not found. Run the setup pipeline first.")
        st.code("python tools/download_data.py\npython tools/data_pipeline.py\npython tools/train_model.py")
        st.stop()

    if model is None:
        st.warning("Model not trained. Run: `python tools/train_model.py`")

    unit_ids = sorted(data["unit_id"].unique())
    selected_unit = st.selectbox("Select Engine Unit", unit_ids, index=0)

    st.divider()
    st.subheader("System Status")
    st.success("Data pipeline: Ready")
    st.success(f"Model: {'Ready' if model else 'Not trained'}")

    agents_ready = bool(Path(".env").exists())
    if agents_ready:
        st.success("LLM Agents: Configured")
    else:
        st.warning("LLM Agents: Add .env file")

    st.divider()
    run_agents = st.button(
        "Run Multi-Agent Analysis",
        type="primary",
        disabled=(model is None),
        help="Runs Sensor Monitor → RUL Predictor → Maintenance Planner"
    )


# ── Main dashboard ───────────────────────────────────────────────────────────

engine_row = data[data["unit_id"] == selected_unit].iloc[0]
from tools.data_pipeline import get_feature_columns_with_rolling
feature_cols = get_feature_columns_with_rolling()
feature_cols = [c for c in feature_cols if c in data.columns]

# Compute predicted RUL
predicted_rul = None
if model:
    X = engine_row[feature_cols].values.reshape(1, -1)
    predicted_rul = float(max(0, model.predict(X)[0]))

true_rul = float(engine_row["RUL"])
cycle = int(engine_row["cycle"])
urgency = get_urgency(predicted_rul or true_rul)
urgency_color = get_rul_color(predicted_rul or true_rul)

# ── KPI row ──────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Engine Unit", f"#{selected_unit}")
with col2:
    st.metric("Current Cycle", cycle)
with col3:
    if predicted_rul is not None:
        delta = f"{predicted_rul - true_rul:+.1f} vs true"
        st.metric("Predicted RUL", f"{predicted_rul:.0f} cycles", delta=delta)
    else:
        st.metric("True RUL", f"{true_rul:.0f} cycles")
with col4:
    badge_class = f"metric-{urgency.lower()}"
    st.markdown(
        f'<div class="{badge_class}"><b>Status</b><br/><span style="font-size:1.4em">{urgency}</span></div>',
        unsafe_allow_html=True,
    )

st.divider()

# ── Charts row ───────────────────────────────────────────────────────────────

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Sensor Readings (Current Cycle)")

    sensor_cols_plot = [c for c in feature_cols if c.startswith("sensor_") and not c.endswith(("_roll_mean", "_roll_std"))]
    sensor_vals = [float(engine_row[c]) for c in sensor_cols_plot]

    fig = go.Figure(go.Bar(
        x=sensor_cols_plot,
        y=sensor_vals,
        marker_color=[
            "#ef4444" if v > 0.85 else "#f59e0b" if v > 0.7 else "#10b981"
            for v in sensor_vals
        ],
    ))
    fig.update_layout(
        xaxis_title="Sensor", yaxis_title="Normalized Value",
        height=320, margin=dict(t=10, b=40),
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(range=[0, 1.1], gridcolor="#f0f0f0"),
    )
    st.plotly_chart(fig, use_container_width=True)


with col_right:
    st.subheader("Fleet Overview")

    if model:
        all_preds = model.predict(data[feature_cols].values)
        all_preds = np.maximum(0, all_preds)
    else:
        all_preds = data["RUL"].values

    fleet_df = pd.DataFrame({
        "unit_id": data["unit_id"].values,
        "predicted_rul": all_preds,
        "true_rul": data["RUL"].values,
    })
    fleet_df["urgency"] = fleet_df["predicted_rul"].apply(get_urgency)

    counts = fleet_df["urgency"].value_counts()
    colors = {"CRITICAL": "#ef4444", "WARNING": "#f59e0b", "NORMAL": "#10b981"}

    fig2 = go.Figure(go.Pie(
        labels=list(counts.index),
        values=list(counts.values),
        marker_colors=[colors.get(u, "#94a3b8") for u in counts.index],
        hole=0.45,
    ))
    fig2.update_layout(height=280, margin=dict(t=10, b=10), showlegend=True)
    st.plotly_chart(fig2, use_container_width=True)

    st.caption(f"Fleet size: {len(fleet_df)} engines")


# ── Engine history plot ───────────────────────────────────────────────────────

full_test = load_full_test()
if full_test is not None:
    st.subheader(f"Engine #{selected_unit} — Sensor Trend History")

    eng_hist = full_test[full_test["unit_id"] == selected_unit].copy()

    # Pick 3 most informative sensors (high variance in FD001)
    watch_sensors = ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_11",
                     "sensor_12", "sensor_15", "sensor_17", "sensor_20", "sensor_21"]
    watch_sensors = [s for s in watch_sensors if s in eng_hist.columns][:4]

    fig3 = go.Figure()
    for s in watch_sensors:
        fig3.add_trace(go.Scatter(
            x=eng_hist["cycle"], y=eng_hist[s],
            mode="lines", name=s, line=dict(width=1.5),
        ))
    fig3.update_layout(
        xaxis_title="Cycle", yaxis_title="Normalized Value",
        height=300, margin=dict(t=10, b=40),
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(gridcolor="#f0f0f0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig3, use_container_width=True)


# ── Fleet table ───────────────────────────────────────────────────────────────

st.subheader("Fleet Status Table")

fleet_display = fleet_df.copy()
fleet_display["priority"] = fleet_display["predicted_rul"].apply(
    lambda r: "P1" if r <= CRITICAL_RUL_THRESHOLD else
              "P2" if r <= WARNING_RUL_THRESHOLD else "P4"
)
fleet_display = fleet_display.sort_values("predicted_rul")
fleet_display.columns = ["Unit ID", "Predicted RUL", "True RUL", "Status", "Priority"]
fleet_display["Predicted RUL"] = fleet_display["Predicted RUL"].round(1)
fleet_display["True RUL"] = fleet_display["True RUL"].round(1)

def highlight_status(row):
    if row["Status"] == "CRITICAL":
        return ["background-color: #fee2e2"] * len(row)
    elif row["Status"] == "WARNING":
        return ["background-color: #fef3c7"] * len(row)
    return [""] * len(row)

st.dataframe(
    fleet_display.style.apply(highlight_status, axis=1),
    use_container_width=True,
    height=280,
)


# ── Multi-agent analysis panel ────────────────────────────────────────────────

if run_agents:
    st.divider()
    st.subheader("Multi-Agent Analysis")

    with st.spinner("Running LangGraph pipeline: Sensor Monitor → RUL Predictor → Maintenance Planner..."):
        try:
            from agents.orchestrator import analyze_engine

            sensor_readings = {
                col: float(engine_row[col])
                for col in feature_cols
                if col.startswith("sensor_") and not col.endswith(("_roll_mean", "_roll_std"))
            }
            fv = [float(engine_row[c]) for c in feature_cols]

            result = analyze_engine(
                unit_id=selected_unit,
                cycle=cycle,
                sensor_readings=sensor_readings,
                feature_vector=fv,
                feature_names=feature_cols,
            )

            # Display results
            agent_col1, agent_col2, agent_col3 = st.columns(3)

            with agent_col1:
                st.markdown("#### Sensor Monitor Agent")
                sa = result.get("sensor_analysis") or {}
                st.markdown(f'<div class="agent-box">{sa.get("llm_response", "No response")}</div>',
                           unsafe_allow_html=True)

            with agent_col2:
                st.markdown("#### RUL Predictor Agent")
                ra = result.get("rul_analysis") or {}
                rul_pred = ra.get("predicted_rul")
                if rul_pred:
                    st.metric("Model Prediction", f"{rul_pred:.1f} cycles")
                st.markdown(f'<div class="agent-box">{ra.get("llm_response", "No response")}</div>',
                           unsafe_allow_html=True)

            with agent_col3:
                st.markdown("#### Maintenance Planner Agent")
                mp = result.get("maintenance_plan") or {}
                tier = mp.get("tier", "")
                if tier:
                    st.info(f"**{tier}** · Score: {mp.get('priority_score', 0):.0f}/100")
                st.markdown(f'<div class="agent-box">{mp.get("llm_response", "No response")}</div>',
                           unsafe_allow_html=True)

            if result.get("error"):
                st.error(f"Pipeline error: {result['error']}")

        except Exception as e:
            st.error(f"Failed to run agents: {e}")
            st.info("Make sure GROQ_API_KEY is set in your .env file and requirements are installed.")


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("NASA C-MAPSS FD001 · XGBoost RUL Model · LangGraph Multi-Agent · Groq llama-3.3-70b-versatile")
