"""Streamlit dashboard — Jet Engine Predictive Maintenance."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import (
    DATA_PROCESSED_DIR, MODELS_DIR,
    CRITICAL_RUL_THRESHOLD, WARNING_RUL_THRESHOLD,
)

st.set_page_config(
    page_title="Engine Health Monitor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  /* Dark base */
  .stApp { background-color: #0f1117; color: #e2e8f0; }
  section[data-testid="stSidebar"] { background-color: #1a1d27; }
  section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

  /* Cards */
  .card {
    background: #1e2130; border-radius: 10px;
    padding: 20px 24px; margin-bottom: 16px;
    border: 1px solid #2d3148;
  }
  .card-critical { border-left: 4px solid #ef4444; }
  .card-warning  { border-left: 4px solid #f59e0b; }
  .card-normal   { border-left: 4px solid #10b981; }

  /* Status badges */
  .badge { display:inline-block; padding:4px 12px; border-radius:20px;
           font-weight:700; font-size:0.85rem; letter-spacing:0.05em; }
  .badge-critical { background:#450a0a; color:#fca5a5; }
  .badge-warning  { background:#451a03; color:#fcd34d; }
  .badge-normal   { background:#052e16; color:#6ee7b7; }

  /* Agent output boxes */
  .agent-output {
    background:#151823; border:1px solid #2d3148; border-radius:8px;
    padding:16px; font-size:0.88rem; line-height:1.65;
    color:#cbd5e1; white-space:pre-wrap; max-height:380px; overflow-y:auto;
  }
  .agent-header {
    font-size:0.75rem; font-weight:700; letter-spacing:0.1em;
    text-transform:uppercase; margin-bottom:8px;
  }
  .agent-header-blue   { color:#60a5fa; }
  .agent-header-purple { color:#a78bfa; }
  .agent-header-green  { color:#34d399; }

  /* Metrics */
  [data-testid="stMetricValue"] { color:#f1f5f9 !important; font-size:2rem !important; }
  [data-testid="stMetricLabel"] { color:#94a3b8 !important; }
  [data-testid="stMetricDelta"] { font-size:0.8rem !important; }

  /* Dividers */
  hr { border-color:#2d3148; }

  /* Dataframe */
  [data-testid="stDataFrame"] { border-radius:8px; overflow:hidden; }

  /* Hide default streamlit elements */
  #MainMenu, footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def urgency(rul):
    if rul <= CRITICAL_RUL_THRESHOLD: return "CRITICAL"
    if rul <= WARNING_RUL_THRESHOLD:  return "WARNING"
    return "NORMAL"

def rul_color(rul):
    if rul <= CRITICAL_RUL_THRESHOLD: return "#ef4444"
    if rul <= WARNING_RUL_THRESHOLD:  return "#f59e0b"
    return "#10b981"

def badge_html(status):
    cls = {"CRITICAL": "badge-critical", "WARNING": "badge-warning", "NORMAL": "badge-normal"}.get(status, "")
    return f'<span class="badge {cls}">{status}</span>'


# ── Data loaders ─────────────────────────────────────────────────────────────

@st.cache_data
def load_test_last():
    p = DATA_PROCESSED_DIR / "test_FD001_last.parquet"
    return pd.read_parquet(p) if p.exists() else None

@st.cache_data
def load_test_full():
    p = DATA_PROCESSED_DIR / "test_FD001_full.parquet"
    return pd.read_parquet(p) if p.exists() else None

@st.cache_resource
def load_model():
    p = MODELS_DIR / "xgb_rul_FD001.joblib"
    if not p.exists(): return None
    import joblib
    return joblib.load(p)

from tools.data_pipeline import get_feature_columns_with_rolling

@st.cache_data
def get_feature_cols(data_cols):
    fc = get_feature_columns_with_rolling()
    return [c for c in fc if c in data_cols]


# ── Load data ─────────────────────────────────────────────────────────────────

data = load_test_last()
model = load_model()

if data is None:
    st.error("No processed data found. Run: `python -m tools.data_pipeline`")
    st.stop()

feature_cols = get_feature_cols(tuple(data.columns))

# Compute fleet predictions
if model:
    fleet_preds = np.maximum(0, model.predict(data[feature_cols].values))
else:
    fleet_preds = data["RUL"].values

fleet_df = pd.DataFrame({
    "unit_id": data["unit_id"].values,
    "predicted_rul": fleet_preds.round(1),
    "true_rul": data["RUL"].values.round(1),
    "cycle": data["cycle"].values,
})
fleet_df["status"] = fleet_df["predicted_rul"].apply(urgency)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ✈️ Engine Health Monitor")
    st.caption("NASA C-MAPSS FD001 · XGBoost + LLM Agents")
    st.divider()

    unit_ids = sorted(data["unit_id"].unique())
    selected_unit = st.selectbox("Select Engine", unit_ids,
                                  format_func=lambda x: f"Engine #{x:03d}")

    st.divider()
    st.markdown("**System Status**")
    st.success("Data pipeline ready")
    st.success("XGBoost model loaded" if model else "Model not trained")
    api_ok = bool(Path(".env").exists())
    if api_ok:
        st.success("Groq API configured")
    else:
        st.warning("Add GROQ_API_KEY to .env")

    st.divider()
    run_agents = st.button("▶ Run Agent Analysis", type="primary",
                            disabled=(not model or not api_ok),
                            use_container_width=True)

    st.divider()
    # Fleet summary in sidebar
    n_critical = (fleet_df["status"] == "CRITICAL").sum()
    n_warning  = (fleet_df["status"] == "WARNING").sum()
    n_normal   = (fleet_df["status"] == "NORMAL").sum()
    st.markdown("**Fleet Summary**")
    st.markdown(f"🔴 Critical: **{n_critical}**")
    st.markdown(f"🟡 Warning: **{n_warning}**")
    st.markdown(f"🟢 Normal: **{n_normal}**")


# ── Selected engine row ───────────────────────────────────────────────────────

row = data[data["unit_id"] == selected_unit].iloc[0]
pred_rul = float(fleet_df[fleet_df["unit_id"] == selected_unit]["predicted_rul"].iloc[0])
true_rul = float(row["RUL"])
cycle    = int(row["cycle"])
status   = urgency(pred_rul)
color    = rul_color(pred_rul)


# ── Header row ────────────────────────────────────────────────────────────────

st.markdown(f"## Engine #{selected_unit:03d} &nbsp; {badge_html(status)}",
            unsafe_allow_html=True)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Current Cycle", cycle)
m2.metric("Predicted RUL", f"{pred_rul:.0f} cycles")
m3.metric("True RUL", f"{true_rul:.0f} cycles")
m4.metric("Error", f"{abs(pred_rul - true_rul):.1f} cycles")
m5.metric("Fleet Size", f"{len(fleet_df)} engines")

st.divider()


# ── Charts ────────────────────────────────────────────────────────────────────

col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown("#### Sensor Readings")
    raw_sensors = [c for c in feature_cols if c.startswith("sensor_")
                   and not c.endswith(("_roll_mean", "_roll_std"))]
    vals = [float(row[c]) for c in raw_sensors]

    fig = go.Figure(go.Bar(
        x=raw_sensors, y=vals,
        marker_color=[rul_color(1 - v) if v > 0.7 else "#3b82f6" for v in vals],
        hovertemplate="%{x}: %{y:.4f}<extra></extra>",
    ))
    fig.add_hline(y=0.85, line_dash="dot", line_color="#ef4444",
                  annotation_text="High stress", annotation_font_color="#ef4444")
    fig.update_layout(
        height=300, margin=dict(t=10, b=40, l=10, r=10),
        plot_bgcolor="#151823", paper_bgcolor="#1e2130",
        font_color="#94a3b8",
        xaxis=dict(gridcolor="#2d3148", tickangle=-45),
        yaxis=dict(range=[0, 1.15], gridcolor="#2d3148", title="Normalized value"),
    )
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.markdown("#### Fleet Health Distribution")
    counts = fleet_df["status"].value_counts()
    colors_map = {"CRITICAL": "#ef4444", "WARNING": "#f59e0b", "NORMAL": "#10b981"}

    fig2 = go.Figure(go.Pie(
        labels=list(counts.index), values=list(counts.values),
        marker_colors=[colors_map.get(u, "#64748b") for u in counts.index],
        hole=0.55,
        textfont_size=13,
        hovertemplate="%{label}: %{value} engines<extra></extra>",
    ))
    fig2.update_layout(
        height=300, margin=dict(t=10, b=10, l=10, r=10),
        plot_bgcolor="#151823", paper_bgcolor="#1e2130",
        font_color="#94a3b8",
        showlegend=True,
        legend=dict(font_color="#e2e8f0"),
        annotations=[dict(text=f"{len(fleet_df)}<br>engines",
                          font_size=16, font_color="#e2e8f0", showarrow=False)],
    )
    st.plotly_chart(fig2, use_container_width=True)


# ── Sensor history ────────────────────────────────────────────────────────────

full_test = load_test_full()
if full_test is not None:
    st.markdown(f"#### Engine #{selected_unit:03d} — Sensor History")
    hist = full_test[full_test["unit_id"] == selected_unit].copy()

    watch = [s for s in ["sensor_2", "sensor_3", "sensor_4", "sensor_7",
                          "sensor_11", "sensor_12", "sensor_15", "sensor_17",
                          "sensor_20", "sensor_21"] if s in hist.columns][:5]

    fig3 = go.Figure()
    palette = ["#60a5fa", "#34d399", "#f59e0b", "#a78bfa", "#f87171"]
    for i, s in enumerate(watch):
        fig3.add_trace(go.Scatter(
            x=hist["cycle"], y=hist[s], mode="lines", name=s,
            line=dict(width=1.8, color=palette[i % len(palette)]),
            hovertemplate=f"{s}: %{{y:.4f}} @ cycle %{{x}}<extra></extra>",
        ))
    fig3.update_layout(
        height=280, margin=dict(t=10, b=40, l=10, r=10),
        plot_bgcolor="#151823", paper_bgcolor="#1e2130",
        font_color="#94a3b8",
        xaxis=dict(title="Cycle", gridcolor="#2d3148"),
        yaxis=dict(title="Normalized value", gridcolor="#2d3148"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    font_color="#e2e8f0", bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ── Fleet table ───────────────────────────────────────────────────────────────

st.markdown("#### Fleet Status")

display = fleet_df[["unit_id", "predicted_rul", "true_rul", "cycle", "status"]].copy()
display.columns = ["Unit", "Predicted RUL", "True RUL", "Last Cycle", "Status"]
display = display.sort_values("Predicted RUL")

def _row_style(row):
    if row["Status"] == "CRITICAL": return ["background-color:#2d1515; color:#fca5a5"] * len(row)
    if row["Status"] == "WARNING":  return ["background-color:#2d2008; color:#fcd34d"] * len(row)
    return ["background-color:#0d2318; color:#6ee7b7"] * len(row)

st.dataframe(
    display.style.apply(_row_style, axis=1),
    use_container_width=True,
    height=260,
)

st.divider()


# ── Multi-agent analysis ──────────────────────────────────────────────────────

if run_agents:
    st.markdown("### Agent Analysis Pipeline")
    st.caption("LangGraph orchestrating 3 specialized agents via Groq llama-3.3-70b-versatile")

    prog = st.progress(0, text="Starting pipeline...")

    try:
        from agents.orchestrator import analyze_engine

        sensor_readings = {
            c: float(row[c]) for c in feature_cols
            if c.startswith("sensor_") and not c.endswith(("_roll_mean", "_roll_std"))
        }
        fv = [float(row[c]) for c in feature_cols]

        prog.progress(10, text="Running Sensor Monitor Agent...")
        result = analyze_engine(
            unit_id=selected_unit, cycle=cycle,
            sensor_readings=sensor_readings,
            feature_vector=fv, feature_names=feature_cols,
        )
        prog.progress(100, text="Complete!")

        if result.get("error"):
            st.error(f"Pipeline error: {result['error']}")
        else:
            a1, a2, a3 = st.columns(3)

            sa = result.get("sensor_analysis") or {}
            ra = result.get("rul_analysis") or {}
            mp = result.get("maintenance_plan") or {}

            with a1:
                st.markdown('<div class="agent-header agent-header-blue">🔍 Sensor Monitor</div>',
                            unsafe_allow_html=True)
                flagged = sa.get("flagged_sensors", [])
                st.markdown(f"Anomalies: **{sa.get('anomaly_count', 0)}** "
                            f"{'— ' + ', '.join(flagged) if flagged else '(none)'}")
                st.markdown(f'<div class="agent-output">{sa.get("llm_response", "")}</div>',
                            unsafe_allow_html=True)

            with a2:
                st.markdown('<div class="agent-header agent-header-purple">📊 RUL Predictor</div>',
                            unsafe_allow_html=True)
                rul_val = ra.get("predicted_rul")
                if rul_val is not None:
                    st.metric("Predicted RUL", f"{rul_val:.1f} cycles",
                              delta=ra.get("urgency", ""))
                st.markdown(f'<div class="agent-output">{ra.get("llm_response", "")}</div>',
                            unsafe_allow_html=True)

            with a3:
                st.markdown('<div class="agent-header agent-header-green">🔧 Maintenance Planner</div>',
                            unsafe_allow_html=True)
                tier = mp.get("tier", "")
                if tier:
                    st.info(f"**{tier}** · Score {mp.get('priority_score', 0):.0f}/100"
                            f" · Est. ${mp.get('estimated_cost_usd', 0):,}")
                st.markdown(f'<div class="agent-output">{mp.get("llm_response", "")}</div>',
                            unsafe_allow_html=True)

    except Exception as e:
        prog.empty()
        st.error(f"Pipeline failed: {e}")
        st.info("Check your GROQ_API_KEY in .env and that requirements are installed.")


# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption("NASA C-MAPSS FD001 · Saxena et al. PHM08 · XGBoost RUL Model · "
           "LangGraph Multi-Agent · Groq llama-3.3-70b-versatile · Arize Phoenix")
