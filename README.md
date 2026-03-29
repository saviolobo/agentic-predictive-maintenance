# Jet Engine Predictive Maintenance — Multi-Agent AI System

A production-quality multi-agent AI system for jet engine predictive maintenance using NASA's C-MAPSS turbofan engine degradation dataset.

## Architecture

```
                    ┌─────────────────────────────────┐
                    │       LangGraph Orchestrator      │
                    │                                   │
  Engine Data  ──►  │  [1] Sensor Monitor Agent         │
                    │         ↓                         │
                    │  [2] RUL Prediction Agent          │
                    │         ↓                         │
                    │  [3] Maintenance Planner Agent     │
                    └─────────────────────────────────┘
                              │
                         Streamlit UI
```

**Tech stack:** LangGraph · Groq (llama-3.3-70b-versatile) · Langfuse · XGBoost · Streamlit · NASA C-MAPSS FD001

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API keys
cp .env.example .env
# Edit .env: add GROQ_API_KEY (and optionally Langfuse keys for tracing)

# 3. One-shot setup (downloads data, builds pipeline, trains model)
python setup.py

# 4. Launch the UI
streamlit run ui/app.py
```

## Agents

| Agent | Responsibility | Tools |
|-------|---------------|-------|
| **Sensor Monitor** | Anomaly detection, degradation trend analysis | `detect_anomalies`, `analyze_degradation_trend` |
| **RUL Predictor** | Predicts remaining useful life, explains predictions | `predict_rul`, `get_feature_importance` |
| **Maintenance Planner** | Prioritized maintenance recommendations, cost-benefit | `calculate_maintenance_priority`, `generate_maintenance_schedule`, `estimate_maintenance_cost` |

## Dataset

[NASA C-MAPSS FD001](https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6):
- 100 turbofan engines
- Single operating condition
- HPC (High Pressure Compressor) degradation mode
- 21 sensor readings per cycle

## Model Performance (FD001)

XGBoost with rolling window features:
- **Test MAE:** ~15 cycles
- **Test RMSE:** ~20 cycles
- Features: 14 sensors × (raw + 5-cycle rolling mean + std) = 42 features

## Project Structure

```
├── agents/
│   ├── sensor_monitor.py      # Sensor anomaly detection agent
│   ├── rul_predictor.py       # RUL prediction agent
│   ├── maintenance_planner.py # Maintenance planning agent
│   └── orchestrator.py        # LangGraph pipeline
├── tools/
│   ├── data_pipeline.py       # C-MAPSS data preprocessing
│   ├── train_model.py         # XGBoost model training
│   └── download_data.py       # Dataset downloader
├── configs/
│   └── config.py              # Central configuration
├── ui/
│   └── app.py                 # Streamlit dashboard
├── data/                      # (gitignored)
├── models/                    # (gitignored)
├── setup.py                   # One-shot setup script
└── requirements.txt
```

## Observability

Two open-source options — both disabled by default:

**Langfuse** (cloud or self-hosted):
```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_TRACING=true
```

**Phoenix** (fully local, zero signup):
```bash
pip install arize-phoenix openinference-instrumentation-langchain
PHOENIX_TRACING=true  # in .env
python -m phoenix.server.main  # start the UI at localhost:6006
```
