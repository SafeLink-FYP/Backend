# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Directory Overview

This directory holds the ML model training code and trained artifacts consumed by the SafeLink FastAPI backend (`../main.py`).

```
models/
├── Earthquake-Aftershocks/   # BiLSTM training code + trained model (used by production API)
├── Flood-Prediction/         # XGBoost standalone server + training pipeline
└── flood-api-data-fetcher/   # Active flood model data + heatmap payloads (used by production API)
```

## Which Files the Production API Loads

`../earthquake_service.py` resolves:
```
models/Earthquake-Aftershocks/trained_model/
  aftershock_lstm_model.h5
  scaler_input.pkl
  scaler_output.pkl
  metadata.pkl
```

`../flood_service.py` resolves:
```
models/flood-api-data-fetcher/
  xgboost_geospatial_pipeline.pkl       # root-level (not in data/models/)
  data/heatmap_payload_xgb_*.json
  data/model_metrics_xgboost.json
```

> `flood-api-data-fetcher/` is a live working copy of the flood pipeline. `Flood-Prediction/` is the standalone research server with a separate `app.py`.

## Retraining the Earthquake Model

```bash
cd Earthquake-Aftershocks
/opt/homebrew/bin/python3.11 train.py          # option 1: train; option 2: predict

# Evaluate accuracy on 129 held-out samples
/opt/homebrew/bin/python3.11 test_model_accuracy.py

# Validate against Pakistan historical catalog
/opt/homebrew/bin/python3.11 validate_against_history.py
# When prompted: query (2).csv
```

Trained artifacts are written to `trained_model/`. The model is deterministic (no randomness) — same input always yields identical output.

**Performance**: Rank-1 aftershock magnitude error ±0.244; location error ±8–15° (weak but acceptable given sparse training data).

## Retraining the Flood Model

`flood-api-data-fetcher/` uses a virtual environment at `.venv/`. Full pipeline:

```bash
cd flood-api-data-fetcher

# 1. Collect historical weather + discharge data (resumes via tracker)
python scripts/pakistan_rainfall_data.py

# 2. Prepare training dataset and label floods
python prepare_model_and_heatmap_data.py --date YYYY-MM-DD

# 3. Train XGBoost and generate heatmap payloads
python xgboost_geospatial_pipeline.py --date YYYY-MM-DD

# 4. Generate a live forecast payload
python future_heatmap_forecast.py --days 2 --city "Karachi"

# 5. Backtest against known flood years (2022–2026)
python scripts/backtest_flood_periods_2022_2026.py
```

Outputs land in `data/`:
- `models/xgb_flood_model.json` — trained XGBoost
- `models/xgb_flood_calibrator.pkl` — isotonic calibrator
- `models/xgb_feature_columns.json` — feature ordering for inference
- `heatmap_payload_xgb_<date>.json` — precomputed heatmap served by the main API

After retraining, copy `xgboost_geospatial_pipeline.pkl` (if regenerated) to the root of `flood-api-data-fetcher/` so the production API can find it.

## Standalone Flood Server (Flood-Prediction/)

A separate FastAPI + Google Maps front-end for development and visualization. Not used by the mobile app. See `Flood-Prediction/CLAUDE.md` for its full API and setup.

```bash
cd Flood-Prediction
bash run_map_server.sh    # requires GOOGLE_MAPS_API_KEY in .env
```

## Open-Meteo API Quota

The data collection scripts consume the Open-Meteo free tier (10,000 calls/day). Usage is tracked in `flood-api-data-fetcher/data/api_call_tracker.json`. Check this before running bulk collection.
