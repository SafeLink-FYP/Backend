# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SafeLink ML API — a FastAPI backend providing real-time earthquake aftershock prediction and flood risk assessment for Pakistan, integrating ML models with live external data sources.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn main:app --reload --port 8000

# Run production server
uvicorn main:app --host 0.0.0.0 --port $PORT

# Interactive API docs (after server starts)
open http://localhost:8000/docs
```

There is no automated test suite — testing is done via Swagger UI (`/docs`) or curl against a running server.

## Environment Setup

Copy `.env.example` to `.env` and set:
- `EARTHQUAKE_MODEL_PATH` — path to `trained_model/` directory containing `.h5` and `.pkl` files
- `FLOOD_MODEL_PATH` — path to directory containing `xgboost_geospatial_pipeline.pkl` and `data/heatmap_payload_xgb_*.json`
- `GOOGLE_MAPS_API_KEY` — optional

Model files live outside this repo at `../lib/Models/`. Both services degrade gracefully if models are missing.

## Architecture

The app is four files with clear separation of concerns:

| File | Responsibility |
|------|---------------|
| `main.py` | FastAPI app, all 9 route handlers, Pydantic request/response models |
| `earthquake_service.py` | `EarthquakeModel` — loads BiLSTM (TensorFlow `.h5`), runs aftershock prediction |
| `flood_service.py` | `FloodModel` — loads XGBoost pipeline, computes risk level + score |
| `data_fetcher.py` | `USGSFetcher`, `RainfallFetcher`, `DataCache` — async HTTP calls to external APIs |

**Request flow:**
```
POST /alerts/check (lat, lon)
  → data_fetcher: fetch USGS earthquakes + Open-Meteo rainfall/discharge (5-min TTL cache)
  → earthquake_service: filter by radius, run BiLSTM for aftershock predictions
  → flood_service: build feature vector, run XGBoost → risk score 0–100 + level
  → combined JSON response
```

**External APIs:**
- USGS Earthquake: `https://earthquake.usgs.gov/fdsnws/event/1/query`
- Open-Meteo Archive: `https://archive-api.open-meteo.com/v1/archive` (historical rainfall)
- Open-Meteo Flood: `https://flood-api.open-meteo.com/v1/flood` (river discharge)
- Open-Meteo Forecast: `https://api.open-meteo.com/v1/forecast`

## Dashboard UI

A browser-based monitoring dashboard lives in `ui/` and is served by FastAPI at `http://localhost:8000/ui`.

| File | Purpose |
|------|---------|
| `ui/index.html` | Two-tab layout: Earthquakes (global) and Flood Risk (Pakistan) |
| `ui/style.css`  | Dark slate theme; Leaflet popup overrides |
| `ui/app.js`     | All map + API logic (vanilla JS, no build step) |

**Earthquake tab** — click anywhere on the world map (or type coordinates) to `POST /earthquake/check`; mainshock markers are sized/colored by magnitude; clicking a result card draws predicted aftershock markers.

**Flood tab** — Pakistan-bounded map; date picker 2014-01-01 → 16 days ahead calls `GET /flood/forecast`; heatmap toggle loads `GET /flood/heatmap`; History panel overlays circles for major flood events (2010–2023).

`StaticFiles` is mounted at the bottom of `main.py` — it must stay **after** all API routes to avoid shadowing them.

Task tracker for ongoing UI work: `TASKS.md`.

## Key Design Decisions

- **Geographic scope**: Pakistan bounding box (lat 23.5–37.5, lon 60.0–78.5) applied as optional filter; timezone hardcoded to `Asia/Karachi`.
- **Graceful degradation**: Both ML services catch import/load errors and fall back to dummy/rule-based predictions so the API never hard-crashes on missing models.
- **CORS**: All origins allowed — designed for a Flutter mobile app client.
- **Async throughout**: All external HTTP calls use `httpx` with `async/await`; model inference is synchronous (runs in the same thread).
- **Heatmap data**: Flood heatmap endpoint serves precomputed JSON files (`heatmap_payload_xgb_*.json`) rather than recomputing on every request.
