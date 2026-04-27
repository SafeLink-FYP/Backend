# SafeLink ML — Project Task Tracker

Use this file to track features, bugs, and ideas across sessions.
Status: `[ ]` pending · `[x]` done · `[-]` in progress · `[~]` blocked

---

## UI — Initial Build
- [x] Create `ui/index.html` — two-tab dark dashboard skeleton
- [x] Create `ui/style.css` — dark slate theme, Leaflet overrides
- [x] Create `ui/app.js` — full earthquake + flood logic
- [x] Mount `ui/` in FastAPI via `StaticFiles` at `/ui`
- [x] Update `CLAUDE.md` with UI section

---

## Earthquake Tab
- [x] World Leaflet map with CARTO dark tiles
- [x] Click-to-set search origin with 200 km radius ring
- [x] "My Location" geolocation button
- [x] Manual lat/lng coordinate inputs
- [x] `POST /earthquake/check` — live USGS + BiLSTM aftershock model
- [x] Mainshock markers scaled & colored by magnitude (M3–M7+)
- [x] Aftershock markers (orange) drawn when card is expanded
- [x] Sidebar cards sorted by magnitude, synced with map clicks
- [ ] Add magnitude filter slider (min M value)
- [ ] Add "hours back" selector (1h / 6h / 24h / 72h)
- [ ] Add USGS event link button in card footer
- [ ] Show shake-intensity heatmap from USGS ShakeMap when available

---

## Flood Tab
- [x] Pakistan-bounded Leaflet map with bounding box outline
- [x] Date picker 2014-01-01 → today (past archive) + up to 16 days ahead (forecast)
- [x] Click-to-set point with blue dot marker
- [x] `GET /flood/forecast?date=…&latitude=…&longitude=…`
- [x] Risk card: level badge, score bar, rainfall stat, nearest areas
- [x] `GET /flood/heatmap` — toggleable risk grid layer
- [x] `GET /flood/historical` — overlay circles for historical events (2010-2023)
- [ ] Add province/district boundary overlay (GeoJSON)
- [ ] Add river network overlay (Indus system)
- [ ] Animate heatmap over a date range (play button)
- [ ] Export current flood result as PDF / image

---

## Infrastructure / Backend
- [x] CORS enabled for all origins (Flutter app integration)
- [x] StaticFiles mount for dashboard at `http://localhost:8000/ui`
- [ ] Add `requirements.txt` entry for `python-multipart` (needed by StaticFiles in some setups)
- [ ] Docker-compose for backend + model volumes
- [ ] Add `/earthquake/global` endpoint (no origin required, global M5+ last 24h)
- [ ] Cache invalidation endpoint `DELETE /cache`

---

## Known Issues / Bugs
- [ ] Aftershock lat/lon from BiLSTM model can fall outside Pakistan when `pakistan_only=false` — expected; model predicts globally
- [ ] `risk_score` returns `-1` for `UNKNOWN` level when model is not loaded — handle gracefully in UI score bar

---

## Session Notes
<!-- Add dated notes here when resuming after a break -->
<!-- Example:
### 2026-04-20
- Built initial UI (index.html, style.css, app.js)
- Mounted at /ui via FastAPI StaticFiles
- Next: test with live server, add magnitude filter slider
-->

### 2026-04-20
- Initial UI scaffold complete: earthquake global view + Pakistan flood view
- Both tabs use Leaflet.js + CARTO dark tiles
- Dashboard served at `http://localhost:8000/ui`

### 2026-04-20 (Session 2 — Model Fixes)
**Earthquake model fixes:**
- `validate_against_history.py`: wrong 7-feature input → corrected to 4-feature [mag, depth, lat, lon]
- `earthquake_service.py`: BiLSTM output parsed as 4-column chunks (model outputs 3) → silent IndexError, always returned dummies. Full rewrite: now uses BiLSTM as cluster centre, Bath's Law + Gutenberg-Richter fill catalogue. Sub-M5 events guarded (scaler min=5.0).
- `process.py`: added depth > 0 and depth ≤ 700 filter (2 records had depth=0)

**Flood model fixes:**
- `xgboost_geospatial_pipeline.py`: duplicate `heavy_rain_days_7d` computation removed
- `prepare_model_and_heatmap_data.py`: 29 cities had discharge thresholds < 10 m³/s (noise-level); enforced 10 m³/s floor. Fixed `attach_labels()` to keep cities without threshold (default=0) so basin/precip propagation works. Added Mianwali→Attock and DI Khan→Attock basin anchors. Added Sibi to `_PRECIP_FLOOD_CITIES`.
- Retrained XGBoost: ROC-AUC 0.9848→0.9921, PR-AUC 0.7417→0.7635, F1 0.6906→0.7070
- Backtest improvements: 2022 recall 22.2%→44.4%, 2023 recall 0%→28.6%
