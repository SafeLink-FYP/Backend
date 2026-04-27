"""
SafeLink ML Integration API

Main FastAPI application that integrates earthquake and flood prediction models
with real-time data fetching.

Run: uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import logging
import httpx
from datetime import datetime, timedelta, date

from earthquake_service import EarthquakeModel, AftershockPrediction
from flood_service import FloodModel
from data_fetcher import USGSFetcher, RainfallFetcher, DataCache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SafeLink ML API",
    description="Real-time earthquake and flood risk prediction API",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

earthquake_model = EarthquakeModel()
flood_model = FloodModel()
usgs_fetcher = USGSFetcher()
rainfall_fetcher = RainfallFetcher()
data_cache = DataCache(ttl_seconds=300)
historical_cache = DataCache(ttl_seconds=86400)  # historical model results cached for 24 h

# Peak flood periods for each supported year
_FLOOD_YEAR_PERIODS: dict[int, tuple[str, str]] = {
    2010: ("2010-07-28", "2010-09-20"),
    2011: ("2011-08-01", "2011-09-15"),
    2014: ("2014-09-01", "2014-09-30"),
    2022: ("2022-06-15", "2022-09-15"),
    2023: ("2023-07-01", "2023-09-30"),
}


def _pakistan_grid() -> list[tuple[float, float]]:
    """1° resolution fetch grid (238 points). Response is upsampled to 0.5° by interpolation."""
    return [
        (float(lat), float(lon))
        for lat in range(24, 38)
        for lon in range(61, 78)
    ]


def _upsample_grid(coarse: list[dict]) -> list[dict]:
    """
    Upsample 1° results to 0.5° by bilinear-style interpolation.
    For each midpoint between coarse nodes, average the four surrounding corners.
    """
    import math

    score_map: dict[tuple, float] = {(p["lat"], p["lon"]): p["risk_score"] for p in coarse}
    level_map: dict[tuple, str]  = {(p["lat"], p["lon"]): p["risk_level"] for p in coarse}
    rain_map:  dict[tuple, float] = {(p["lat"], p["lon"]): p["rainfall_mm"] for p in coarse}

    def _risk_level(score: float) -> str:
        if score >= 80: return "CRITICAL"
        if score >= 60: return "HIGH"
        if score >= 40: return "MODERATE"
        return "LOW"

    output: list[dict] = list(coarse)  # keep original nodes

    # Add midpoints: (lat+0.5, lon), (lat, lon+0.5), (lat+0.5, lon+0.5)
    lats = sorted({p["lat"] for p in coarse})
    lons = sorted({p["lon"] for p in coarse})

    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            neighbours = [score_map.get((lat, lon))]
            if i + 1 < len(lats): neighbours.append(score_map.get((lats[i+1], lon)))
            if j + 1 < len(lons): neighbours.append(score_map.get((lat, lons[j+1])))
            if i + 1 < len(lats) and j + 1 < len(lons):
                neighbours.append(score_map.get((lats[i+1], lons[j+1])))

            valid = [v for v in neighbours if v is not None]
            if not valid:
                continue
            avg = sum(valid) / len(valid)
            avg_rain = sum(
                rain_map.get((ll, lo), 0)
                for ll, lo in [
                    (lat, lon),
                    (lats[i+1] if i+1 < len(lats) else lat, lon),
                    (lat, lons[j+1] if j+1 < len(lons) else lon),
                    (lats[i+1] if i+1 < len(lats) else lat, lons[j+1] if j+1 < len(lons) else lon),
                ]
            ) / 4

            half_lat = round(lat + 0.5, 1)
            half_lon = round(lon + 0.5, 1)

            if j + 1 < len(lons):
                output.append({"lat": lat, "lon": round(lon + 0.5, 1),
                                "risk_score": round(avg, 2), "risk_level": _risk_level(avg),
                                "rainfall_mm": round(avg_rain, 1)})
            if i + 1 < len(lats):
                output.append({"lat": round(lat + 0.5, 1), "lon": lon,
                                "risk_score": round(avg, 2), "risk_level": _risk_level(avg),
                                "rainfall_mm": round(avg_rain, 1)})
            if i + 1 < len(lats) and j + 1 < len(lons):
                output.append({"lat": half_lat, "lon": half_lon,
                                "risk_score": round(avg, 2), "risk_level": _risk_level(avg),
                                "rainfall_mm": round(avg_rain, 1)})

    return output

# Pakistan geographic bounds
PK_LAT_MIN, PK_LAT_MAX = 23.5, 37.5
PK_LNG_MIN, PK_LNG_MAX = 60.0, 78.5


# ==================== Request/Response Models ====================

class LocationRequest(BaseModel):
    latitude: float
    longitude: float
    user_id: Optional[str] = None
    pakistan_only: bool = False   # Filter earthquake results to Pakistan bounding box


class AftershockResponse(BaseModel):
    rank: int
    magnitude: float
    latitude: float
    longitude: float
    depth_km: float
    confidence: float             # 0.0–1.0 — multiply by 100 for % in UI


class EarthquakeAlertResponse(BaseModel):
    mainshock_event_id: str
    mainshock_magnitude: float
    mainshock_latitude: float
    mainshock_longitude: float
    mainshock_depth_km: float
    mainshock_timestamp: str
    mainshock_location: str = ""  # Human-readable place name from USGS
    predicted_aftershocks: List[AftershockResponse]
    distance_to_user_km: float
    should_alert: bool
    message: str


class FloodAlertResponse(BaseModel):
    risk_level: str               # LOW | MODERATE | HIGH | CRITICAL
    risk_score: float             # 0–100 (show as %)
    rainfall_mm: float
    affected_areas: List[str]
    should_alert: bool
    data_date: Optional[str] = None   # ISO date string — which date this is for


class HistoricalFloodRegion(BaseModel):
    lat: float
    lng: float
    radius_km: float
    district: str


class HistoricalFloodEvent(BaseModel):
    year: int
    label: str
    description: str
    deaths: int
    affected_millions: float
    regions: List[HistoricalFloodRegion]


class HealthResponse(BaseModel):
    status: str
    models_loaded: dict
    timestamp: str


# ==================== Health ====================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return {
        "status": "ok",
        "models_loaded": {
            "earthquake": earthquake_model.is_loaded,
            "flood": flood_model.is_loaded,
        },
        "timestamp": datetime.now().isoformat()
    }


# ==================== Earthquake ====================

@app.post("/earthquake/check", response_model=List[EarthquakeAlertResponse])
async def check_earthquakes(location: LocationRequest):
    """
    Check nearby earthquakes and predict aftershocks.
    Set pakistan_only=true to restrict results to the Pakistan bounding box
    (useful for Pakistan-focused monitoring — also lowers min-magnitude to 2.5).
    """
    try:
        cache_key = f"earthquakes_{location.latitude}_{location.longitude}_{location.pakistan_only}"
        cached = data_cache.get(cache_key)
        if cached:
            return cached

        min_mag = 2.5 if location.pakistan_only else 4.5
        radius  = 500 if location.pakistan_only else 5000

        earthquakes = await usgs_fetcher.get_earthquakes_near(
            latitude=location.latitude,
            longitude=location.longitude,
            radius_km=radius,
            min_magnitude=min_mag,
            hours_back=24
        )

        # Pakistan-only filter
        if location.pakistan_only:
            earthquakes = [
                eq for eq in earthquakes
                if PK_LAT_MIN <= eq.latitude <= PK_LAT_MAX
                and PK_LNG_MIN <= eq.longitude <= PK_LNG_MAX
            ]

        alerts = []
        ALERT_RADIUS_KM = 50

        for eq in earthquakes:
            aftershocks = earthquake_model.predict_aftershocks(
                mainshock_magnitude=eq.magnitude,
                mainshock_depth=eq.depth_km,
                mainshock_latitude=eq.latitude,
                mainshock_longitude=eq.longitude,
                top_k=5
            )

            distance = usgs_fetcher._calculate_distance(
                location.latitude, location.longitude,
                eq.latitude, eq.longitude
            )

            should_alert = distance <= ALERT_RADIUS_KM

            alert = EarthquakeAlertResponse(
                mainshock_event_id=eq.event_id,
                mainshock_magnitude=eq.magnitude,
                mainshock_latitude=eq.latitude,
                mainshock_longitude=eq.longitude,
                mainshock_depth_km=eq.depth_km,
                mainshock_timestamp=eq.timestamp.isoformat(),
                mainshock_location=eq.location_name,
                predicted_aftershocks=[
                    AftershockResponse(**as_.to_dict())
                    for as_ in aftershocks
                ],
                distance_to_user_km=round(distance, 2),
                should_alert=should_alert,
                message=(
                    f"M{eq.magnitude} earthquake near {eq.location_name} — "
                    f"{distance:.0f} km away. "
                    f"{len(aftershocks)} aftershock(s) predicted."
                )
            )
            alerts.append(alert)

        data_cache.set(cache_key, alerts)
        return alerts

    except Exception as e:
        logger.error(f"Error checking earthquakes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/earthquake/predict")
async def predict_aftershocks(
    magnitude: float = Query(..., ge=0, le=9),
    depth: float = Query(..., ge=0, le=700),
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
    top_k: int = Query(5, ge=1, le=10)
):
    """Predict aftershocks for a specific mainshock (manual input)."""
    try:
        aftershocks = earthquake_model.predict_aftershocks(
            mainshock_magnitude=magnitude,
            mainshock_depth=depth,
            mainshock_latitude=latitude,
            mainshock_longitude=longitude,
            top_k=top_k
        )
        return {
            "mainshock": {
                "magnitude": magnitude,
                "depth_km": depth,
                "latitude": latitude,
                "longitude": longitude
            },
            "predicted_aftershocks": [a.to_dict() for a in aftershocks],
            "count": len(aftershocks)
        }
    except Exception as e:
        logger.error(f"Error predicting aftershocks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Flood ====================

@app.post("/flood/check", response_model=FloodAlertResponse)
async def check_flood_risk(location: LocationRequest):
    """
    Check real-time flood risk at user location using Open-Meteo data (last 7 days).
    """
    try:
        cache_key = f"flood_{location.latitude}_{location.longitude}"
        cached = data_cache.get(cache_key)
        if cached:
            return cached

        weather_data = await rainfall_fetcher.get_rainfall_near(
            latitude=location.latitude,
            longitude=location.longitude,
            days_back=7
        )

        risk = flood_model.assess_risk(
            latitude=location.latitude,
            longitude=location.longitude,
            rainfall_mm=weather_data['rainfall_mm'],
            river_discharge=weather_data['river_discharge']
        )

        alert = FloodAlertResponse(
            risk_level=risk.risk_level,
            risk_score=risk.risk_score,
            rainfall_mm=weather_data['rainfall_mm'],
            affected_areas=risk.affected_areas,
            should_alert=risk.risk_level in ["HIGH", "CRITICAL"],
            data_date=datetime.utcnow().date().isoformat()
        )

        data_cache.set(cache_key, alert)
        return alert

    except Exception as e:
        logger.error(f"Error checking flood risk: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/flood/forecast", response_model=FloodAlertResponse)
async def get_flood_forecast(
    forecast_date: str = Query(
        ...,
        alias="date",
        description="Date to forecast (YYYY-MM-DD). Past dates use archive data; future dates use forecast."
    ),
    latitude:  float = Query(30.3753, ge=23.0, le=38.0, description="Latitude (Pakistan: 23–38)"),
    longitude: float = Query(69.3451, ge=60.0, le=78.5, description="Longitude (Pakistan: 60–78.5)"),
):
    """
    Get flood risk for any specific date — historical archive or 7-day forecast.
    Uses Open-Meteo archive API for past dates and forecast API for future dates.
    """
    try:
        target = datetime.strptime(forecast_date, "%Y-%m-%d").date()
        today  = datetime.utcnow().date()

        rainfall_mm = 0.0

        async with httpx.AsyncClient(timeout=15.0) as client:
            if target <= today:
                # Historical: use archive API, 3-day window around target date
                window_start = target - timedelta(days=2)
                params = {
                    "latitude":   latitude,
                    "longitude":  longitude,
                    "start_date": window_start.isoformat(),
                    "end_date":   target.isoformat(),
                    "daily":      "precipitation_sum",
                    "timezone":   "Asia/Karachi",
                }
                resp = await client.get(
                    "https://archive-api.open-meteo.com/v1/archive",
                    params=params
                )
            else:
                # Forecast: use open-meteo forecast API
                # Clamp to max 7 days ahead
                delta = (target - today).days
                if delta > 16:
                    raise HTTPException(400, "Can forecast at most 16 days into the future.")
                params = {
                    "latitude":     latitude,
                    "longitude":    longitude,
                    "daily":        "precipitation_sum",
                    "timezone":     "Asia/Karachi",
                    "forecast_days": delta + 1,
                }
                resp = await client.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params=params
                )

            data = resp.json()

        if "daily" in data and "precipitation_sum" in data["daily"]:
            vals = [v for v in data["daily"]["precipitation_sum"] if v is not None]
            rainfall_mm = float(sum(vals))

        risk = flood_model.assess_risk(
            latitude=latitude,
            longitude=longitude,
            rainfall_mm=rainfall_mm,
            river_discharge=None
        )

        return FloodAlertResponse(
            risk_level=risk.risk_level,
            risk_score=risk.risk_score,
            rainfall_mm=rainfall_mm,
            affected_areas=risk.affected_areas,
            should_alert=risk.risk_level in ["HIGH", "CRITICAL"],
            data_date=target.isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting flood forecast: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/flood/heatmap")
async def get_flood_heatmap():
    """Pakistan-wide flood risk heatmap grid (for map visualisation)."""
    try:
        return flood_model.get_pakistan_heatmap()
    except Exception as e:
        logger.error(f"Error fetching flood heatmap: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/flood/historical", response_model=List[HistoricalFloodEvent])
async def get_historical_floods():
    """Major historical Pakistan flood events with affected districts."""
    return [
        HistoricalFloodEvent(
            year=2022, label="2022 Super Floods",
            description="One-third of Pakistan submerged. Heaviest monsoon in 30 years.",
            deaths=1739, affected_millions=33.0,
            regions=[
                HistoricalFloodRegion(lat=26.5, lng=68.2, radius_km=160, district="Sindh — Dadu, Jacobabad, Larkana"),
                HistoricalFloodRegion(lat=29.8, lng=70.9, radius_km=130, district="South Punjab — DG Khan, Rajanpur"),
                HistoricalFloodRegion(lat=29.0, lng=66.2, radius_km=110, district="Balochistan — Jafferabad, Naseerabad"),
                HistoricalFloodRegion(lat=34.1, lng=72.1, radius_km=80,  district="KPK — Swat, Charsadda"),
            ]
        ),
        HistoricalFloodEvent(
            year=2010, label="2010 Mega Floods",
            description="Worst floods in Pakistan's recorded history. 20 million displaced.",
            deaths=2000, affected_millions=20.0,
            regions=[
                HistoricalFloodRegion(lat=31.4, lng=71.3, radius_km=200, district="Punjab — Muzaffargarh, Mianwali, Jhang"),
                HistoricalFloodRegion(lat=34.3, lng=72.0, radius_km=150, district="KPK — Nowshera, Charsadda, Peshawar"),
                HistoricalFloodRegion(lat=25.8, lng=69.0, radius_km=170, district="Sindh — Sukkur, Hyderabad, Thatta"),
                HistoricalFloodRegion(lat=27.5, lng=68.5, radius_km=90,  district="Sindh — Kashmore, Ghotki"),
            ]
        ),
        HistoricalFloodEvent(
            year=2014, label="2014 Floods",
            description="Flash floods and river flooding across Punjab and AJK.",
            deaths=367, affected_millions=2.5,
            regions=[
                HistoricalFloodRegion(lat=32.5, lng=74.5, radius_km=90,  district="Punjab — Sialkot, Gujranwala, Narowal"),
                HistoricalFloodRegion(lat=33.6, lng=73.5, radius_km=70,  district="AJK — Mirpur, Bhimber"),
                HistoricalFloodRegion(lat=34.4, lng=73.5, radius_km=60,  district="KPK — Mansehra, Abbottabad"),
            ]
        ),
        HistoricalFloodEvent(
            year=2011, label="2011 Floods",
            description="Second consecutive year of widespread Sindh flooding.",
            deaths=520, affected_millions=9.6,
            regions=[
                HistoricalFloodRegion(lat=26.0, lng=68.8, radius_km=140, district="Sindh — Badin, Thatta, Mirpurkhas"),
                HistoricalFloodRegion(lat=27.8, lng=68.9, radius_km=100, district="Sindh — Kashmore, Shikarpur"),
            ]
        ),
        HistoricalFloodEvent(
            year=2023, label="2023 Monsoon Floods",
            description="Balochistan and KPK severely hit by flash floods.",
            deaths=300, affected_millions=1.5,
            regions=[
                HistoricalFloodRegion(lat=28.5, lng=66.8, radius_km=100, district="Balochistan — Kalat, Khuzdar, Lasbela"),
                HistoricalFloodRegion(lat=35.0, lng=72.3, radius_km=80,  district="KPK — Upper Dir, Chitral"),
            ]
        ),
    ]


@app.get("/flood/historical/model")
async def get_historical_flood_model(year: int = Query(..., description="Flood year — one of 2010, 2011, 2014, 2022, 2023")):
    """
    Run the XGBoost flood model on real historical weather data for the given year.
    Fetches rainfall from Open-Meteo archive for 238 points (1° grid), then upsamples
    to 0.5° via interpolation (~924 points) for smooth Flutter circle rendering.
    Results are cached for 24 h (data never changes for past years).
    """
    if year not in _FLOOD_YEAR_PERIODS:
        raise HTTPException(400, f"Supported years: {sorted(_FLOOD_YEAR_PERIODS.keys())}")

    cache_key = f"hist_model_{year}"
    cached = historical_cache.get(cache_key)
    if cached:
        return cached

    start_date, end_date = _FLOOD_YEAR_PERIODS[year]
    grid = _pakistan_grid()

    import asyncio
    semaphore = asyncio.Semaphore(10)  # max 10 concurrent Open-Meteo calls

    async def assess_point(lat: float, lon: float) -> dict:
        async with semaphore:
            weather = await rainfall_fetcher.get_historical_rainfall(lat, lon, start_date, end_date)
            risk = flood_model.assess_risk(
                lat, lon,
                weather["rainfall_mm"],
                weather["river_discharge"],
                use_geo_blend=False,   # compare years purely on weather data
            )
            return {
                "lat": lat,
                "lon": lon,
                "risk_score": round(risk.risk_score, 2),
                "risk_level": risk.risk_level,
                "rainfall_mm": round(weather["rainfall_mm"], 1),
            }

    coarse_results = await asyncio.gather(*[assess_point(lat, lon) for lat, lon in grid])

    # Upsample 1° → 0.5° by interpolation so Flutter circles overlap and fill gaps
    upsampled = _upsample_grid(list(coarse_results))

    response = {
        "year": year,
        "period": f"{start_date} to {end_date}",
        "grid": upsampled,
        "total_assessed": len(upsampled),
    }
    historical_cache.set(cache_key, response)
    return response


# ==================== Combined ====================

@app.post("/alerts/check")
async def check_all_alerts(location: LocationRequest):
    """Check both earthquake and flood risks in a single call."""
    try:
        eq_alerts  = await check_earthquakes(location)
        flood_alert = await check_flood_risk(location)
        critical   = [a for a in eq_alerts if a.should_alert]
        return {
            "timestamp": datetime.now().isoformat(),
            "location": {"latitude": location.latitude, "longitude": location.longitude},
            "earthquake_alerts": critical,
            "flood_alert": flood_alert,
            "has_critical_alerts": len(critical) > 0 or flood_alert.should_alert,
            "alert_count": len(critical) + (1 if flood_alert.should_alert else 0),
        }
    except Exception as e:
        logger.error(f"Error checking all alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Root ====================

@app.get("/")
async def root():
    return {
        "name": "SafeLink ML API", "version": "1.1.0", "docs": "/docs",
        "endpoints": {
            "health":             "GET  /health",
            "earthquake_check":   "POST /earthquake/check  (pakistan_only=true for PK filter)",
            "earthquake_predict": "POST /earthquake/predict",
            "flood_check":        "POST /flood/check",
            "flood_forecast":     "GET  /flood/forecast?date=YYYY-MM-DD&latitude=X&longitude=Y",
            "flood_heatmap":      "GET  /flood/heatmap",
            "flood_historical":   "GET  /flood/historical",
            "all_alerts":         "POST /alerts/check",
        }
    }


app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
