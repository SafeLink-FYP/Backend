"""
Flood Risk Prediction Service

Provides flood risk assessment based on user location and historical weather data.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class FloodRiskLevel:
    """Flood risk assessment result"""
    
    def __init__(
        self,
        risk_level: str,  # "LOW", "MODERATE", "HIGH", "CRITICAL"
        risk_score: float,  # 0-100
        rainfall_mm: Optional[float] = None,
        river_discharge: Optional[float] = None,
        affected_areas: Optional[List[str]] = None
    ):
        self.risk_level = risk_level
        self.risk_score = risk_score
        self.rainfall_mm = rainfall_mm or 0
        self.river_discharge = river_discharge or 0
        self.affected_areas = affected_areas or []
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self):
        return {
            "risk_level": self.risk_level,
            "risk_score": self.risk_score,
            "rainfall_mm": self.rainfall_mm,
            "river_discharge": self.river_discharge,
            "affected_areas": self.affected_areas,
            "timestamp": self.timestamp,
        }


class FloodModel:
    """Wrapper around the XGBoost flood prediction model"""
    
    def __init__(self, model_dir: str = None):
        """
        Initialize the flood model
        
        Args:
            model_dir: Path to flood-api-data-fetcher directory
        """
        if model_dir is None:
            model_dir = os.path.join(
                os.path.dirname(__file__),
                "models/flood-api-data-fetcher"
            )
        
        self.model_dir = Path(model_dir)
        self.heatmap_data = {}
        self.is_loaded = False

        self._load_heatmap()

    def _load_heatmap(self):
        """Load the most recent precomputed XGBoost heatmap payload.

        The XGBoost model itself runs offline (see
        `models/flood-api-data-fetcher/xgboost_geospatial_pipeline.py`) and bakes
        its spatial predictions into `data/heatmap_payload_xgb_*.json`. Runtime
        risk = heuristic on live rainfall/discharge blended with this heatmap.
        """
        try:
            data_dir = self.model_dir / "data"
            if not data_dir.exists():
                logger.warning("Flood data dir missing: %s", data_dir)
                return

            heatmap_files = sorted(
                data_dir.glob("heatmap_payload_xgb_*.json"),
                reverse=True,
            )
            if not heatmap_files:
                logger.warning("No heatmap payload found in %s", data_dir)
                return

            with open(heatmap_files[0], "r") as f:
                self.heatmap_data = json.load(f)
            self.is_loaded = True
            logger.info("Flood heatmap loaded from %s", heatmap_files[0].name)

        except Exception:
            logger.exception("Failed to load flood heatmap")
            self.is_loaded = False
    
    def assess_risk(
        self,
        latitude: float,
        longitude: float,
        rainfall_mm: Optional[float] = None,
        river_discharge: Optional[float] = None,
        use_geo_blend: bool = False,
    ) -> FloodRiskLevel:
        """
        Assess flood risk at given location
        
        Args:
            latitude: User latitude
            longitude: User longitude
            rainfall_mm: Recent rainfall in mm (optional)
            river_discharge: River discharge level (optional)
        
        Returns:
            FloodRiskLevel object with assessment
        """
        
        try:
            # Real-time risk driven by actual rainfall + discharge data
            dynamic_risk = self._compute_risk_from_features(
                rainfall_mm or 0,
                river_discharge or 0
            )

            if use_geo_blend and self.heatmap_data:
                geo_risk = self._get_heatmap_risk(latitude, longitude)
                # Blend: real-time conditions (70%) + static geography (30%).
                risk_score = geo_risk * 0.30 + dynamic_risk * 0.70
            else:
                # Historical mode: use only weather-driven risk so different
                # years are compared purely on their actual weather data.
                risk_score = dynamic_risk

            risk_score = max(0.0, min(100.0, float(risk_score)))
            
            # Determine risk level from score
            if risk_score >= 80:
                level = "CRITICAL"
            elif risk_score >= 60:
                level = "HIGH"
            elif risk_score >= 40:
                level = "MODERATE"
            else:
                level = "LOW"
            
            # Find affected areas from geolocation
            affected = self._get_affected_areas(latitude, longitude)
            
            return FloodRiskLevel(
                risk_level=level,
                risk_score=risk_score,
                rainfall_mm=rainfall_mm,
                river_discharge=river_discharge,
                affected_areas=affected
            )
        
        except Exception:
            logger.exception("Error assessing flood risk")
            return FloodRiskLevel(
                risk_level="UNKNOWN",
                risk_score=-1,
                affected_areas=[],
            )
    
    def _get_heatmap_risk(self, lat: float, lon: float) -> float:
        """Get risk score (0–100) from the precomputed XGBoost heatmap.

        The heatmap payload from the training pipeline nests points under
        `grid.points` with `lng` / `weight` keys (weight is a 0–1 calibrated
        probability). Older fallback payloads may store `grid` as a flat list
        with `lon` / `risk_score`. Handle both.
        """
        try:
            points = self._heatmap_points()
            if not points:
                return 20.0

            min_distance_sq = float("inf")
            risk_value = 20.0
            for plat, plon, score in points:
                d_sq = (lat - plat) ** 2 + (lon - plon) ** 2
                if d_sq < min_distance_sq:
                    min_distance_sq = d_sq
                    risk_value = score
            return risk_value
        except Exception:
            logger.exception("Heatmap lookup error")
            return 20.0

    def _heatmap_points(self) -> List[tuple]:
        """Return cached normalised heatmap points as (lat, lon, score_0_100)."""
        if getattr(self, "_normalised_points", None) is not None:
            return self._normalised_points

        raw = self.heatmap_data.get("grid") if self.heatmap_data else None
        if isinstance(raw, dict):
            raw_points = raw.get("points", [])
        elif isinstance(raw, list):
            raw_points = raw
        else:
            raw_points = []

        out = []
        for p in raw_points:
            plat = p.get("lat")
            plon = p.get("lon", p.get("lng"))
            if plat is None or plon is None:
                continue
            # Training pipeline emits 0–1 probabilities under "weight";
            # fallback payloads use 0–100 under "risk_score".
            if "risk_score" in p:
                score = float(p["risk_score"])
            else:
                score = float(p.get("weight", 0.0)) * 100.0
            out.append((float(plat), float(plon), score))

        self._normalised_points = out
        return out
    
    # Calibrated against historical 7-day peak rainfall sums (~0–100 mm range).
    # Do NOT raise without re-running the historical backtest in
    # /flood/historical/model — that pipeline depends on this scale.
    _RAINFALL_SATURATION_MM = 100.0
    _DISCHARGE_SATURATION = 5000.0

    def _compute_risk_from_features(self, rainfall: float, discharge: float) -> float:
        """Compute risk (0–100) from recent rainfall and river discharge."""
        rainfall_risk = min(100.0, (rainfall / self._RAINFALL_SATURATION_MM) * 100.0)
        discharge_risk = min(100.0, (discharge / self._DISCHARGE_SATURATION) * 100.0)
        combined = rainfall_risk * 0.6 + discharge_risk * 0.4
        return min(100.0, max(0.0, combined))
    
    def _get_affected_areas(self, lat: float, lon: float) -> List[str]:
        """Return nearest city + district/province for the given coordinates."""
        import math

        # (city, district/province, lat, lon)
        _CITIES = [
            ("Karachi",       "Sindh",           24.86, 67.01),
            ("Hyderabad",     "Sindh",           25.39, 68.37),
            ("Sukkur",        "Sindh",           27.71, 68.86),
            ("Larkana",       "Sindh",           27.56, 68.22),
            ("Jacobabad",     "Sindh",           28.28, 68.44),
            ("Dadu",          "Sindh",           26.73, 67.78),
            ("Badin",         "Sindh",           24.65, 68.84),
            ("Thatta",        "Sindh",           24.75, 67.92),
            ("Nawabshah",     "Sindh",           26.24, 68.41),
            ("Lahore",        "Punjab",          31.55, 74.35),
            ("Rawalpindi",    "Punjab",          33.60, 73.04),
            ("Multan",        "Punjab",          30.19, 71.47),
            ("Faisalabad",    "Punjab",          31.42, 73.08),
            ("Gujranwala",    "Punjab",          32.16, 74.19),
            ("Sialkot",       "Punjab",          32.49, 74.53),
            ("DG Khan",       "Punjab",          30.06, 70.63),
            ("Rajanpur",      "Punjab",          29.10, 70.33),
            ("Muzaffargarh",  "Punjab",          30.07, 71.19),
            ("Mianwali",      "Punjab",          32.58, 71.53),
            ("Bahawalpur",    "Punjab",          29.39, 71.68),
            ("Sargodha",      "Punjab",          32.08, 72.67),
            ("Islamabad",     "ICT",             33.72, 73.06),
            ("Peshawar",      "KPK",             34.01, 71.57),
            ("Nowshera",      "KPK",             34.01, 71.98),
            ("Charsadda",     "KPK",             34.15, 71.73),
            ("Swat",          "KPK",             35.22, 72.42),
            ("Chitral",       "KPK",             35.85, 71.83),
            ("Abbottabad",    "KPK",             34.15, 73.22),
            ("Mansehra",      "KPK",             34.33, 73.20),
            ("Dir",           "KPK",             35.21, 71.88),
            ("Quetta",        "Balochistan",     30.18, 67.00),
            ("Khuzdar",       "Balochistan",     27.82, 66.61),
            ("Turbat",        "Balochistan",     26.00, 63.05),
            ("Gwadar",        "Balochistan",     25.12, 62.33),
            ("Naseerabad",    "Balochistan",     28.42, 67.92),
            ("Jafferabad",    "Balochistan",     28.34, 68.28),
            ("Zhob",          "Balochistan",     31.34, 69.45),
            ("Muzaffarabad",  "AJK",             34.37, 73.47),
            ("Mirpur",        "AJK",             33.14, 73.75),
            ("Gilgit",        "Gilgit-Baltistan",35.92, 74.31),
            ("Skardu",        "Gilgit-Baltistan",35.29, 75.63),
            ("Hunza",         "Gilgit-Baltistan",36.32, 74.65),
        ]

        # Approximate-distance ranking with cosine correction so longitudes near
        # 30°N don't over-count vs. latitudes.
        cos_lat = math.cos(math.radians(lat))
        candidates = []
        for city, province, city_lat, city_lon in _CITIES:
            dlat = lat - city_lat
            dlon = (lon - city_lon) * cos_lat
            km = math.sqrt(dlat * dlat + dlon * dlon) * 111
            candidates.append((km, city, province))

        candidates.sort()

        results: List[str] = []
        nearest_km, nearest_city, nearest_province = candidates[0]
        # Most major Pakistani cities are also district headquarters and share
        # the district name (Karachi, Lahore, Sukkur, Quetta, …). Surface that
        # explicitly so the UI shows "Karachi District, Sindh" rather than
        # an ambiguous "Karachi, Sindh".
        if nearest_km < 20:
            results.append(f"{nearest_city} District, {nearest_province}")
        elif nearest_km < 60:
            results.append(
                f"{nearest_city} District, {nearest_province} "
                f"(~{int(nearest_km)} km)"
            )
        else:
            results.append(
                f"{nearest_province} — near {nearest_city} District "
                f"(~{int(nearest_km)} km)"
            )

        # Add a second nearby district within 80 km from a different province
        # so cross-border risk gets surfaced (e.g. Sindh / Balochistan flooding).
        for km, city, province in candidates[1:5]:
            if km < 80 and province != nearest_province:
                results.append(f"{city} District, {province}")
                break

        return results
    
    def get_pakistan_heatmap(self) -> Dict:
        """
        Generate Pakistan-wide flood risk heatmap
        
        Returns grid of coordinates with risk scores for visualization
        Grid is 0.5° resolution covering Pakistan (23.5°N to 37.5°N, 61°E to 77.5°E)
        """
        
        heatmap_grid = []
        
        # Pakistan bounds
        LAT_MIN, LAT_MAX = 23.5, 37.5
        LON_MIN, LON_MAX = 61.0, 77.5
        GRID_STEP = 0.5  # 0.5 degree resolution
        
        # If we have precomputed heatmap, normalise it to the flat list schema
        # the Flutter app expects (lat / lon / risk_score / risk_level).
        if self.heatmap_data:
            points = self._heatmap_points()
            if points:
                grid_out = [
                    {
                        "lat": round(plat, 3),
                        "lon": round(plon, 3),
                        "risk_score": round(score, 2),
                        "risk_level": self._get_risk_level(score),
                    }
                    for plat, plon, score in points
                ]
                return {
                    "timestamp": datetime.now().isoformat(),
                    "grid": grid_out,
                    "bounds": {
                        "north": LAT_MAX,
                        "south": LAT_MIN,
                        "east": LON_MAX,
                        "west": LON_MIN,
                    },
                    "resolution_degrees": GRID_STEP,
                    "total_points": len(grid_out),
                    "data_source": "xgboost_model",
                }
        
        # Fallback: Generate grid from model predictions
        lat = LAT_MIN
        while lat <= LAT_MAX:
            lon = LON_MIN
            while lon <= LON_MAX:
                risk = self._compute_risk_from_features(
                    rainfall=self._estimate_rainfall(lat, lon),
                    discharge=self._estimate_discharge(lat, lon)
                )
                
                heatmap_grid.append({
                    "lat": round(lat, 2),
                    "lon": round(lon, 2),
                    "risk_score": round(risk, 2),
                    "risk_level": self._get_risk_level(risk)
                })
                
                lon += GRID_STEP
            lat += GRID_STEP
        
        return {
            "timestamp": datetime.now().isoformat(),
            "grid": heatmap_grid,
            "bounds": {
                "north": LAT_MAX,
                "south": LAT_MIN,
                "east": LON_MAX,
                "west": LON_MIN
            },
            "resolution_degrees": GRID_STEP,
            "total_points": len(heatmap_grid),
            "data_source": "computed_model"
        }
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Convert risk score to level"""
        if risk_score >= 80:
            return "CRITICAL"
        elif risk_score >= 60:
            return "HIGH"
        elif risk_score >= 40:
            return "MODERATE"
        else:
            return "LOW"
    
    def _estimate_rainfall(self, lat: float, lon: float) -> float:
        """Estimate rainfall at location based on geography"""
        # Higher rainfall in northern regions (Kashmir/Himalayas)
        # Lower in Balochistan
        
        if lat > 32:  # Northern Pakistan
            return 60 + (lat - 32) * 20
        elif 26 < lat <= 32:  # Central Pakistan (Punjab, KPK)
            return 40 + (32 - lat) * 5
        else:  # Southern Pakistan (Sindh, Lower Balochistan)
            return 30 - (26 - lat) * 2
    
    def _estimate_discharge(self, lat: float, lon: float) -> float:
        """Estimate river discharge at location"""
        # Indus River: primarily in western Pakistan
        # Sutlej and other tributaries: in northern Punjab
        
        if 68 < lon < 72 and 25 < lat < 35:  # Indus Valley
            return 2500 + (lat - 25) * 100
        elif lon < 68:  # Balochistan (lower discharge)
            return 800 - (68 - lon) * 50
        else:  # Eastern regions
            return 1200 + (lon - 72) * 100
