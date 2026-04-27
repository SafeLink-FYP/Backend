"""
Flood Risk Prediction Service

Provides flood risk assessment based on user location and historical weather data.
"""

import os
import json
import pickle
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime, timedelta


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
        self.model = None
        self.scaler = None
        self.heatmap_data = {}
        self.is_loaded = False
        
        self._load_model()
    
    def _load_model(self):
        """Load the trained model and heatmap data"""
        try:
            # Try to load XGBoost model
            model_path = self.model_dir / "xgboost_geospatial_pipeline.pkl"
            if model_path.exists():
                with open(model_path, "rb") as f:
                    self.model = pickle.load(f)
                self.is_loaded = True
            
            # Load latest heatmap data (precomputed)
            data_dir = self.model_dir / "data"
            if data_dir.exists():
                # Load the most recent heatmap
                heatmap_files = sorted(
                    data_dir.glob("heatmap_payload_xgb_*.json"),
                    reverse=True
                )
                if heatmap_files:
                    with open(heatmap_files[0], "r") as f:
                        self.heatmap_data = json.load(f)
            
            print(f"✅ Flood model loaded from {self.model_dir}")
            
        except Exception as e:
            print(f"⚠️  Flood model not fully available: {e}")
            self.is_loaded = False
    
    def assess_risk(
        self,
        latitude: float,
        longitude: float,
        rainfall_mm: Optional[float] = None,
        river_discharge: Optional[float] = None,
        use_geo_blend: bool = True,
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
        
        except Exception as e:
            print(f"❌ Error assessing flood risk: {e}")
            return FloodRiskLevel(
                risk_level="UNKNOWN",
                risk_score=-1,
                affected_areas=[]
            )
    
    def _get_heatmap_risk(self, lat: float, lon: float) -> float:
        """Get risk score from precomputed heatmap (Pakistan-wide coverage)"""
        try:
            # Heatmap is grid-based, find closest grid point
            if "grid" in self.heatmap_data:
                grid = self.heatmap_data["grid"]
                # Find nearest grid point
                min_distance = float('inf')
                risk_value = 20.0
                
                for point in grid:
                    distance = ((lat - point["lat"]) ** 2 + (lon - point["lon"]) ** 2) ** 0.5
                    if distance < min_distance:
                        min_distance = distance
                        risk_value = point.get("risk_score", 20.0)
                
                return risk_value
            return 20.0
        except Exception as e:
            print(f"Heatmap lookup error: {e}")
            return 20.0
    
    def _compute_risk_from_features(self, rainfall: float, discharge: float) -> float:
        """Compute risk from rainfall and discharge features"""
        # Simple heuristic: higher rainfall and discharge = higher risk
        # Adjust thresholds based on Pakistan climate
        
        rainfall_risk = min(100, (rainfall / 100) * 100)  # 100mm = 100% from rainfall
        discharge_risk = min(100, (discharge / 5000) * 100)  # 5000 units = 100% from discharge
        
        # Combined risk (weighted average)
        combined_risk = (rainfall_risk * 0.6) + (discharge_risk * 0.4)
        
        return min(100, max(0, combined_risk))
    
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

        candidates = []
        for city, province, city_lat, city_lon in _CITIES:
            km = math.sqrt((lat - city_lat) ** 2 + (lon - city_lon) ** 2) * 111
            candidates.append((km, city, province))

        candidates.sort()

        results = []
        nearest_km, nearest_city, nearest_province = candidates[0]
        if nearest_km < 20:
            results.append(f"{nearest_city}, {nearest_province}")
        else:
            results.append(f"~{int(nearest_km)} km from {nearest_city}, {nearest_province}")

        # Add a second nearby city if within 60 km and different province
        for km, city, province in candidates[1:4]:
            if km < 60 and province != nearest_province:
                results.append(f"{city}, {province}")
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
        
        # If we have precomputed heatmap, use it
        if self.heatmap_data and "grid" in self.heatmap_data:
            return {
                "timestamp": datetime.now().isoformat(),
                "regions": self.heatmap_data.get("regions", {}),
                "grid": self.heatmap_data["grid"],
                "bounds": {
                    "north": LAT_MAX,
                    "south": LAT_MIN,
                    "east": LON_MAX,
                    "west": LON_MIN
                },
                "resolution_degrees": GRID_STEP,
                "data_source": "xgboost_model"
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
