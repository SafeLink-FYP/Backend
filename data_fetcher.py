"""
External Data Fetching Service

Fetches real-time earthquake data from USGS and rainfall data from external sources.
"""

import httpx
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class EarthquakeEvent:
    """Representing a seismic event from USGS"""
    
    def __init__(
        self,
        event_id: str,
        magnitude: float,
        latitude: float,
        longitude: float,
        depth_km: float,
        timestamp: datetime,
        location_name: str = "",
        url: str = ""
    ):
        self.event_id = event_id
        self.magnitude = magnitude
        self.latitude = latitude
        self.longitude = longitude
        self.depth_km = depth_km
        self.timestamp = timestamp
        self.location_name = location_name
        self.url = url
    
    def to_dict(self):
        return {
            "event_id": self.event_id,
            "magnitude": self.magnitude,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "depth_km": self.depth_km,
            "timestamp": self.timestamp.isoformat(),
            "location_name": self.location_name,
            "url": self.url,
        }


class RainfallData:
    """Rainfall observation"""
    
    def __init__(
        self,
        location_name: str,
        latitude: float,
        longitude: float,
        rainfall_mm: float,
        timestamp: datetime,
        source: str = "unknown"
    ):
        self.location_name = location_name
        self.latitude = latitude
        self.longitude = longitude
        self.rainfall_mm = rainfall_mm
        self.timestamp = timestamp
        self.source = source
    
    def to_dict(self):
        return {
            "location_name": self.location_name,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "rainfall_mm": self.rainfall_mm,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
        }


class USGSFetcher:
    """Fetch earthquake data from USGS"""
    
    BASE_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    
    def __init__(self):
        self.session = None
    
    async def get_earthquakes_near(
        self,
        latitude: float,
        longitude: float,
        radius_km: float = 200,
        min_magnitude: float = 3.0,
        hours_back: int = 24
    ) -> List[EarthquakeEvent]:
        """
        Fetch earthquakes near a location
        
        Args:
            latitude: User latitude
            longitude: User longitude
            radius_km: Search radius in kilometers
            min_magnitude: Minimum magnitude to return
            hours_back: How many hours back to search
        
        Returns:
            List of EarthquakeEvent objects
        """
        
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                # Calculate time range
                endtime = datetime.utcnow()
                starttime = endtime - timedelta(hours=hours_back)
                
                params = {
                    "format": "geojson",
                    "starttime": starttime.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "endtime": endtime.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "latitude": latitude,
                    "longitude": longitude,
                    "maxradiuskm": radius_km,
                    "minmagnitude": min_magnitude,
                    "orderby": "time",
                    "limit": 100,
                }
                
                response = await client.get(self.BASE_URL, params=params)
                response.raise_for_status()
                
                data = response.json()
                earthquakes = []
                
                for feature in data.get("features", []):
                    props = feature.get("properties", {})
                    coords = feature.get("geometry", {}).get("coordinates", [])
                    
                    if len(coords) >= 3:
                        eq = EarthquakeEvent(
                            event_id=feature.get("id", "unknown"),
                            magnitude=props.get("mag", 0),
                            latitude=coords[1],
                            longitude=coords[0],
                            depth_km=coords[2],
                            timestamp=datetime.fromtimestamp(props.get("time", 0) / 1000),
                            location_name=props.get("place", ""),
                            url=props.get("url", "")
                        )
                        
                        # Filter by distance (rough approximation)
                        distance = self._calculate_distance(
                            latitude, longitude,
                            eq.latitude, eq.longitude
                        )
                        
                        if distance <= radius_km:
                            earthquakes.append(eq)
                
                logger.info(f"✅ Fetched {len(earthquakes)} earthquakes near ({latitude}, {longitude})")
                return earthquakes
        
        except Exception as e:
            logger.error(f"❌ Error fetching earthquakes: {e}")
            return []
    
    @staticmethod
    def _calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate approximate distance between two points in km
        Using simplified formula (not Haversine, but good enough for alerts)
        """
        import math
        
        R = 6371  # Earth radius in km
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c


class RainfallFetcher:
    """Fetch rainfall and river discharge data from Open-Meteo APIs for Pakistan"""
    
    WEATHER_API_URL = "https://archive-api.open-meteo.com/v1/archive"
    FLOOD_API_URL = "https://flood-api.open-meteo.com/v1/flood"
    
    def __init__(self):
        pass
    
    async def get_rainfall_near(
        self,
        latitude: float,
        longitude: float,
        radius_km: float = 100,
        days_back: int = 7
    ) -> dict:
        """
        Fetch real-time rainfall and discharge data from Open-Meteo for Pakistan
        Returns: {"rainfall_mm": float, "discharge": float, "data_source": str}
        """
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                # Get last 7 days of data
                end_date = datetime.utcnow().date()
                start_date = end_date - timedelta(days=days_back)
                
                # 1. Fetch weather data (precipitation)
                weather_params = {
                    "latitude": latitude,
                    "longitude": longitude,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "daily": "precipitation_sum,rain_sum",
                    "timezone": "Asia/Karachi",
                    "precipitation_unit": "mm",
                }
                
                weather_response = await client.get(
                    self.WEATHER_API_URL,
                    params=weather_params
                )
                weather_data = weather_response.json()
                
                # 2. Fetch river discharge (flood data)
                flood_params = {
                    "latitude": latitude,
                    "longitude": longitude,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "daily": "river_discharge",
                    "models": "seamless_v4",
                }
                
                flood_response = await client.get(
                    self.FLOOD_API_URL,
                    params=flood_params
                )
                flood_data = flood_response.json()
                
                # Calculate rainfall: sum of last 3 days
                rainfall_mm = 0.0
                if "daily" in weather_data and "precipitation_sum" in weather_data["daily"]:
                    precip_list = weather_data["daily"]["precipitation_sum"]
                    # Sum last 3 days of precipitation
                    rainfall_mm = sum(precip_list[-3:]) if len(precip_list) >= 3 else sum(precip_list)
                
                # Get latest river discharge
                discharge = 0.0
                if "daily" in flood_data and "river_discharge" in flood_data["daily"]:
                    discharge_list = flood_data["daily"]["river_discharge"]
                    # Get the latest non-null discharge value
                    discharge = next((d for d in reversed(discharge_list) if d is not None), 0.0)
                
                logger.info(f"✅ Fetched real-time data: rainfall={rainfall_mm}mm, discharge={discharge} m³/s")
                
                return {
                    "rainfall_mm": float(rainfall_mm),
                    "river_discharge": float(discharge),
                    "data_source": "Open-Meteo",
                    "timestamp": datetime.utcnow().isoformat()
                }
        
        except Exception:
            logger.exception("Error fetching rainfall data from Open-Meteo")
            return {
                "rainfall_mm": 0.0,
                "river_discharge": 0.0,
                "data_source": "error",
                "timestamp": datetime.utcnow().isoformat(),
            }



    async def get_historical_rainfall(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
    ) -> dict:
        """
        Fetch the peak 7-day rolling rainfall sum for a historical date range.

        Using the peak 7-day window (not the season total) keeps the value in the
        same range the live model is calibrated for (~0–100 mm).  A 3-month
        cumulative sum would saturate the risk formula at 100 % for every monsoon
        grid point, making every year look identical.
        """
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                params = {
                    "latitude": latitude,
                    "longitude": longitude,
                    "start_date": start_date,
                    "end_date": end_date,
                    "daily": "precipitation_sum",
                    "timezone": "Asia/Karachi",
                }
                resp = await client.get(self.WEATHER_API_URL, params=params)
                data = resp.json()

            rainfall_mm = 0.0
            if "daily" in data and "precipitation_sum" in data["daily"]:
                vals = [v if v is not None else 0.0
                        for v in data["daily"]["precipitation_sum"]]
                if len(vals) >= 7:
                    # Peak 7-day rolling sum
                    rainfall_mm = float(max(
                        sum(vals[i:i + 7]) for i in range(len(vals) - 6)
                    ))
                else:
                    rainfall_mm = float(sum(vals))

            return {"rainfall_mm": rainfall_mm, "river_discharge": 0.0}
        except Exception as e:
            logger.error(f"Historical rainfall fetch error ({latitude},{longitude}): {e}")
            return {"rainfall_mm": 0.0, "river_discharge": 0.0}


class DataCache:
    """Simple cache for recently fetched data"""
    
    def __init__(self, ttl_seconds: int = 600):
        self.data = {}
        self.ttl = ttl_seconds
    
    def get(self, key: str) -> Optional[any]:
        if key in self.data:
            value, timestamp = self.data[key]
            if (datetime.now() - timestamp).total_seconds() < self.ttl:
                return value
            else:
                del self.data[key]
        return None
    
    def set(self, key: str, value: any):
        self.data[key] = (value, datetime.now())
    
    def clear(self):
        self.data.clear()
