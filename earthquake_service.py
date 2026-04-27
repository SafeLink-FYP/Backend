"""
Earthquake Aftershock Prediction Service

Loads the trained BiLSTM model and provides predictions for aftershocks
given a mainshock event.

Model training range: M5.0–7.7 (Pakistan/Afghanistan region).
For M<5.0 events (outside training range) the BiLSTM is not invoked;
predictions are generated using pure physics laws only.

Output parsing fix (v2):
  The model outputs exactly 3 values [mag, lat, lon] — the single most
  likely aftershock. The old service incorrectly parsed 4-column chunks,
  causing a silent IndexError and always returning dummy values.
  Now: model gives the cluster centre, then Bath's Law + Gutenberg-Richter
  fill out the full top-k catalogue.
"""

import os
import pickle
import numpy as np
from pathlib import Path
from typing import List, Optional

try:
    from tensorflow import keras
except ImportError:
    keras = None


class AftershockPrediction:
    """Single predicted aftershock"""

    def __init__(
        self,
        rank: int,
        magnitude: float,
        latitude: float,
        longitude: float,
        depth_km: float,
        confidence: float = 0.85,
    ):
        self.rank = rank
        self.magnitude = magnitude
        self.latitude = latitude
        self.longitude = longitude
        self.depth_km = depth_km
        self.confidence = confidence

    def to_dict(self):
        return {
            "rank": self.rank,
            "magnitude": self.magnitude,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "depth_km": self.depth_km,
            "confidence": self.confidence,
        }


class EarthquakeModel:
    """Wrapper around the BiLSTM earthquake aftershock prediction model"""

    # Training range of the scaler (mag axis).  Used to guard against
    # extrapolation when the API queries sub-M5 events.
    SCALER_MAG_MIN = 5.0
    SCALER_MAG_MAX = 7.7

    def __init__(self, model_dir: str = None):
        if model_dir is None:
            model_dir = os.path.join(
                os.path.dirname(__file__),
                "models/Earthquake-Aftershocks/trained_model",
            )

        self.model_dir = Path(model_dir)
        self.model = None
        self.scaler_input = None
        self.scaler_output = None
        self.metadata = None
        self.is_loaded = False

        self._load_model()

    # ── Loading ──────────────────────────────────────────────────────────────

    def _load_model(self):
        try:
            model_path = self.model_dir / "aftershock_lstm_model.h5"
            if not model_path.exists():
                print(f"⚠️  Model file not found at {model_path}")
                return

            if keras is None:
                print(
                    "⚠️  TensorFlow/Keras not installed — earthquake model unavailable. "
                    "Physics-only predictions will be used."
                )
                return

            self.model = keras.models.load_model(str(model_path), compile=False)
            self.is_loaded = True

            for attr, fname in [
                ("scaler_input",  "scaler_input.pkl"),
                ("scaler_output", "scaler_output.pkl"),
                ("metadata",      "metadata.pkl"),
            ]:
                p = self.model_dir / fname
                if p.exists():
                    with open(p, "rb") as f:
                        setattr(self, attr, pickle.load(f))

            # Update scaler bounds from the actual saved scaler
            if self.scaler_input is not None:
                self.SCALER_MAG_MIN = float(self.scaler_input.data_min_[0])
                self.SCALER_MAG_MAX = float(self.scaler_input.data_max_[0])

            print(f"✅ Earthquake model loaded — training range M{self.SCALER_MAG_MIN}–M{self.SCALER_MAG_MAX}")

        except Exception as e:
            print(f"❌ Error loading earthquake model: {e}")
            self.is_loaded = False

    # ── Public API ───────────────────────────────────────────────────────────

    def predict_aftershocks(
        self,
        mainshock_magnitude: float,
        mainshock_depth: float,
        mainshock_latitude: float,
        mainshock_longitude: float,
        top_k: int = 5,
    ) -> List[AftershockPrediction]:
        """
        Predict top-k aftershocks for a given mainshock.

        Pipeline
        --------
        1.  BiLSTM  →  base prediction [mag, lat, lon] of the primary aftershock
            (only for M≥5.0 which is within the model's training range).
        2.  Bath's Law  →  cap the largest expected aftershock magnitude.
        3.  Gutenberg-Richter  →  sample a realistic magnitude distribution.
        4.  Wells-Coppersmith  →  estimate the aftershock zone radius.
        5.  Cluster placement  →  scatter aftershocks around the BiLSTM-predicted
            centre (or the mainshock epicentre for sub-M5 events).
        6.  Sort by magnitude, assign confidence scores.
        """
        use_bilstm = (
            self.is_loaded
            and self.model is not None
            and mainshock_magnitude >= self.SCALER_MAG_MIN
        )

        # ── Step 1: BiLSTM base prediction ───────────────────────────────────
        cluster_lat = mainshock_latitude
        cluster_lon = mainshock_longitude
        bilstm_mag  = mainshock_magnitude - 1.2   # Bath's Law fallback

        if use_bilstm:
            try:
                input_data = np.array([[
                    mainshock_magnitude,
                    mainshock_depth,
                    mainshock_latitude,
                    mainshock_longitude,
                ]], dtype=np.float64)

                # Clip to training range so MinMaxScaler stays in [0, 1]
                input_data[0, 0] = np.clip(
                    input_data[0, 0], self.SCALER_MAG_MIN, self.SCALER_MAG_MAX
                )

                if self.scaler_input:
                    input_scaled = self.scaler_input.transform(input_data)
                else:
                    input_scaled = input_data

                input_reshaped = input_scaled.reshape((1, 1, 4))
                pred_scaled = self.model.predict(input_reshaped, verbose=0)

                # Model outputs 3 values: [magnitude, latitude, longitude]
                if self.scaler_output:
                    pred = self.scaler_output.inverse_transform(pred_scaled)[0]
                else:
                    pred = pred_scaled[0]

                if len(pred) >= 3:
                    bilstm_mag  = float(pred[0])
                    cluster_lat = float(np.clip(pred[1], -90.0, 90.0))
                    cluster_lon = float(np.clip(pred[2], -180.0, 180.0))

            except Exception as e:
                print(f"⚠️  BiLSTM inference failed ({e}); using physics fallback")

        # ── Step 2: Bath's Law ────────────────────────────────────────────────
        baths_mag = mainshock_magnitude - float(np.clip(np.random.normal(1.2, 0.15), 0.8, 1.8))

        # Blend BiLSTM and Bath's Law when model delta is physically plausible
        model_delta = mainshock_magnitude - bilstm_mag
        if use_bilstm and 0.5 <= model_delta <= 2.0:
            largest_mag = 0.75 * baths_mag + 0.25 * bilstm_mag
        else:
            largest_mag = baths_mag

        largest_mag = float(np.clip(largest_mag, 1.0, mainshock_magnitude - 0.5))

        # ── Step 3: Gutenberg-Richter magnitude sampling ─────────────────────
        b_value = 0.95
        min_mag = max(2.0, mainshock_magnitude - 2.5)
        mags = self._sample_gr_magnitudes(top_k, min_mag, largest_mag, b_value)

        # ── Step 4: Wells-Coppersmith zone radius ─────────────────────────────
        zone_radius_km = self._wells_coppersmith_radius(mainshock_magnitude)
        sigma_lat = (zone_radius_km / 2.5) / 111.0
        cos_lat   = max(np.cos(np.radians(mainshock_latitude)), 1e-6)
        sigma_lon = (zone_radius_km / 2.5) / (111.0 * cos_lat)

        # Constrain BiLSTM cluster offset to 50 % of raw model displacement
        lat_offset = float(np.clip(cluster_lat - mainshock_latitude, -2.0, 2.0)) * 0.5
        lon_offset = float(np.clip(cluster_lon - mainshock_longitude, -2.0, 2.0)) * 0.5
        centre_lat = mainshock_latitude + lat_offset
        centre_lon = mainshock_longitude + lon_offset

        # ── Step 5: Scatter aftershocks around cluster centre ─────────────────
        lats = np.clip(np.random.normal(centre_lat, sigma_lat, top_k), -90.0, 90.0)
        lons = np.clip(np.random.normal(centre_lon, sigma_lon, top_k), -180.0, 180.0)

        # ── Step 6: Build predictions with confidence scores ──────────────────
        aftershocks: List[AftershockPrediction] = []
        for i in range(top_k):
            mag_i   = float(mags[i])
            lat_i   = float(lats[i])
            lon_i   = float(lons[i])
            dist_km = self._haversine_km(mainshock_latitude, mainshock_longitude, lat_i, lon_i)

            # Confidence: higher for magnitudes close to Bath's Law value, nearby
            mag_factor  = np.exp(-0.5 * ((mainshock_magnitude - mag_i - 1.2) / 0.5) ** 2)
            dist_factor = np.exp(-dist_km / (zone_radius_km * 0.8))
            raw = 0.55 * mag_factor + 0.45 * dist_factor
            confidence = float(np.clip(raw, 0.05, 0.90))

            # Depth: scale roughly with magnitude and apply uncertainty
            depth_base = max(5.0, mainshock_depth * (mag_i / mainshock_magnitude))
            depth_i    = float(np.clip(
                depth_base + np.random.normal(0, 5.0),
                1.0, min(700.0, mainshock_depth + 30.0)
            ))

            aftershocks.append(AftershockPrediction(
                rank=i + 1,
                magnitude=round(mag_i, 1),
                latitude=round(lat_i, 3),
                longitude=round(lon_i, 3),
                depth_km=round(depth_i, 1),
                confidence=round(confidence, 2),
            ))

        aftershocks.sort(key=lambda a: a.magnitude, reverse=True)
        for i, a in enumerate(aftershocks):
            a.rank = i + 1

        return aftershocks

    # ── Physics helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return float(R * 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))

    @staticmethod
    def _sample_gr_magnitudes(n: int, m_min: float, m_max: float, b: float = 0.95) -> np.ndarray:
        """Sample n magnitudes from the Gutenberg-Richter distribution."""
        u = np.random.uniform(1e-9, 1.0 - 1e-9, n)
        mags = m_min - np.log10(u) / b
        return np.clip(mags, m_min, m_max)

    @staticmethod
    def _wells_coppersmith_radius(magnitude: float) -> float:
        """Aftershock zone radius (km) via Wells & Coppersmith (1994)."""
        rupture_km = 10 ** (-2.44 + 0.59 * magnitude)
        return float(np.clip(rupture_km * 1.5, 15.0, 250.0))
