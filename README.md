# SafeLink ML Backend API

Real-time earthquake and flood prediction API for the SafeLink disaster warning system.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and add:
- Your Google Maps API key
- API port (optional, defaults to 8000)

### 3. Run API

```bash
uvicorn main:app --reload --port 8000
```

The API will be available at: http://localhost:8000

### 4. Test the API

Open browser to: http://localhost:8000/docs

Try the following endpoints:

- **GET `/health`** - Check API & model status
- **POST `/alerts/check`** - Check all alerts (earthquakes + floods)
- **POST `/earthquake/check`** - Just earthquake alerts
- **POST `/flood/check`** - Just flood risk

## API Endpoints

### Health Check
```
GET /health
```

Returns status of API and loaded models.

### Main Alert Endpoint (Recommended)
```
POST /alerts/check

Body:
{
  "latitude": 33.5,
  "longitude": 73.5
}
```

Checks both earthquakes and floods. Most efficient for production.

### Earthquake Endpoints

**Check nearby earthquakes:**
```
POST /earthquake/check

Body:
{
  "latitude": 33.5,
  "longitude": 73.5,
  "user_id": "optional_user_id"
}
```

**Predict aftershocks for specific earthquake:**
```
POST /earthquake/predict?magnitude=5.2&depth=15&latitude=33.5&longitude=73.5
```

**Get recent earthquakes globally:**
```
GET /earthquakes/recent?min_magnitude=3.5&hours_back=24
```

### Flood Endpoints

**Check flood risk:**
```
POST /flood/check

Body:
{
  "latitude": 33.5,
  "longitude": 73.5
}
```

## Project Structure

```
backend/
├── main.py                  # FastAPI application
├── earthquake_service.py    # BiLSTM model wrapper
├── flood_service.py         # XGBoost model wrapper
├── data_fetcher.py          # External API clients
├── requirements.txt         # Python dependencies
└── .env.example            # Configuration template
```

## How It Works

### Data Flow

1. **User Location**: Received from Flutter app
2. **Data Fetching**: 
   - USGS API: Real-time earthquake data
   - Weather APIs: Rainfall/discharge data
3. **Predictions**:
   - BiLSTM: Predicts aftershocks
   - XGBoost: Assesses flood risk
4. **Filtering**: Alerts within user's radius
5. **Response**: Returns critical alerts only

### Model Details

#### Earthquake Model
- **Algorithm**: BiLSTM (Bidirectional LSTM)
- **Trained on**: Pakistan earthquake data (2005-2023)
- **Prediction**: Top 5 aftershocks per mainshock
- **Accuracy**: 
  - Magnitude: ±0.24 error
  - Location: ±8-15° error
- **Latency**: <100ms per prediction

#### Flood Model
- **Algorithm**: XGBoost Geospatial Pipeline
- **Input**: Location + recent rainfall + river discharge
- **Output**: Risk level (LOW/MODERATE/HIGH/CRITICAL)
- **Coverage**: Pakistan

## Configuration

Edit `.env` to configure:

```ini
# API
FLASK_ENV=development
API_PORT=8000

# Model Paths
EARTHQUAKE_MODEL_PATH=../lib/Models/Earthquake-Aftershocks/trained_model
FLOOD_MODEL_PATH=../lib/Models/flood-api-data-fetcher

# External APIs
USGS_API_BASE=https://earthquake.usgs.gov/fdsnws/event/1/query
GOOGLE_MAPS_API_KEY=your_key_here

# Alert Thresholds
EARTHQUAKE_ALERT_RADIUS=50  # km
FLOOD_ALERT_RADIUS=75       # km

# Update Intervals
EARTHQUAKE_CHECK_INTERVAL=600   # 10 minutes
FLOOD_CHECK_INTERVAL=3600       # 1 hour
```

## Testing

### Test All Endpoints via Swagger UI

```
http://localhost:8000/docs
```

Click "Try it out" on any endpoint.

### Test with curl

```bash
# Health check
curl http://localhost:8000/health

# Check alerts (Islamabad, Pakistan)
curl -X POST http://localhost:8000/alerts/check \
  -H "Content-Type: application/json" \
  -d '{"latitude": 33.7298, "longitude": 73.1772}'

# Get recent earthquakes
curl "http://localhost:8000/earthquakes/recent?min_magnitude=4&hours_back=48"
```

### Test the Models Directly

```bash
# Test earthquake model in Python
cd ../lib/Models/Earthquake-Aftershocks
/opt/homebrew/bin/python3.11 test_model_accuracy.py

# Test on historical data
/opt/homebrew/bin/python3.11 validate_against_history.py
```

## Deployment

### Local Network Testing

To test from another device on the same network:

1. Find your machine's IP:
   ```bash
   ifconfig | grep inet
   ```

2. Update Flutter app's `API_BASE_URL`:
   ```dart
   static const String API_BASE_URL = 'http://YOUR_IP:8000';
   ```

3. Run API on all interfaces:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

### Production Deployment

For cloud deployment (Heroku, AWS, etc.):

1. Create `Procfile`:
   ```
   web: uvicorn main:app --host 0.0.0.0 --port $PORT
   ```

2. Deploy with dependencies installed

3. Update Flutter app to use production URL

## Troubleshooting

### Models not loading?

**Error**: `⚠️ Model not found at /path/to/model`

```bash
# Check model files exist
ls lib/Models/Earthquake-Aftershocks/trained_model/
ls lib/Models/flood-api-data-fetcher/data/

# Model needs these files:
# Earthquake: aftershock_lstm_model.h5, scaler_input.pkl, scaler_output.pkl
# Flood: xgboost_geospatial_pipeline.pkl, data/heatmap_payload_xgb_*.json
```

### CORS errors?

The API already has CORS enabled for all origins. If you still get CORS errors:

1. Check API is running
2. Use correct API URL in Flutter app
3. Make sure app makes HTTP requests (not HTTPS to localhost)

### Slow responses?

- First request is slow while models load (~2-3s)
- Subsequent requests are faster (~200-500ms)
- Add caching to reduce USGS API calls

### 503 Service Unavailable?

Usually means USGS API is unavailable. The service gracefully handles this and:
- Returns cached data if available
- Falls back to dummy predictions
- Logs the error for debugging

## Performance Tips

1. **Caching**: Implemented automatically (10 min cache)
2. **Batch Requests**: Use `/alerts/check` instead of separate calls
3. **Async**: All external API calls are async
4. **Model Caching**: Models loaded once at startup

## Architecture Overview

```
Request → FastAPI → Data Fetcher
              ↓
         USGS API
         Weather API
              ↓
         Models (Async)
              ↓
         BiLSTM + XGBoost
              ↓
         Response
```

## Dependencies

- **fastapi** - Web framework
- **uvicorn** - ASGI server
- **tensorflow** - Deep learning (BiLSTM)
- **xgboost** - Gradient boosting (Flood)
- **scikit-learn** - ML utilities
- **httpx** - Async HTTP client
- **pydantic** - Data validation

## API Response Examples

### Alert Found
```json
{
  "timestamp": "2024-04-16T10:30:00",
  "earthquake_alerts": [
    {
      "mainshock_magnitude": 5.2,
      "distance_to_user_km": 45.2,
      "should_alert": true,
      "predicted_aftershocks": [
        {
          "rank": 1,
          "magnitude": 4.5,
          "confidence": 0.85
        }
      ]
    }
  ],
  "flood_alert": {
    "risk_level": "HIGH",
    "risk_score": 72.5,
    "should_alert": true
  },
  "has_critical_alerts": true,
  "alert_count": 2
}
```

### No Alerts
```json
{
  "timestamp": "2024-04-16T10:30:00",
  "earthquake_alerts": [],
  "flood_alert": {
    "risk_level": "LOW",
    "risk_score": 15.0,
    "should_alert": false
  },
  "has_critical_alerts": false,
  "alert_count": 0
}
```

## Support

For issues or questions:

1. Check server logs: `uvicorn` output
2. Test API health: GET `/health`
3. Try Swagger UI: `http://localhost:8000/docs`
4. Check model paths match configuration

## License

Part of SafeLink FYP Project
