# Electricity Demand Forecasting API

A machine learning-powered API for forecasting electricity demand, performing clustering analysis of usage patterns, and providing interactive AI assistance for energy forecasting questions.

## Features

- **Demand Forecasting**: Predict future electricity demand using XGBoost and ensemble models
- **Clustering Analysis**: Identify patterns in electricity usage using K-means clustering
- **AI Assistant**: Get explanations and insights about electricity forecasting using a Gemini AI-powered chatbot

## Project Structure

```
├── dataset/               # Data files
│   ├── cleaned_*.csv      # Cleaned datasets
│   ├── processed/         # Processed data
│   └── samples/           # Sample datasets
├── models/                # Trained ML models
│   ├── xgboost_demand_forecasting.pkl
│   ├── lstm_best_model.keras
│   ├── target_scaler.pkl
│   └── ensemble_weights.json
├── notebooks/             # Jupyter notebooks for analysis
├── scripts/               # Python scripts
│   └── electricity_demand_forecasting.py
├── src/                   # API source code
│   ├── models/            # Data schemas
│   ├── routes/            # API endpoints
│   ├── services/          # Business logic
│   └── utils/             # Utility functions
├── .env                   # Environment variables
├── main.py                # FastAPI application
└── requirements.txt       # Dependencies
```

## Installation

1. Clone the repository
```bash
git clone https://github.com/aaqib-ahmed-nazir/data_mining_project.git
cd data_mining_project
```
2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with your Gemini API key:

```
GEMINI_API_KEY="your_gemini_api_key_here"
```

## Running the Application

Start the API server:

```bash
python main.py
```

The API will be available at `http://localhost:8000`.

## API Endpoints

### Root Endpoint

- `GET /`: Get basic information about the API

```bash
curl http://localhost:8000
```

Example response:
```json
{
  "message": "Electricity Demand Forecasting API",
  "status": "online",
  "endpoints": {
    "prediction": "/predict",
    "clustering": "/cluster",
    "chatbot": "/chat"
  },
  "documentation": "/docs"
}
```

### Prediction Endpoint

- `POST /predict`: Predict electricity demand

```bash
curl -L -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "city": "dallas",
    "start_date": "2019-11-01",
    "end_date": "2019-11-07",
    "look_back_window": 168,
    "model": "xgboost"
  }'
```

Example response:
```json
{
  "model": "xgboost",
  "forecast": [45321.2, 44933.8, 44533.1, ...],
  "actual": [45421.0, 45012.5, 44619.8, ...],
  "timestamps": ["2019-11-01 00:00:00", "2019-11-01 01:00:00", ...],
  "confidence_bounds": {
    "lower": [40789.1, 40440.4, 40079.8, ...],
    "upper": [49853.3, 49427.2, 48986.4, ...]
  }
}
```

### Clustering Endpoint

- `POST /cluster`: Perform clustering analysis on electricity demand data

```bash
curl -L -X POST http://localhost:8000/cluster \
  -H "Content-Type: application/json" \
  -d '{
    "city": "dallas",
    "num_clusters": 3,
    "features": ["demand", "temperature_scaled", "hour_sin", "hour_cos"]
  }'
```

Example response:
```json
{
  "clusters": [
    {
      "cluster_id": 0,
      "size": 423,
      "center": [0.68, -0.12, 0.54, 0.83],
      "points": [
        {
          "x": 0.78,
          "y": -0.15,
          "demand": 45231.5,
          "timestamp": "2019-10-01 13:00:00"
        },
        ...
      ]
    },
    ...
  ],
  "pca_components": {
    "x_label": "PC1 (45.20% variance)",
    "y_label": "PC2 (32.15% variance)"
  },
  "cluster_centers": [[0.68, -0.12, 0.54, 0.83], ...],
  "feature_importance": {
    "demand": 0.72,
    "temperature_scaled": 0.64,
    "hour_sin": 0.34,
    "hour_cos": 0.41
  }
}
```

### Chat Assistant Endpoint

- `POST /chat`: Ask questions about electricity demand forecasting

```bash
curl -L -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What factors affect electricity demand forecasting?"
  }'
```

Example response:
```json
{
  "response": "Electricity demand forecasting is influenced by several key factors:\n\n1. Weather conditions - temperature, humidity, and precipitation have significant impacts on electricity usage, especially for heating and cooling.\n\n2. Temporal patterns - time of day, day of week, and seasonal cycles create regular patterns in demand.\n\n3. Economic factors - industrial activity, GDP growth, and business cycles affect commercial and industrial electricity usage.\n\n4. Demographic changes - population growth and shifts in population density impact residential demand patterns.\n\n5. Special events - holidays, major sporting events, or unexpected situations can cause significant deviations from normal patterns.\n\nOur forecasting models incorporate these factors using features like temperature data, cyclical time encodings, and historical demand patterns to make accurate predictions.",
  "model": "gemini-1.5-pro"
}
```

## Troubleshooting API Requests

If you're experiencing 307 Temporary Redirect status codes with your cURL requests, try these solutions:

1. **Add the `-L` flag to your cURL command** to follow redirects automatically:
   ```bash
   curl -L -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"city": "dallas",...}'
   ```

2. **Add a trailing slash** to the endpoint URL:
   ```bash
   curl -X POST http://localhost:8000/predict/ -H "Content-Type: application/json" -d '{"city": "dallas",...}'
   ```

3. **Restart the API server** after making changes to the `main.py` file to ensure the new configuration takes effect.

4. If using **Postman or other API clients**, make sure to enable the option to follow redirects.

## API Documentation

Access the interactive API documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Models

The API uses machine learning models for predictions:

- **XGBoost**: Gradient boosting model optimized for tabular data
- **Ensemble**: Weighted combination of multiple models for improved accuracy

## Error Handling

The API returns appropriate HTTP status codes:
- 200: Successful response
- 400: Invalid request parameters
- 404: Resource not found
- 500: Server error

## Dataset

The project uses electricity demand data with the following features:
- Timestamp
- Demand (kWh)
- Weather data (temperature, humidity, etc.)
- Location information

## Development

### Adding New Features

To add new features to the API, follow these steps:

1. Define new schema models in `src/models/schemas.py`
2. Create service implementations in `src/services/`
3. Add new route handlers in `src/routes/`
4. Update the main application in `main.py`

### Training New Models

The script `scripts/electricity_demand_forecasting.py` contains the full pipeline for training new forecasting models.
