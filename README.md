# Electricity Demand Forecasting API

A machine learning-powered API for forecasting electricity demand, performing clustering analysis of usage patterns, and providing interactive AI assistance for energy forecasting questions.

## Features

- **Demand Forecasting**: Predict future electricity demand using XGBoost and ensemble models
- **Clustering Analysis**: Identify patterns in electricity usage using K-means clustering
- **AI Assistant**: Get explanations and insights about electricity forecasting using a Gemini AI-powered chatbot
- **Interactive UI**: Modern React frontend with visualizations and intuitive interface
- **Real-time Analytics**: Live data processing and visualization capabilities

## Project Overview

This project aims to predict electricity demand patterns by leveraging historical consumption data and various influential factors such as weather conditions, time patterns, and seasonal variations. The system uses advanced machine learning techniques to provide accurate forecasts that can help energy providers optimize resource allocation and plan for future demand.

### Why Electricity Demand Forecasting?

Accurate electricity demand forecasting is crucial for:
- **Grid Stability**: Ensuring balanced supply and demand
- **Resource Optimization**: Reducing waste and operational costs
- **Renewable Integration**: Better planning for intermittent renewable energy sources
- **Environmental Impact**: Minimizing carbon footprint through efficient energy use
- **Economic Planning**: Making informed investment decisions for infrastructure

### Key Technologies

- **Backend**: FastAPI, Python, scikit-learn, TensorFlow, XGBoost
- **Frontend**: React, Vite, Tailwind CSS, Shadcn UI components
- **Data Processing**: Pandas, NumPy, SciPy
- **Visualization**: Recharts, D3.js
- **AI Integration**: Google Gemini API for conversational assistance

## User Interface

The application features a modern, responsive interface designed for both technical and non-technical users.

### Main Dashboard

![Main Dashboard](/assets/Main_page.png)

The dashboard provides an overview of current demand patterns, historical data, and forecast visualizations. Users can select different cities and timeframes for analysis.

### Dark Mode Support

![Dark Mode Dashboard](/assets/Main_page_Dark_mode.png)

The application includes full dark mode support for comfortable viewing in all environments.

### Detailed Analytics

![Additional Analytics](/assets/Main_page2.png)

Deeper insights are available through advanced visualization tools, showing correlations between electricity demand and other factors like temperature and time of day.

### Interactive Chat Assistant

![Chat Interface](/assets/Chat_Page.png)

The AI-powered chat interface allows users to ask questions about electricity demand, get insights about the data, and receive explanations about forecasting methodologies in natural language.

### Documentation & Help

![Documentation](/assets/Document_and_help.png)

Comprehensive documentation helps users understand how to use the application effectively and interpret results correctly.

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
├── frontend/              # React frontend application
│   ├── src/               # Frontend source code
│   ├── components/        # UI components
│   └── pages/             # Application pages
├── assets/                # Images and static assets
├── .env                   # Environment variables
├── main.py                # FastAPI application
└── requirements.txt       # Dependencies
```

## Technical Approach

### Data Processing Pipeline

Our data processing pipeline involves several key steps:
1. **Data Collection**: Gathering electricity demand data from EIA930 datasets along with weather information from multiple cities
2. **Data Loading and Inspection**:
   - Standardizing city names across all data sources
   - Converting timestamps to datetime objects
   - Merging data on both city and timestamp using inner join
   - Validating schema and data integrity
3. **Missing Value Handling**:
   - Time-series features: Linear interpolation for short gaps, pattern-based imputation for longer gaps
   - Weather features: City-specific median imputation, forward fill for gradual changes
4. **Feature Engineering**:
   - Temporal features: Cyclical encoding of time using sine/cosine transformations
   - Weather transformations: Scaling, non-linear transformations, and binned features
   - Lag features: Previous day/week demand and rolling statistics across multiple windows
5. **Anomaly Detection**:
   - Statistical methods: Z-score, IQR, and rolling z-score
   - ML methods: Isolation Forest, One-class SVM, and LSTM-Autoencoder
   - Domain-specific rules for physical impossibilities
6. **Dimensionality Reduction**: Using PCA for visualization and clustering analysis
7. **Model Training**: Developing and training various forecasting models

### Models Implemented

We implemented multiple forecasting approaches to capture different aspects of demand patterns:

- **XGBoost**: Excellent for capturing non-linear relationships and feature interactions
- **LSTM Neural Networks**: Specialized in learning long-term dependencies in time series data
- **Ensemble Model**: Combining predictions from multiple models for improved accuracy and robustness
- **Linear and Polynomial Regression**: Used as interpretable baselines
- **Time Series Models**: ARIMA/SARIMA for temporal dependencies
- **Random Forest**: Robust to outliers, captures non-linear patterns

### Clustering Analysis

The clustering module uses unsupervised learning techniques to identify natural patterns in electricity consumption:

- **K-means Clustering**: Groups similar demand patterns based on multiple features
  - Used elbow method and silhouette scores to determine optimal cluster count (4-6 clusters)
  - Provides simple interpretation and even cluster sizes
  - Optimal for our visualization purposes

- **DBSCAN**: Identifies unusual demand patterns that may indicate anomalies
  - Automatically detects the number of clusters based on density
  - Particularly effective at finding irregular-shaped clusters
  - Identifies noise points that don't belong to any cluster

- **Hierarchical Clustering**: Provides nested grouping structures for analyzing relationships
  - Uses agglomerative clustering with different linkage methods
  - Creates dendrograms to visualize clustering hierarchy
  - Enables multi-level analysis of demand pattern relationships

#### Dimensionality Reduction Techniques

To effectively visualize and analyze the high-dimensional electricity demand data, we applied multiple dimensionality reduction techniques:

- **Principal Component Analysis (PCA)**: Linear technique that captures 70-80% of variance with 2-3 components
- **t-SNE**: Non-linear technique that preserves local relationships and reveals complex patterns
- **UMAP**: Balances local and global structure preservation with faster computation for large datasets

#### Cluster Interpretation

After generating clusters, we perform extensive analysis to characterize each segment:

- **Feature Importance Analysis**: Identifies which features most strongly define each cluster
- **Cluster Profiling**: Statistical summaries and radar charts for multi-dimensional profiles
- **Temporal Distribution**: Analysis of how clusters distribute across times of day, days of week, and seasons

Example cluster interpretations:
- **Cluster 0**: "High-demand hot afternoons" - Peak demand during hot summer afternoons (3-7 PM)
- **Cluster 1**: "Low-demand cool nights" - Minimal demand during night hours with mild temperatures
- **Cluster 2**: "Winter morning peaks" - Demand spikes during cold winter mornings (6-9 AM)
- **Cluster 3**: "Weekend plateau" - Sustained medium demand throughout weekend days
- **Cluster 4**: "Commercial workday" - Steady high demand during business hours

## Model Development

### Problem Formulation

We formulated the electricity demand forecasting challenge as a supervised learning regression problem with the following characteristics:

1. **Forecasting Horizon**
   - Primary task: 24-hour ahead hourly forecasts
   - Secondary tasks: 7-day ahead daily forecasts and next-hour forecasts
   - Models designed to be retrained daily with sliding window approach

2. **Target Variable**
   - Hourly electricity demand in kWh
   - Both raw demand values and percentage changes

3. **Feature Set**
   - Calendar features (time, date, holidays)
   - Weather features (current and forecast)
   - Historical demand (lagged values and patterns)
   - Engineered features (interactions, transformations)

4. **Data Splitting Strategy**
   - Chronological split to respect time series nature
   - Training: 70% of historical data
   - Validation: 15% of data 
   - Test: Most recent 15% of data
   - Rolling window evaluation to simulate real-world forecasting

### Training and Validation Strategy

We employed a robust training and validation approach:

1. **Temporal Cross-Validation**
   - Used time series cross-validation with expanding window
   - Ensured no data leakage from future to past
   - Simulated real-world forecasting scenario

2. **Hyperparameter Tuning**
   - Grid search and random search for traditional models
   - Bayesian optimization for complex model tuning
   - Learning rate scheduling for neural networks

3. **Feature Importance Analysis**
   - Used SHAP values to understand model decisions
   - Permutation importance to identify critical features
   - Recursive feature elimination to optimize feature set

### Baseline Comparison

We established naive forecasting baselines to benchmark our advanced models:

1. **Previous Day Baseline**
   - Uses the same hour from the previous day as prediction
   - Simple but surprisingly effective in some cases

2. **Previous Week Baseline**
   - Uses the same hour and day from previous week
   - Captures both time-of-day and day-of-week patterns

3. **Historical Average Baseline**
   - Uses the average for that hour and day of week
   - Smooths out anomalies but misses recent trends

The baseline comparisons showed:
- Advanced models outperformed naive methods by 15-40% on MAPE
- XGBoost achieved the best overall performance, especially for peak hours
- LSTM models performed particularly well for longer forecast horizons
- Ensemble methods consistently beat individual models across all metrics

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

3. Install required packages:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your Gemini API key:

```
GEMINI_API_KEY="your_gemini_api_key_here"
```

5. Install and set up the frontend:

```bash
cd frontend
npm install
```

## Running the Application

### Backend API

Start the API server:

```bash
python main.py
```

The API will be available at `http://localhost:8000`.

### Frontend Application

In a separate terminal:

```bash
cd frontend
npm run dev
```

The frontend will be available at `http://localhost:5173`.

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

The API uses several machine learning models for predictions:

### XGBoost

A gradient boosting framework optimized for tabular data. Our implementation:
- Handles non-linear relationships between features
- Captures complex feature interactions
- Provides feature importance for interpretability
- Offers excellent performance for time-series forecasting

### LSTM Neural Network

Long Short-Term Memory networks are specialized for sequential data:
- Capable of learning long-term dependencies in demand patterns
- Robust to noise in the data
- Effective at capturing seasonal and cyclical patterns
- Particularly strong for multi-step forecasting

### Ensemble Model

A weighted combination of multiple models to leverage their strengths:
- Reduces overfitting and improves generalization
- Balances biases of individual models
- Provides more robust predictions across different scenarios
- Weights determined through validation to maximize accuracy

## Evaluation Metrics

Our models are evaluated using multiple metrics:
- **RMSE (Root Mean Square Error)**: Measures overall prediction accuracy
- **MAE (Mean Absolute Error)**: Less sensitive to outliers than RMSE
- **MAPE (Mean Absolute Percentage Error)**: Provides relative error perspective
- **R² (Coefficient of Determination)**: Indicates how well the model explains variance

## Dataset

The project uses electricity demand data with the following features:
- **Timestamp**: Date and time of the observation
- **Demand**: Electricity consumption in kWh
- **Weather data**: Temperature, humidity, wind speed, pressure, precipitation
- **Temporal features**: Hour, day of week, month, season
- **Derived features**: Cyclical encodings of time features
- **Location information**: City and subregion details

### Data Sources

The data comes from multiple sources:
- EIA930 electricity demand datasets
- Weather historical data for each city
- Additional geographic and demographic information

## Development

### Adding New Features

To add new features to the API, follow these steps:

1. Define new schema models in `src/models/schemas.py`
2. Create service implementations in `src/services/`
3. Add new route handlers in `src/routes/`
4. Update the main application in `main.py`

### Training New Models

The script `scripts/electricity_demand_forecasting.py` contains the full pipeline for training new forecasting models. To train a new model:

1. Prepare your dataset according to the required format
2. Configure hyperparameters in the script
3. Run the training pipeline:
   ```bash
   python scripts/electricity_demand_forecasting.py --model xgboost --config config/model_config.json
   ```
4. Evaluate the model performance using the metrics provided
5. Save the best performing model to the models directory

## Future Work

We plan to extend this project with:
- **Real-time data integration**: Connecting to live electricity data feeds
- **Reinforcement learning**: For adaptive forecasting strategies
- **Uncertainty quantification**: More sophisticated confidence intervals
- **Explainable AI**: Enhanced interpretability of complex models
- **Transfer learning**: Applying knowledge from one city to another

## License

This project is licensed under the MIT License - see the [LICENSE](https://opensource.org/license/mit) file for details.

