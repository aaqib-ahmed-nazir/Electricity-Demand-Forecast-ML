"""
API routes for electricity demand prediction.
"""
import os
from fastapi import APIRouter, HTTPException
import xgboost as xgb
import pandas as pd
import traceback
from datetime import datetime, timedelta

from src.models.schemas import PredictionRequest
from src.services.model_service import ModelService
from src.utils.data_processor import load_dataset, preprocess_data, create_features_targets

router = APIRouter(prefix="/predict", tags=["Predictions"])

# Define constants
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "dataset/processed/samples")

# Initialize model service
model_service = ModelService(MODEL_DIR)

# Define data availability constants
DATA_START_DATE = "2018-07-01"  # Earliest date in our dataset
DATA_END_DATE = "2020-05-31"    # Latest date in our dataset

@router.post("")  # This will match /predict
async def predict(request: PredictionRequest):
    """
    Predict electricity demand for the given parameters.
    """
    try:
        # Validate date range is within available data
        start_date = pd.to_datetime(request.start_date)
        end_date = pd.to_datetime(request.end_date)
        data_start = pd.to_datetime(DATA_START_DATE)
        data_end = pd.to_datetime(DATA_END_DATE)
        
        # If requested dates are outside our dataset range, provide a helpful error
        if end_date < data_start or start_date > data_end:
            return {
                "error": "No data available for the selected date range",
                "available_data_range": f"{DATA_START_DATE} to {DATA_END_DATE}",
                "requested_range": f"{request.start_date} to {request.end_date}",
                "status": "error"
            }
        
        # If dates are partially outside range, adjust them and warn
        date_adjusted = False
        original_start = start_date
        original_end = end_date
        
        if start_date < data_start:
            start_date = data_start
            date_adjusted = True
            
        if end_date > data_end:
            end_date = data_end
            date_adjusted = True
        
        # Load dataset
        data_path = os.path.join(DATA_DIR, "sample_10000_clean_merged_data.csv")
        df = load_dataset(data_path)
        
        if df is None:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Preprocess data
        processed_df, feature_columns = preprocess_data(df, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        if processed_df.empty:
            return {
                "error": "No data available for the selected date range",
                "available_data_range": f"{DATA_START_DATE} to {DATA_END_DATE}",
                "requested_range": f"{request.start_date} to {request.end_date}",
                "suggestion": "Try selecting a different date range or city",
                "status": "error"
            }
        
        # Get predictions based on the requested model
        if request.model == "xgboost":
            result = predict_xgboost(processed_df, feature_columns, request.look_back_window)
        else:  # ensemble (defaulting to xgboost if ensemble is not available)
            result = predict_ensemble(processed_df, feature_columns, request.look_back_window)
            
        # Add a warning if dates were adjusted
        if date_adjusted:
            result["date_range_adjusted"] = True
            result["original_requested_range"] = f"{original_start.strftime('%Y-%m-%d')} to {original_end.strftime('%Y-%m-%d')}"
            result["adjusted_range"] = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
            result["warning"] = "The requested date range was adjusted to match available data"
        
        result["status"] = "success"
        return result
            
    except Exception as e:
        # Log the full error for debugging
        print(f"Prediction error: {str(e)}")
        print(traceback.format_exc())
        
        # Return a structured error response rather than throwing an exception
        return {
            "error": f"Prediction error: {str(e)}",
            "status": "error",
            "detail": "There was an error processing your prediction request",
            "available_data_range": f"{DATA_START_DATE} to {DATA_END_DATE}"
        }

def predict_xgboost(df, feature_columns, look_back_window):
    """
    Generate predictions using the XGBoost model.
    """
    try:
        # Create features for XGBoost
        X, y = create_features_targets(df, lag_hours=24, window_size=look_back_window)
        
        if X.empty:
            raise ValueError("Failed to create features - insufficient data for the given window size")
        
        # Make predictions
        predictions = model_service.predict_xgboost(X)
        
        # Format the results
        result = {
            "model": "xgboost",
            "forecast": predictions.tolist(),
            "actual": y.values.tolist() if y is not None and not y.empty else [],
            "timestamps": df['datetime'].iloc[-len(predictions):].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            "confidence_bounds": model_service.create_confidence_bounds(predictions, "xgboost"),
            "data_points": len(predictions)
        }
        
        return result
    except Exception as e:
        print(f"XGBoost prediction error: {str(e)}")
        print(traceback.format_exc())
        raise ValueError(f"XGBoost prediction failed: {str(e)}")

def predict_ensemble(df, feature_columns, look_back_window):
    """
    Generate predictions using the ensemble model (XGBoost + weights).
    """
    try:
        # Get XGBoost predictions
        xgb_result = predict_xgboost(df, feature_columns, look_back_window)
        xgb_preds = pd.Series(xgb_result["forecast"])
        
        # Since we don't have LSTM with Python 3.12, we'll simulate ensemble with weights
        # Get ensemble weights
        weights = model_service.get_ensemble_weights()
        
        # Create ensemble predictions - adapt the weights
        # We're using only XGBoost predictions, so we adjust the formula:
        # Original: ensemble_preds = weights[0] * xgb_preds + weights[1] * lstm_preds
        # Adjusted: ensemble_preds = (weights[0] + weights[1] * 0.9) * xgb_preds 
        adjusted_weight = weights[0] + (weights[1] * 0.9)
        ensemble_preds = xgb_preds * adjusted_weight
        
        # Format the results
        result = {
            "model": "ensemble",
            "forecast": ensemble_preds.tolist(),
            "actual": xgb_result["actual"],
            "timestamps": xgb_result["timestamps"],
            "confidence_bounds": model_service.create_confidence_bounds(ensemble_preds, "ensemble"),
            "component_predictions": {
                "xgboost": xgb_preds.tolist(),
                "weight_adjustment": f"Using adjusted weight of {adjusted_weight:.2f} for XGBoost predictions to simulate ensemble"
            },
            "note": "LSTM model is not available with Python 3.12, so ensemble is simulated using XGBoost with adjusted weights",
            "data_points": len(ensemble_preds)
        }
        
        return result
    except Exception as e:
        print(f"Ensemble prediction error: {str(e)}")
        print(traceback.format_exc())
        raise ValueError(f"Ensemble prediction failed: {str(e)}")
