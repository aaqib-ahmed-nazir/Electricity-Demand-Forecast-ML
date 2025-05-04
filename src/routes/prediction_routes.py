"""
API routes for electricity demand prediction.
"""
import os
from fastapi import APIRouter, HTTPException
import xgboost as xgb
import pandas as pd
import traceback

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

@router.post("")  # This will match /predict
async def predict(request: PredictionRequest):
    """
    Predict electricity demand for the given parameters.
    """
    try:
        # Load dataset
        data_path = os.path.join(DATA_DIR, "sample_10000_clean_merged_data.csv")
        df = load_dataset(data_path)
        
        if df is None:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Preprocess data
        processed_df, feature_columns = preprocess_data(df, request.start_date, request.end_date)
        
        if processed_df.empty:
            raise HTTPException(status_code=400, detail="No data available for the selected date range")
        
        # Get predictions based on the requested model
        if request.model == "xgboost":
            return predict_xgboost(processed_df, feature_columns, request.look_back_window)
        else:  # ensemble (defaulting to xgboost if ensemble is not available)
            return predict_ensemble(processed_df, feature_columns, request.look_back_window)
            
    except Exception as e:
        # Log the full error for debugging
        print(f"Prediction error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

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
            "confidence_bounds": model_service.create_confidence_bounds(predictions, "xgboost")
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
            "note": "LSTM model is not available with Python 3.12, so ensemble is simulated using XGBoost with adjusted weights"
        }
        
        return result
    except Exception as e:
        print(f"Ensemble prediction error: {str(e)}")
        print(traceback.format_exc())
        raise ValueError(f"Ensemble prediction failed: {str(e)}")
