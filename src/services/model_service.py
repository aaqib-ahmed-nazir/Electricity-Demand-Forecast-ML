"""
Model service for loading and using machine learning models.
"""
import os
import json
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from fastapi import HTTPException
from typing import Dict, Any, List
import traceback

class ModelService:
    """Service for managing ML models for electricity demand forecasting."""
    
    def __init__(self, model_dir):
        """
        Initialize the model service.
        
        Args:
            model_dir (str): Directory containing the trained models
        """
        self.model_dir = model_dir
        self.models = self._load_models()
        if not self.models:
            print("WARNING: Failed to load models. Service may not function correctly.")
    
    def _load_models(self) -> Dict[str, Any]:
        """
        Load all required models from the model directory.
        
        Returns:
            Dict: Dictionary containing loaded models and configurations
        """
        try:
            # Check if model files exist
            xgboost_path = os.path.join(self.model_dir, "xgboost_demand_forecasting.pkl")
            target_scaler_path = os.path.join(self.model_dir, "target_scaler.pkl")
            ensemble_config_path = os.path.join(self.model_dir, "ensemble_weights.json")
            
            if not os.path.exists(xgboost_path):
                print(f"ERROR: XGBoost model file not found at {xgboost_path}")
                return None
                
            if not os.path.exists(target_scaler_path):
                print(f"ERROR: Target scaler file not found at {target_scaler_path}")
                return None
                
            if not os.path.exists(ensemble_config_path):
                print(f"ERROR: Ensemble config file not found at {ensemble_config_path}")
                return None
            
            # Load XGBoost model
            xgboost_model = joblib.load(xgboost_path)
            
            # Load target scaler
            target_scaler = joblib.load(target_scaler_path)
            
            # Load ensemble weights
            with open(ensemble_config_path, "r") as f:
                ensemble_config = json.load(f)
            
            print("Models loaded successfully.")
            return {
                "xgboost": xgboost_model,
                "target_scaler": target_scaler,
                "ensemble_config": ensemble_config
            }
        except Exception as e:
            print(f"Error loading models: {e}")
            print(traceback.format_exc())
            return None
    
    def predict_xgboost(self, X_test):
        """
        Make predictions using the XGBoost model.
        
        Args:
            X_test: Features for prediction
            
        Returns:
            np.array: Array of predictions
        """
        if self.models is None or "xgboost" not in self.models:
            raise HTTPException(status_code=500, detail="XGBoost model not loaded")
        
        try:
            expected_features = self.models["xgboost"].feature_names
            
            if isinstance(X_test, pd.DataFrame):
                current_features = list(X_test.columns)
                
                missing_features = [f for f in expected_features if f not in current_features]
                if missing_features:
                    print(f"Warning: Missing features: {missing_features}")
                    # Add missing features with zeros
                    for feature in missing_features:
                        X_test[feature] = 0.0
                
                X_test = X_test[expected_features]
            
            dtest = xgb.DMatrix(X_test, feature_names=expected_features)
            
            # Make predictions
            predictions = self.models["xgboost"].predict(dtest)
            
            if np.isnan(predictions).any():
                print("WARNING: NaN values detected in predictions. Replacing with zeros.")
                predictions = np.nan_to_num(predictions)
                
            return predictions
        except Exception as e:
            print(f"Error making XGBoost predictions: {e}")
            print(traceback.format_exc())
            
            if "feature_names mismatch" in str(e):
                try:
                    print("Attempting fallback prediction without feature names...")
                    dtest = xgb.DMatrix(X_test)
                    
                    self.models["xgboost"].feature_names = None
                    
                    predictions = self.models["xgboost"].predict(dtest)
                    
                    if np.isnan(predictions).any():
                        print("WARNING: NaN values detected in predictions. Replacing with zeros.")
                        predictions = np.nan_to_num(predictions)
                        
                    return predictions
                except Exception as fallback_error:
                    print(f"Fallback prediction failed: {fallback_error}")
                    print(traceback.format_exc())
            
            raise HTTPException(status_code=500, detail=f"XGBoost prediction error: {str(e)}")
    
    def get_ensemble_weights(self):
        """
        Get the ensemble model weights.
        
        Returns:
            list: List of weights for ensemble components
        """
        if self.models is None or "ensemble_config" not in self.models:
            raise HTTPException(status_code=500, detail="Ensemble configuration not loaded")
        
        try:
            if "weights" in self.models["ensemble_config"]:
                return self.models["ensemble_config"]["weights"]
            else:
                print("Weights field not found in ensemble_config. Using default weights [0.7, 0.3]")
                return [0.7, 0.3]
        except Exception as e:
            print(f"Error getting ensemble weights: {e}")
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error retrieving ensemble weights: {str(e)}")
    
    def create_confidence_bounds(self, predictions, model_type="xgboost"):
        """
        Create confidence bounds for the predictions.
        
        Args:
            predictions (np.array): Array of predictions
            model_type (str): Model type for adjusting bounds
            
        Returns:
            dict: Dictionary with lower and upper bounds
        """
        try:
            if model_type == "xgboost":
                lower_factor, upper_factor = 0.9, 1.1
            else:  # ensemble
                lower_factor, upper_factor = 0.85, 1.15
            
            # Convert to numpy array if it's not already
            predictions_array = np.array(predictions)
            
            if np.isnan(predictions_array).any():
                print("WARNING: NaN values detected in predictions for confidence bounds. Replacing with zeros.")
                predictions_array = np.nan_to_num(predictions_array)
            
            return {
                "lower": [max(0, float(pred * lower_factor)) for pred in predictions_array],
                "upper": [float(pred * upper_factor) for pred in predictions_array]
            }
        except Exception as e:
            print(f"Error creating confidence bounds: {e}")
            print(traceback.format_exc())
            # Return default bounds to avoid breaking the API
            return {
                "lower": [0 for _ in range(len(predictions))],
                "upper": [100 for _ in range(len(predictions))]
            }
