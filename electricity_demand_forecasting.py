import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import json
import os
from datetime import datetime
import math  # Add import for math to calculate sqrt

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class ElectricityDemandForecaster:
    """
    A class to handle electricity demand forecasting using multiple models and ensemble techniques.
    """
    
    def __init__(self, data_path=None, forecast_horizon=24):
        """
        Initialize the forecaster with data path and forecast horizon.
        
        Args:
            data_path (str): Path to the CSV data file
            forecast_horizon (int): Number of hours to forecast ahead (default: 24)
        """
        self.data_path = data_path
        self.forecast_horizon = forecast_horizon
        self.df = None
        self.train_df = None
        self.test_df = None
        self.feature_columns = None
        self.target_column = 'demand'
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        
        # Create directories for saving models if they don't exist
        os.makedirs('models', exist_ok=True)
    
    def load_data(self, data_path=None):
        """
        Load data from CSV file.
        
        Args:
            data_path (str, optional): Path to the CSV data file. If not provided, 
                                      will use the one from initialization.
        """
        if data_path:
            self.data_path = data_path
            
        print(f"Loading data from: {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        
        # Check for missing values
        missing_values = self.df.isnull().sum()
        print("\nMissing values in each column:")
        print(missing_values)
        
        # Basic information about the dataset
        print("\nDataset info:")
        print(f"Number of rows: {self.df.shape[0]}")
        print(f"Number of columns: {self.df.shape[1]}")
        print("\nColumn names:")
        print(self.df.columns.tolist())
        
        return self.df
    
    def preprocess_data(self):
        """
        Preprocess the data by converting timestamps, extracting features, and scaling.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("\nPreprocessing data...")
        
        # Convert timestamp to datetime
        self.df['datetime'] = pd.to_datetime(self.df['timestamp'])
        
        # Extract temporal features
        self.df['hour'] = self.df['datetime'].dt.hour
        self.df['dayofweek'] = self.df['datetime'].dt.dayofweek
        self.df['day'] = self.df['datetime'].dt.day
        self.df['month'] = self.df['datetime'].dt.month
        
        # Create cyclical features for time variables
        self.df['hour_sin'] = np.sin(2 * np.pi * self.df['hour']/24)
        self.df['hour_cos'] = np.cos(2 * np.pi * self.df['hour']/24)
        self.df['day_sin'] = np.sin(2 * np.pi * self.df['day']/31)
        self.df['day_cos'] = np.cos(2 * np.pi * self.df['day']/31)
        self.df['month_sin'] = np.sin(2 * np.pi * self.df['month']/12)
        self.df['month_cos'] = np.cos(2 * np.pi * self.df['month']/12)
        
        # Create weekend indicator
        self.df['is_weekend'] = self.df['dayofweek'].isin([5, 6]).astype(int)
        
        # Scale weather features
        weather_features = ['temperature', 'humidity', 'windSpeed', 'pressure', 
                           'precipIntensity', 'precipProbability']
        
        scaler = MinMaxScaler()
        
        for feature in weather_features:
            if feature in self.df.columns:
                self.df[f'{feature}_scaled'] = scaler.fit_transform(self.df[[feature]])
        
        # Define feature columns
        self.feature_columns = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 
                              'month_sin', 'month_cos', 'is_weekend']
                              
        # Add scaled weather features to feature columns
        for feature in weather_features:
            if f'{feature}_scaled' in self.df.columns:
                self.feature_columns.append(f'{feature}_scaled')
        
        print(f"Features created: {self.feature_columns}")
        
        return self.df
    
    def split_train_test(self, test_size=0.2):
        """
        Split the data into training and testing sets.
        
        Args:
            test_size (float): Proportion of data to use for testing (default: 0.2)
        """
        if self.df is None:
            raise ValueError("Data not preprocessed. Call preprocess_data() first.")
        
        print("\nSplitting data into training and testing sets...")
        
        # Use the last test_size proportion of the data for testing
        train_size = int(len(self.df) * (1 - test_size))
        self.train_df = self.df.iloc[:train_size].copy()
        self.test_df = self.df.iloc[train_size:].copy()
        
        # Define features and target variable
        self.X_train = self.train_df[self.feature_columns]
        self.y_train = self.train_df[self.target_column]
        self.X_test = self.test_df[self.feature_columns]
        self.y_test = self.test_df[self.target_column]
        
        print(f"Training set: {self.train_df.shape[0]} samples")
        print(f"Testing set: {self.test_df.shape[0]} samples")
        
        return self.X_train, self.y_train, self.X_test, self.y_test
    
    def implement_naive_forecast(self):
        """
        Implement a naive forecast model using the value from 24 hours ago.
        """
        print("\nImplementing naive forecast model...")
        
        # Create a shifted version of the demand
        naive_predictions = self.df[self.target_column].shift(self.forecast_horizon)
        
        # Evaluate naive forecast on test set
        train_size = len(self.train_df)
        naive_test_predictions = naive_predictions.iloc[train_size:]
        naive_test_actuals = self.y_test
        
        # Remove NaN values that come from the shifting
        valid_idx = ~naive_test_predictions.isna()
        naive_test_predictions = naive_test_predictions[valid_idx]
        naive_test_actuals = naive_test_actuals[valid_idx]
        
        # Calculate error metrics
        naive_mae = mean_absolute_error(naive_test_actuals, naive_test_predictions)
        naive_mse = mean_squared_error(naive_test_actuals, naive_test_predictions)
        naive_rmse = math.sqrt(naive_mse)
        naive_mape = mean_absolute_percentage_error(naive_test_actuals, naive_test_predictions) * 100
        
        # Store metrics and predictions
        self.models['naive'] = 'Naive Forecast'
        self.predictions['naive'] = naive_test_predictions
        self.metrics['naive'] = {
            'mae': naive_mae,
            'rmse': naive_rmse,
            'mape': naive_mape
        }
        
        print("Naive Forecast Performance (24-hour lag):")
        print(f"MAE: {naive_mae:.2f}")
        print(f"RMSE: {naive_rmse:.2f}")
        print(f"MAPE: {naive_mape:.2f}%")
        
        return naive_test_predictions, naive_test_actuals
    
    def create_features_targets(self, data, lag_hours=24, window_size=7*24):
        """
        Create a dataset with lagged features for time series forecasting.
        
        Args:
            data: DataFrame containing the data
            lag_hours: Number of hours to lag for forecasting
            window_size: Size of the window to consider for historical data
            
        Returns:
            X: Features DataFrame
            y: Target Series
        """
        lagged_features = pd.DataFrame()
        
        # Add lagged target variables (demand from previous hours)
        for lag in range(lag_hours, lag_hours + window_size, 24):  # Use data from the same hour in previous days
            lagged_features[f'demand_lag_{lag}'] = data[self.target_column].shift(lag)
        
        # Add original features
        for col in self.feature_columns:
            lagged_features[col] = data[col]
        
        # Target is the original demand
        targets = data[self.target_column]
        
        # Remove rows with NaN values due to lagging
        valid_idx = ~lagged_features.isna().any(axis=1)
        
        return lagged_features[valid_idx], targets[valid_idx]
    
    def implement_xgboost(self):
        """
        Implement an XGBoost model for time series forecasting.
        """
        print("\nImplementing XGBoost model...")
        
        # Create time series features with lagged values
        X_train_ts, y_train_ts = self.create_features_targets(self.train_df)
        X_test_ts, y_test_ts = self.create_features_targets(self.test_df)
        
        print(f"Training set with time series features: {X_train_ts.shape}")
        print(f"Testing set with time series features: {X_test_ts.shape}")
        
        # Set XGBoost parameters
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'eta': 0.1,  # learning rate
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': 42
        }
        
        print("Training XGBoost model...")
        
        # Convert data to DMatrix format for XGBoost
        dtrain = xgb.DMatrix(X_train_ts, label=y_train_ts)
        dvalid = xgb.DMatrix(X_test_ts, label=y_test_ts)
        
        # Set up evaluation list
        evals = [(dtrain, 'train'), (dvalid, 'valid')]
        
        # Train model with early stopping
        xgb_model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=100,
            evals=evals,
            early_stopping_rounds=10,
            verbose_eval=False
        )
        
        # Make predictions
        xgb_predictions = xgb_model.predict(dvalid)
        
        # Calculate error metrics
        xgb_mae = mean_absolute_error(y_test_ts, xgb_predictions)
        xgb_mse = mean_squared_error(y_test_ts, xgb_predictions)
        xgb_rmse = math.sqrt(xgb_mse)
        xgb_mape = mean_absolute_percentage_error(y_test_ts, xgb_predictions) * 100
        
        # Store model, predictions and metrics
        self.models['xgboost'] = xgb_model
        self.predictions['xgboost'] = {
            'predictions': xgb_predictions,
            'actuals': y_test_ts,
            'X_test': X_test_ts
        }
        self.metrics['xgboost'] = {
            'mae': xgb_mae,
            'rmse': xgb_rmse,
            'mape': xgb_mape
        }
        
        print("XGBoost Model Performance:")
        print(f"MAE: {xgb_mae:.2f}")
        print(f"RMSE: {xgb_rmse:.2f}")
        print(f"MAPE: {xgb_mape:.2f}%")
        
        return xgb_model, xgb_predictions, y_test_ts, X_test_ts
    
    def create_sequences(self, data, seq_length):
        """
        Create sequences for LSTM model
        
        Args:
            data: DataFrame containing features and target
            seq_length: Length of the sequence
            
        Returns:
            X: Input sequences
            y: Target values
        """
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data.iloc[i:i+seq_length].values)
            y.append(data.iloc[i+seq_length][self.target_column])
        return np.array(X), np.array(y)
    
    def implement_lstm(self, seq_length=24, epochs=50, batch_size=32):
        """
        Implement an LSTM model for time series forecasting.
        
        Args:
            seq_length (int): Length of the sequence for LSTM
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            Tuple containing the LSTM model, predictions, and actual values
        """
        print("\nImplementing LSTM model...")
        
        # Standardize the data for LSTM
        scaler = StandardScaler()
        lstm_train_data = self.train_df[self.feature_columns + [self.target_column]].copy()
        lstm_test_data = self.test_df[self.feature_columns + [self.target_column]].copy()
        
        # Standardize the target variable for better LSTM performance
        target_scaler = StandardScaler()
        lstm_train_data[self.target_column] = target_scaler.fit_transform(lstm_train_data[[self.target_column]])
        lstm_test_data[self.target_column] = target_scaler.transform(lstm_test_data[[self.target_column]])
        
        # Create sequences
        X_train_seq, y_train_seq = self.create_sequences(lstm_train_data, seq_length)
        X_test_seq, y_test_seq = self.create_sequences(lstm_test_data, seq_length)
        
        print(f"LSTM training sequences shape: {X_train_seq.shape}")
        print(f"LSTM testing sequences shape: {X_test_seq.shape}")
        
        # Build the LSTM model
        lstm_model = Sequential([
            LSTM(units=50, activation='relu', return_sequences=True, input_shape=(seq_length, X_train_seq.shape[2])),
            Dropout(0.2),
            LSTM(units=50, activation='relu'),
            Dropout(0.2),
            Dense(units=1)
        ])
        
        lstm_model.compile(optimizer='adam', loss='mse')
        
        # Define early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train the model
        print("Training LSTM model...")
        lstm_history = lstm_model.fit(
            X_train_seq, y_train_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Make predictions
        lstm_predictions = lstm_model.predict(X_test_seq)
        lstm_predictions = target_scaler.inverse_transform(lstm_predictions)
        y_test_actual = target_scaler.inverse_transform(y_test_seq.reshape(-1, 1))
        
        # Calculate error metrics
        lstm_mae = mean_absolute_error(y_test_actual, lstm_predictions)
        lstm_mse = mean_squared_error(y_test_actual, lstm_predictions)
        lstm_rmse = math.sqrt(lstm_mse)
        lstm_mape = mean_absolute_percentage_error(y_test_actual, lstm_predictions) * 100
        
        # Store model, predictions and metrics
        self.models['lstm'] = {
            'model': lstm_model,
            'target_scaler': target_scaler
        }
        self.predictions['lstm'] = {
            'predictions': lstm_predictions,
            'actuals': y_test_actual,
            'X_test': X_test_seq
        }
        self.metrics['lstm'] = {
            'mae': lstm_mae,
            'rmse': lstm_rmse,
            'mape': lstm_mape,
            'history': lstm_history.history
        }
        
        print("LSTM Model Performance:")
        print(f"MAE: {lstm_mae:.2f}")
        print(f"RMSE: {lstm_rmse:.2f}")
        print(f"MAPE: {lstm_mape:.2f}%")
        
        return lstm_model, lstm_predictions, y_test_actual, X_test_seq, target_scaler
    
    def create_ensemble(self, weights_options=None):
        """
        Create an ensemble model by combining XGBoost and LSTM predictions.
        
        Args:
            weights_options (list): List of weight options to try [XGBoost weight, LSTM weight]
            
        Returns:
            Tuple containing the best weights and ensemble predictions
        """
        print("\nCreating ensemble model...")
        
        if 'xgboost' not in self.predictions or 'lstm' not in self.predictions:
            raise ValueError("XGBoost and LSTM models must be implemented first.")
        
        if weights_options is None:
            weights_options = [
                [0.7, 0.3],  # More weight to XGBoost
                [0.5, 0.5],  # Equal weights
                [0.3, 0.7]   # More weight to LSTM
            ]
        
        # Get aligned predictions
        xgb_preds = self.predictions['xgboost']['predictions']
        lstm_preds = self.predictions['lstm']['predictions']
        
        # We need to make sure the test sets used by XGBoost and LSTM align
        # Use the smaller of the two test sets
        min_test_samples = min(len(xgb_preds), len(lstm_preds))
        
        # Get aligned predictions
        xgb_aligned = xgb_preds[-min_test_samples:]
        lstm_aligned = lstm_preds[-min_test_samples:]
        
        # Get aligned actuals (use LSTM actuals since they are properly inverse transformed)
        actual_aligned = self.predictions['lstm']['actuals'][-min_test_samples:]
        
        best_mae = float('inf')
        best_weights = None
        best_ensemble_preds = None
        best_metrics = {}
        
        for weights in weights_options:
            # Weighted average
            ensemble_preds = weights[0] * xgb_aligned + weights[1] * lstm_aligned.flatten()
            
            # Calculate metrics
            ensemble_mae = mean_absolute_error(actual_aligned, ensemble_preds)
            ensemble_mse = mean_squared_error(actual_aligned, ensemble_preds)
            ensemble_rmse = math.sqrt(ensemble_mse)
            ensemble_mape = mean_absolute_percentage_error(actual_aligned, ensemble_preds) * 100
            
            print(f"Ensemble with weights {weights}:")
            print(f"MAE: {ensemble_mae:.2f}")
            print(f"RMSE: {ensemble_rmse:.2f}")
            print(f"MAPE: {ensemble_mape:.2f}%")
            
            if ensemble_mae < best_mae:
                best_mae = ensemble_mae
                best_weights = weights
                best_ensemble_preds = ensemble_preds
                best_metrics = {
                    'mae': ensemble_mae,
                    'rmse': ensemble_rmse,
                    'mape': ensemble_mape
                }
        
        print(f"\nBest ensemble weights: {best_weights}")
        print("Final Ensemble Model Performance:")
        print(f"MAE: {best_metrics['mae']:.2f}")
        print(f"RMSE: {best_metrics['rmse']:.2f}")
        print(f"MAPE: {best_metrics['mape']:.2f}%")
        
        # Store ensemble results
        self.models['ensemble'] = {
            'weights': best_weights,
            'models': ['xgboost', 'lstm']
        }
        self.predictions['ensemble'] = {
            'predictions': best_ensemble_preds,
            'actuals': actual_aligned,
            'xgb_preds': xgb_aligned,
            'lstm_preds': lstm_aligned
        }
        self.metrics['ensemble'] = best_metrics
        
        return best_weights, best_ensemble_preds, actual_aligned, xgb_aligned, lstm_aligned
    
    def compare_models(self):
        """
        Compare all implemented models and visualize results.
        """
        print("\nComparing model performance...")
        
        if not self.metrics:
            raise ValueError("No models have been implemented yet.")
        
        # Create lists for comparison
        model_names = []
        mae_values = []
        rmse_values = []
        mape_values = []
        
        # Collect metrics from all models
        if 'naive' in self.metrics:
            model_names.append('Naive Forecast')
            mae_values.append(self.metrics['naive']['mae'])
            rmse_values.append(self.metrics['naive']['rmse'])
            mape_values.append(self.metrics['naive']['mape'])
        
        if 'xgboost' in self.metrics:
            model_names.append('XGBoost')
            mae_values.append(self.metrics['xgboost']['mae'])
            rmse_values.append(self.metrics['xgboost']['rmse'])
            mape_values.append(self.metrics['xgboost']['mape'])
        
        if 'lstm' in self.metrics:
            model_names.append('LSTM')
            mae_values.append(self.metrics['lstm']['mae'])
            rmse_values.append(self.metrics['lstm']['rmse'])
            mape_values.append(self.metrics['lstm']['mape'])
        
        if 'ensemble' in self.metrics:
            model_names.append('Ensemble')
            mae_values.append(self.metrics['ensemble']['mae'])
            rmse_values.append(self.metrics['ensemble']['rmse'])
            mape_values.append(self.metrics['ensemble']['mape'])
        
        # Create a DataFrame for the results
        results_df = pd.DataFrame({
            'Model': model_names,
            'MAE': mae_values,
            'RMSE': rmse_values,
            'MAPE (%)': mape_values
        })
        
        print(results_df)
        
        return results_df
    
    def save_models(self, output_dir='models'):
        """
        Save all models for future use.
        
        Args:
            output_dir (str): Directory to save models
        """
        print(f"\nSaving models to {output_dir}...")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save XGBoost model
        if 'xgboost' in self.models:
            joblib.dump(self.models['xgboost'], f'{output_dir}/xgboost_demand_forecasting.pkl')
            print("XGBoost model saved.")
        
        # Save LSTM model
        if 'lstm' in self.models:
            self.models['lstm']['model'].save(f'{output_dir}/lstm_demand_forecasting.h5')
            joblib.dump(self.models['lstm']['target_scaler'], f'{output_dir}/target_scaler.pkl')
            print("LSTM model and scaler saved.")
        
        # Save ensemble weights
        if 'ensemble' in self.models:
            with open(f'{output_dir}/ensemble_weights.json', 'w') as f:
                json.dump({
                    'weights': self.models['ensemble']['weights'],
                    'models': self.models['ensemble']['models']
                }, f)
            print("Ensemble weights saved.")
        
        print("All models saved successfully.")
    
    def visualize_results(self, sample_days=7):
        """
        Visualize forecasting results for all models.
        
        Args:
            sample_days (int): Number of days to visualize
        """
        print("\nVisualizing results...")
        
        # Set up the sample size and start point
        sample_size = sample_days * 24  # Convert days to hours
        
        # Plot predictions for each model
        if 'naive' in self.predictions:
            plt.figure(figsize=(14, 6))
            naive_preds = self.predictions['naive']
            sample_start = max(0, len(naive_preds) - sample_size)
            
            plt.plot(naive_preds.iloc[sample_start:].index, 
                     naive_preds.iloc[sample_start:].values, 
                     label='Naive Forecast', 
                     color='red', 
                     linestyle='--')
            plt.plot(naive_preds.iloc[sample_start:].index, 
                     self.y_test.iloc[sample_start:sample_start+len(naive_preds.iloc[sample_start:])].values, 
                     label='Actual Demand', 
                     color='blue')
            
            plt.title('Naive Forecast vs Actual Demand')
            plt.xlabel('Time')
            plt.ylabel('Electricity Demand')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        
        # If ensemble is available, plot all models together
        if 'ensemble' in self.predictions:
            plt.figure(figsize=(14, 6))
            ensemble_data = self.predictions['ensemble']
            sample_start = max(0, len(ensemble_data['actuals']) - sample_size)
            
            plt.plot(ensemble_data['actuals'][sample_start:].flatten(), 
                     label='Actual Demand', 
                     color='blue')
            plt.plot(ensemble_data['xgb_preds'][sample_start:], 
                     label='XGBoost', 
                     color='green', 
                     linestyle='--')
            plt.plot(ensemble_data['lstm_preds'][sample_start:].flatten(), 
                     label='LSTM', 
                     color='purple', 
                     linestyle='--')
            plt.plot(ensemble_data['predictions'][sample_start:], 
                     label='Ensemble', 
                     color='red')
            
            plt.title('Model Comparison: XGBoost vs LSTM vs Ensemble')
            plt.xlabel('Hours')
            plt.ylabel('Electricity Demand')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    
    def run_pipeline(self):
        """
        Run the complete forecasting pipeline.
        """
        print("Running complete electricity demand forecasting pipeline...")
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Preprocess data
        self.preprocess_data()
        
        # Step 3: Split data into train and test sets
        self.split_train_test()
        
        # Step 4: Implement naive forecast
        self.implement_naive_forecast()
        
        # Step 5: Implement XGBoost model
        self.implement_xgboost()
        
        # Step 6: Implement LSTM model
        self.implement_lstm()
        
        # Step 7: Create ensemble model
        self.create_ensemble()
        
        # Step 8: Compare models
        self.compare_models()
        
        # Step 9: Visualize results
        self.visualize_results()
        
        # Step 10: Save models
        self.save_models()
        
        print("Forecasting pipeline completed successfully!")


def main():
    """
    Main function to run the electricity demand forecasting system.
    """
    # Define the data path
    data_path = "./dataset/processed/samples/sample_10000_clean_merged_data.csv"
    
    # Create a forecaster instance
    forecaster = ElectricityDemandForecaster(data_path=data_path)
    
    # Run the complete pipeline
    forecaster.run_pipeline()


if __name__ == "__main__":
    main()
