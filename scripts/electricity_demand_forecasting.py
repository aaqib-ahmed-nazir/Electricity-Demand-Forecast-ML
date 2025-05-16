import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import xgboost as xgb
import math
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import joblib
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import json
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

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
        print("\nPreprocessing data...")
        
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Create a copy of the dataframe to avoid SettingWithCopyWarning
        self.df = self.df.copy()
        
        # Convert timestamp to datetime
        self.df.loc[:, 'datetime'] = pd.to_datetime(self.df['timestamp'])
        
        # Extract temporal features
        self.df.loc[:, 'hour'] = self.df['datetime'].dt.hour
        self.df.loc[:, 'day'] = self.df['datetime'].dt.day
        self.df.loc[:, 'month'] = self.df['datetime'].dt.month
        self.df.loc[:, 'dayofweek'] = self.df['datetime'].dt.dayofweek
        self.df.loc[:, 'quarter'] = self.df['datetime'].dt.quarter
        self.df.loc[:, 'year'] = self.df['datetime'].dt.year
        self.df.loc[:, 'dayofyear'] = self.df['datetime'].dt.dayofyear
        self.df.loc[:, 'weekofyear'] = self.df['datetime'].dt.isocalendar().week
        
        # Create cyclical features for time variables
        self.df.loc[:, 'hour_sin'] = np.sin(2 * np.pi * self.df['hour']/24)
        self.df.loc[:, 'hour_cos'] = np.cos(2 * np.pi * self.df['hour']/24)
        self.df.loc[:, 'day_sin'] = np.sin(2 * np.pi * self.df['day']/31)
        self.df.loc[:, 'day_cos'] = np.cos(2 * np.pi * self.df['day']/31)
        self.df.loc[:, 'month_sin'] = np.sin(2 * np.pi * self.df['month']/12)
        self.df.loc[:, 'month_cos'] = np.cos(2 * np.pi * self.df['month']/12)
        self.df.loc[:, 'dayofweek_sin'] = np.sin(2 * np.pi * self.df['dayofweek']/7)
        self.df.loc[:, 'dayofweek_cos'] = np.cos(2 * np.pi * self.df['dayofweek']/7)
        
        # Add weekend indicator
        self.df.loc[:, 'is_weekend'] = (self.df['dayofweek'] >= 5).astype(int)
        
        # Add day part indicators (morning, afternoon, evening, night)
        self.df.loc[:, 'day_part_0'] = ((self.df['hour'] >= 5) & (self.df['hour'] < 12)).astype(int)  # Morning
        self.df.loc[:, 'day_part_1'] = ((self.df['hour'] >= 12) & (self.df['hour'] < 17)).astype(int)  # Afternoon
        self.df.loc[:, 'day_part_2'] = ((self.df['hour'] >= 17) & (self.df['hour'] < 22)).astype(int)  # Evening
        self.df.loc[:, 'day_part_3'] = ((self.df['hour'] >= 22) | (self.df['hour'] < 5)).astype(int)   # Night
        
        if 'is_holiday' in self.df.columns:
            self.df.loc[:, 'is_day_before_holiday'] = self.df['is_holiday'].shift(-24).fillna(0)
            self.df.loc[:, 'is_day_after_holiday'] = self.df['is_holiday'].shift(24).fillna(0)
        else:
            self.df.loc[:, 'is_holiday'] = 0
            self.df.loc[:, 'is_day_before_holiday'] = 0
            self.df.loc[:, 'is_day_after_holiday'] = 0
        
        weather_features = ['temperature', 'humidity', 'wind_speed', 'precipitation', 'cloud_cover']
        available_weather_features = [f for f in weather_features if f in self.df.columns]
        
        for feature in available_weather_features:
            if self.df[feature].isna().any():
                self.df.loc[:, feature] = self.df[feature].fillna(method='ffill')
                self.df.loc[:, feature] = self.df[feature].fillna(method='bfill')
                self.df.loc[:, feature] = self.df[feature].fillna(self.df[feature].median())
            
            scaler = MinMaxScaler()
            self.df.loc[:, f'{feature}_scaled'] = scaler.fit_transform(self.df[[feature]])
            
            self.df.loc[:, f'{feature}_squared'] = self.df[f'{feature}_scaled'] ** 2
            self.df.loc[:, f'{feature}_cubed'] = self.df[f'{feature}_scaled'] ** 3
            
            self.df.loc[:, f'{feature}_binned'] = pd.qcut(self.df[feature], 5, labels=False, duplicates='drop')
        
        for i, feat1 in enumerate(available_weather_features):
            for feat2 in available_weather_features[i+1:]:
                self.df.loc[:, f'{feat1}_{feat2}_interaction'] = self.df[f'{feat1}_scaled'] * self.df[f'{feat2}_scaled']
        
        self.df.loc[:, 'demand_lag_24h'] = self.df['demand'].shift(24)  # Previous day, same hour
        self.df.loc[:, 'demand_lag_48h'] = self.df['demand'].shift(48)  # Two days ago, same hour
        self.df.loc[:, 'demand_lag_168h'] = self.df['demand'].shift(168)  # Previous week, same hour
        
        self.df.loc[:, 'demand_rolling_mean_24h'] = self.df['demand'].shift(1).rolling(window=24).mean()
        self.df.loc[:, 'demand_rolling_std_24h'] = self.df['demand'].shift(1).rolling(window=24).std()
        self.df.loc[:, 'demand_rolling_min_24h'] = self.df['demand'].shift(1).rolling(window=24).min()
        self.df.loc[:, 'demand_rolling_max_24h'] = self.df['demand'].shift(1).rolling(window=24).max()
        
        self.df.loc[:, 'demand_rolling_mean_7d'] = self.df['demand'].shift(1).rolling(window=168).mean()
        self.df.loc[:, 'demand_rolling_std_7d'] = self.df['demand'].shift(1).rolling(window=168).std()
        
        self.df.loc[:, 'demand_ewm_24h'] = self.df['demand'].shift(1).ewm(span=24).mean()  # 1 day EWM
        self.df.loc[:, 'demand_ewm_7d'] = self.df['demand'].shift(1).ewm(span=168).mean()  # 7 day EWM
        
        self.df.loc[:, 'demand_diff_1d'] = self.df['demand'] - self.df['demand_lag_24h']
        self.df.loc[:, 'demand_diff_1w'] = self.df['demand'] - self.df['demand_lag_168h']
        
        self.df.loc[:, 'demand_pct_change_1d'] = self.df['demand'].pct_change(24).fillna(0)
        self.df.loc[:, 'demand_pct_change_1w'] = self.df['demand'].pct_change(168).fillna(0)
        
        for period in [24, 168, 8760]:  # Daily, weekly, yearly
            for harmonic in range(1, 4):  # Multiple harmonics
                self.df.loc[:, f'fourier_sin_{period}_{harmonic}'] = np.sin(2 * np.pi * harmonic * np.arange(len(self.df)) / period)
                self.df.loc[:, f'fourier_cos_{period}_{harmonic}'] = np.cos(2 * np.pi * harmonic * np.arange(len(self.df)) / period)
        
        # Handle anomalies by clipping extreme values
        if 'anomaly_z' in self.df.columns or 'anomaly_iqr' in self.df.columns or 'anomaly_iso' in self.df.columns:
            anomaly_cols = [col for col in ['anomaly_z', 'anomaly_iqr', 'anomaly_iso'] if col in self.df.columns]
            if anomaly_cols:
                self.df.loc[:, 'is_anomaly'] = self.df[anomaly_cols].any(axis=1).astype(int)
            else:
                self.df.loc[:, 'is_anomaly'] = 0
        else:
            # Create a simple anomaly detection based on Z-score
            z_scores = np.abs((self.df['demand'] - self.df['demand'].mean()) / self.df['demand'].std())
            self.df.loc[:, 'is_anomaly'] = (z_scores > 3).astype(int)
        
        self.feature_columns = [
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 
            'month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos',
            'is_weekend', 'day_part_0', 'day_part_1', 'day_part_2', 'day_part_3',
            'is_holiday', 'is_day_before_holiday', 'is_day_after_holiday'
        ]
        
        fourier_features = [col for col in self.df.columns if 'fourier_' in col]
        self.feature_columns.extend(fourier_features)
        
        if 'city' in self.df.columns:
            city_columns = [col for col in self.df.columns if col.startswith('city_')]
            self.feature_columns.extend(city_columns)
                              
        for feature in available_weather_features:
            weather_derived_features = [
                f'{feature}_scaled', f'{feature}_squared', f'{feature}_cubed', f'{feature}_binned'
            ]
            self.feature_columns.extend([f for f in weather_derived_features if f in self.df.columns])
        
        interaction_features = [col for col in self.df.columns if '_interaction' in col]
        self.feature_columns.extend(interaction_features)
        
        lag_features = [
            'demand_lag_24h', 'demand_lag_48h', 'demand_lag_168h',
            'demand_rolling_mean_24h', 'demand_rolling_max_24h', 
            'demand_rolling_min_24h', 'demand_rolling_std_24h',
            'demand_rolling_mean_7d', 'demand_rolling_std_7d',
            'demand_ewm_24h', 'demand_ewm_7d',
            'demand_diff_1d', 'demand_diff_1w',
            'demand_pct_change_1d', 'demand_pct_change_1w'
        ]
        
        self.feature_columns.extend(lag_features)
        
        if 'is_anomaly' in self.df.columns:
            self.feature_columns.append('is_anomaly')
        
        print(f"Features created: {len(self.feature_columns)} features")
        
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
        naive_r2 = r2_score(naive_test_actuals, naive_test_predictions)
        
        self.models['naive'] = 'Naive Forecast'
        self.predictions['naive'] = naive_test_predictions
        self.metrics['naive'] = {
            'mae': naive_mae,
            'rmse': naive_rmse,
            'mape': naive_mape,
            'r2': naive_r2
        }
        
        print("Naive Forecast Performance (24-hour lag):")
        print(f"MAE: {naive_mae:.2f}")
        print(f"RMSE: {naive_rmse:.2f}")
        print(f"MAPE: {naive_mape:.2f}%")
        print(f"R2: {naive_r2:.2f}")
        
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
            
        # Add more granular lags for recent hours
        for lag in range(1, 25):  # Last 24 hours
            lagged_features[f'demand_lag_{lag}'] = data[self.target_column].shift(lag)
        
        # Add rolling statistics for different windows
        for window in [24, 48, 168]:  # 1 day, 2 days, 1 week
            lagged_features[f'demand_rolling_mean_{window}'] = data[self.target_column].shift(1).rolling(window=window).mean()
            lagged_features[f'demand_rolling_std_{window}'] = data[self.target_column].shift(1).rolling(window=window).std()
            lagged_features[f'demand_rolling_min_{window}'] = data[self.target_column].shift(1).rolling(window=window).min()
            lagged_features[f'demand_rolling_max_{window}'] = data[self.target_column].shift(1).rolling(window=window).max()
        
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
        
        # Handle any NaN or infinite values
        X_train_ts = X_train_ts.replace([np.inf, -np.inf], np.nan)
        X_test_ts = X_test_ts.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with column medians (more robust than mean)
        for col in X_train_ts.columns:
            if X_train_ts[col].isna().any():
                col_median = X_train_ts[col].median()
                X_train_ts.loc[:, col] = X_train_ts[col].fillna(col_median)
                X_test_ts.loc[:, col] = X_test_ts[col].fillna(col_median)
        
        # Calculate correlation with target for feature selection
        correlations = {}
        for col in X_train_ts.columns:
            correlations[col] = abs(np.corrcoef(X_train_ts[col].values, y_train_ts.values)[0, 1])
        
        # Sort features by correlation
        sorted_correlations = {k: v for k, v in sorted(correlations.items(), key=lambda item: item[1], reverse=True)}
        
        # Select top features by correlation (top 70%)
        num_features_to_keep = int(len(sorted_correlations) * 0.7)
        top_correlated_features = list(sorted_correlations.keys())[:num_features_to_keep]
        
        print(f"Selected {len(top_correlated_features)} features by correlation out of {X_train_ts.shape[1]}")
        print(f"Top 10 correlated features: {list(sorted_correlations.keys())[:10]}")
        
        # Create new dataframes with selected features
        X_train_corr = X_train_ts[top_correlated_features].copy()
        X_test_corr = X_test_ts[top_correlated_features].copy()
        
        # Train a preliminary model to get feature importances
        prelim_model = xgb.XGBRegressor(
            n_estimators=100,  
            learning_rate=0.1, 
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        prelim_model.fit(X_train_corr, y_train_ts)
        
        # Get feature importance
        feature_importance = prelim_model.feature_importances_
        feature_names = X_train_corr.columns
        
        # Create a dataframe of feature importances
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        # Select top features by importance (top 80%)
        importance_threshold = feature_importance_df['Importance'].sum() * 0.8
        cumulative_importance = 0
        selected_features = []
        
        for _, row in feature_importance_df.iterrows():
            cumulative_importance += row['Importance']
            selected_features.append(row['Feature'])
            if cumulative_importance >= importance_threshold:
                break
        
        print(f"Selected {len(selected_features)} features by importance out of {len(top_correlated_features)}")
        print(f"Top 10 important features: {selected_features[:10]}")
        
        # Create final feature set
        X_train_selected = X_train_corr[selected_features].copy()
        X_test_selected = X_test_corr[selected_features].copy()
        
        # Hyperparameter tuning - use a more optimized set of parameters
        # These parameters are chosen to balance performance and training time
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 8,
            'learning_rate': 0.05,  # Increased for faster convergence
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'colsample_bylevel': 0.75,
            'min_child_weight': 5,
            'gamma': 0.05,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'n_jobs': -1,
            'seed': 42,
            'tree_method': 'hist',  
            'max_bin': 256          
        }
        
        print("Training XGBoost model...")
        
        # Convert data to DMatrix format for XGBoost
        dtrain = xgb.DMatrix(X_train_selected, label=y_train_ts)
        dvalid = xgb.DMatrix(X_test_selected, label=y_test_ts)
        
        # Set up evaluation list
        evals = [(dtrain, 'train'), (dvalid, 'valid')]
        
        # Train model with early stopping and fewer boosting rounds
        xgb_model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=2000,  # Reduced from 10000
            evals=evals,
            early_stopping_rounds=50,  # Reduced from 100
            verbose_eval=100
        )
        
        # Get feature importance
        feature_importance = xgb_model.get_score(importance_type='gain')
        importance_df = pd.DataFrame({
            'Feature': list(feature_importance.keys()),
            'Importance': list(feature_importance.values())
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 10 most important features:")
        print(importance_df.head(10))
        
        # Make predictions
        xgb_predictions = xgb_model.predict(dvalid)
        
        # Calculate error metrics
        xgb_mae = mean_absolute_error(y_test_ts, xgb_predictions)
        xgb_mse = mean_squared_error(y_test_ts, xgb_predictions)
        xgb_rmse = math.sqrt(xgb_mse)
        xgb_mape = mean_absolute_percentage_error(y_test_ts, xgb_predictions) * 100
        xgb_r2 = r2_score(y_test_ts, xgb_predictions)
        
        # Store model, predictions and metrics
        self.models['xgboost'] = xgb_model
        self.predictions['xgboost'] = {
            'predictions': xgb_predictions,
            'actuals': y_test_ts,
            'X_test': X_test_selected,
            'feature_importance': importance_df,
            'selected_features': selected_features
        }
        self.metrics['xgboost'] = {
            'mae': xgb_mae,
            'rmse': xgb_rmse,
            'mape': xgb_mape,
            'r2': xgb_r2
        }
        
        print("XGBoost Model Performance:")
        print(f"MAE: {xgb_mae:.2f}")
        print(f"RMSE: {xgb_rmse:.2f}")
        print(f"MAPE: {xgb_mape:.2f}%")
        print(f"R2: {xgb_r2:.2f}")
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        plt.barh(importance_df['Feature'].head(15), importance_df['Importance'].head(15))
        plt.title('XGBoost Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()
        
        return xgb_model, xgb_predictions, y_test_ts, X_test_selected
    
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
    
    def implement_lstm(self, seq_length=24, epochs=30, batch_size=64):
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
        
        lstm_train_data = self.train_df[self.feature_columns + [self.target_column]].copy()
        lstm_test_data = self.test_df[self.feature_columns + [self.target_column]].copy()
        
        lstm_train_data = lstm_train_data.replace([np.inf, -np.inf], np.nan)
        lstm_test_data = lstm_test_data.replace([np.inf, -np.inf], np.nan)
        
        for col in lstm_train_data.columns:
            if lstm_train_data[col].isna().any():
                col_median = lstm_train_data[col].median()
                lstm_train_data.loc[:, col] = lstm_train_data[col].fillna(col_median)
                lstm_test_data.loc[:, col] = lstm_test_data[col].fillna(col_median)
        
        # Calculate correlation with target for feature selection
        correlations = lstm_train_data.corr()[self.target_column].abs().sort_values(ascending=False)
        
        num_top_features = min(30, len(correlations) - 1)  
        top_features = correlations.index.tolist()[:num_top_features+1] 
        
        if self.target_column in top_features:
            top_features.remove(self.target_column)
            
        print(f"Selected {len(top_features)} features for LSTM model")
        print(f"Top 10 features by correlation: {top_features[:10]}")
        
        # Create new DataFrames with only selected features to avoid SettingWithCopyWarning
        lstm_train_data_selected = lstm_train_data[top_features + [self.target_column]].copy()
        lstm_test_data_selected = lstm_test_data[top_features + [self.target_column]].copy()
        
        # Standardize the target variable for better LSTM performance
        target_scaler = MinMaxScaler(feature_range=(0, 1))
        target_values = lstm_train_data_selected[[self.target_column]].values
        lstm_train_data_selected.loc[:, self.target_column] = target_scaler.fit_transform(target_values)
        
        target_test_values = lstm_test_data_selected[[self.target_column]].values
        lstm_test_data_selected.loc[:, self.target_column] = target_scaler.transform(target_test_values)
        
        # Standardize the features - create a new DataFrame to avoid warnings
        feature_cols = top_features
        
        # Create new DataFrames for scaled features
        lstm_train_scaled = lstm_train_data_selected.copy()
        lstm_test_scaled = lstm_test_data_selected.copy()
        
        # Create a scaler dictionary to store all scalers
        feature_scalers = {}
        
        for col in feature_cols:
            # Fit on training data
            col_scaler = StandardScaler()
            col_values = lstm_train_scaled[[col]].values
            scaled_values = col_scaler.fit_transform(col_values)
            
            # Apply scaling and clipping to training data - convert to float to avoid dtype warning
            lstm_train_scaled.loc[:, col] = np.clip(scaled_values, -3, 3).astype(float)
            
            # Apply same scaling to test data - convert to float to avoid dtype warning
            col_test_values = lstm_test_scaled[[col]].values
            scaled_test_values = col_scaler.transform(col_test_values)
            lstm_test_scaled.loc[:, col] = np.clip(scaled_test_values, -3, 3).astype(float)
            
            # Store the scaler
            feature_scalers[col] = col_scaler
        
        # Create sequences
        X_train_seq, y_train_seq = self.create_sequences(lstm_train_scaled, seq_length)
        X_test_seq, y_test_seq = self.create_sequences(lstm_test_scaled, seq_length)
        
        print(f"LSTM training sequences shape: {X_train_seq.shape}")
        print(f"LSTM testing sequences shape: {X_test_seq.shape}")
        
        # Input layer
        input_layer = Input(shape=(seq_length, X_train_seq.shape[2]))
        
        # First LSTM layer with batch normalization and dropout
        lstm1 = LSTM(units=64, activation='tanh', return_sequences=True, 
                    recurrent_dropout=0.1)(input_layer)
        batch_norm1 = BatchNormalization()(lstm1)
        dropout1 = Dropout(0.3)(batch_norm1)
        
        # Second LSTM layer
        lstm2 = LSTM(units=32, activation='tanh', return_sequences=False)(dropout1)
        batch_norm2 = BatchNormalization()(lstm2)
        dropout2 = Dropout(0.2)(batch_norm2)
        
        # Dense layers
        dense1 = Dense(units=16, activation='relu')(dropout2)
        dropout3 = Dropout(0.1)(dense1)
        
        # Output layer
        output_layer = Dense(units=1)(dropout3)
        
        # Create the model
        lstm_model = Model(inputs=input_layer, outputs=output_layer)
        
        # Compile with Adam optimizer and learning rate
        optimizer = Adam(learning_rate=0.001)
        lstm_model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        # Print model summary
        lstm_model.summary()
        
        # Define callbacks for better training
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        # Add learning rate reduction on plateau
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
        
        # Add model checkpoint to save best model
        model_checkpoint = ModelCheckpoint(
            filepath='models/lstm_best_model.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        
        # Train the model
        print("Training LSTM model...")
        lstm_history = lstm_model.fit(
            X_train_seq, y_train_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr, model_checkpoint],
            verbose=1
        )
        
        # Make predictions
        lstm_predictions = lstm_model.predict(X_test_seq)
        
        # Check for NaN values in predictions
        if np.isnan(lstm_predictions).any():
            print("Warning: NaN values detected in LSTM predictions. Using fallback approach.")
            # Use a simple fallback approach - predict mean of training data
            mean_target = np.mean(y_train_seq)
            lstm_predictions = np.full_like(lstm_predictions, mean_target)
        
        # Inverse transform predictions and actual values
        lstm_predictions = target_scaler.inverse_transform(lstm_predictions)
        y_test_actual = target_scaler.inverse_transform(y_test_seq.reshape(-1, 1))
        
        # Calculate error metrics
        lstm_mae = mean_absolute_error(y_test_actual, lstm_predictions)
        lstm_mse = mean_squared_error(y_test_actual, lstm_predictions)
        lstm_rmse = math.sqrt(lstm_mse)
        lstm_mape = mean_absolute_percentage_error(y_test_actual, lstm_predictions) * 100
        lstm_r2 = r2_score(y_test_actual, lstm_predictions)
        
        # Store model, predictions and metrics
        self.models['lstm'] = {
            'model': lstm_model,
            'target_scaler': target_scaler,
            'feature_scalers': feature_scalers,
            'selected_features': top_features
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
            'r2': lstm_r2,
            'history': lstm_history.history
        }
        
        print("LSTM Model Performance:")
        print(f"MAE: {lstm_mae:.2f}")
        print(f"RMSE: {lstm_rmse:.2f}")
        print(f"MAPE: {lstm_mape:.2f}%")
        print(f"R2: {lstm_r2:.2f}")
        
        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(lstm_history.history['loss'], label='Training Loss')
        plt.plot(lstm_history.history['val_loss'], label='Validation Loss')
        plt.title('LSTM Training History')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
        
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
                [0.8, 0.2],    # More weight to XGBoost
                [0.5, 0.5],    # Equal weights
                [0.2, 0.8]     # More weight to LSTM
            ]
        
        # Get aligned predictions
        xgb_preds = self.predictions['xgboost']['predictions']
        lstm_preds = self.predictions['lstm']['predictions'].flatten()
        
        # We need to make sure the test sets used by XGBoost and LSTM align
        # Use the smaller of the two test sets
        min_test_samples = min(len(xgb_preds), len(lstm_preds))
        
        # Get aligned predictions
        xgb_aligned = xgb_preds[-min_test_samples:]
        lstm_aligned = lstm_preds[-min_test_samples:]
        
        # Get aligned actuals (use LSTM actuals since they are properly inverse transformed)
        actual_aligned = self.predictions['lstm']['actuals'][-min_test_samples:].flatten()
        
        # Try a simplified stacking approach
        # Prepare features for the meta-model
        meta_features = pd.DataFrame({
            'xgb_pred': xgb_aligned,
            'lstm_pred': lstm_aligned
        })
        
        # Create a simple Ridge regression model for stacking
        from sklearn.linear_model import Ridge
        stacking_model = Ridge(alpha=0.5)
        
        # Train on all data
        stacking_model.fit(meta_features, actual_aligned)
        
        # Make predictions
        stacking_preds = stacking_model.predict(meta_features)
        
        # Calculate metrics for stacking approach
        stacking_mae = mean_absolute_error(actual_aligned, stacking_preds)
        stacking_mse = mean_squared_error(actual_aligned, stacking_preds)
        stacking_rmse = math.sqrt(stacking_mse)
        stacking_mape = mean_absolute_percentage_error(actual_aligned, stacking_preds) * 100
        stacking_r2 = r2_score(actual_aligned, stacking_preds)
        
        print("Final Stacking Ensemble Performance:")
        print(f"MAE: {stacking_mae:.2f}")
        print(f"RMSE: {stacking_rmse:.2f}")
        print(f"MAPE: {stacking_mape:.2f}%")
        print(f"R2: {stacking_r2:.2f}")
        
        # Store ensemble results
        self.models['ensemble'] = {
            'approach': 'stacking',
            'models': ['xgboost', 'lstm']
        }
        self.predictions['ensemble'] = {
            'predictions': stacking_preds,
            'actuals': actual_aligned,
            'xgb_preds': xgb_aligned,
            'lstm_preds': lstm_aligned
        }
        self.metrics['ensemble'] = {
            'mae': stacking_mae,
            'rmse': stacking_rmse,
            'mape': stacking_mape,
            'r2': stacking_r2
        }
        
        return 'stacking', stacking_preds, actual_aligned, xgb_aligned, lstm_aligned
    
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
        r2_values = []
        
        # Collect metrics from all models
        if 'naive' in self.metrics:
            model_names.append('Naive Forecast')
            mae_values.append(self.metrics['naive']['mae'])
            rmse_values.append(self.metrics['naive']['rmse'])
            mape_values.append(self.metrics['naive']['mape'])
            r2_values.append(self.metrics['naive']['r2'])
        
        if 'xgboost' in self.metrics:
            model_names.append('XGBoost')
            mae_values.append(self.metrics['xgboost']['mae'])
            rmse_values.append(self.metrics['xgboost']['rmse'])
            mape_values.append(self.metrics['xgboost']['mape'])
            r2_values.append(self.metrics['xgboost']['r2'])
        
        if 'lstm' in self.metrics:
            model_names.append('LSTM')
            mae_values.append(self.metrics['lstm']['mae'])
            rmse_values.append(self.metrics['lstm']['rmse'])
            mape_values.append(self.metrics['lstm']['mape'])
            r2_values.append(self.metrics['lstm']['r2'])
        
        if 'ensemble' in self.metrics:
            model_names.append('Ensemble')
            mae_values.append(self.metrics['ensemble']['mae'])
            rmse_values.append(self.metrics['ensemble']['rmse'])
            mape_values.append(self.metrics['ensemble']['mape'])
            r2_values.append(self.metrics['ensemble']['r2'])
        
        # Create a DataFrame for the results
        results_df = pd.DataFrame({
            'Model': model_names,
            'MAE': mae_values,
            'RMSE': rmse_values,
            'MAPE (%)': mape_values,
            'R2': r2_values
        })
        
        # Store the comparison dataframe as an attribute for later use
        self.comparison_df = results_df
        
        print(results_df)
        
        # Create a bar chart to visualize the comparison
        plt.figure(figsize=(14, 10))
        
        # Plot MAE
        plt.subplot(2, 2, 1)
        sns.barplot(x='Model', y='MAE', data=results_df)
        plt.title('Mean Absolute Error (MAE)')
        plt.xlabel('Model')
        plt.ylabel('MAE')
        plt.xticks(rotation=45)
        
        # Plot RMSE
        plt.subplot(2, 2, 2)
        sns.barplot(x='Model', y='RMSE', data=results_df)
        plt.title('Root Mean Squared Error (RMSE)')
        plt.xlabel('Model')
        plt.ylabel('RMSE')
        plt.xticks(rotation=45)
        
        # Plot MAPE
        plt.subplot(2, 2, 3)
        sns.barplot(x='Model', y='MAPE (%)', data=results_df)
        plt.title('Mean Absolute Percentage Error (MAPE)')
        plt.xlabel('Model')
        plt.ylabel('MAPE (%)')
        plt.xticks(rotation=45)
        
        # Plot R2
        plt.subplot(2, 2, 4)
        sns.barplot(x='Model', y='R2', data=results_df)
        plt.title('R-squared (R²)')
        plt.xlabel('Model')
        plt.ylabel('R²')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
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
            self.models['lstm']['model'].save(f'{output_dir}/lstm_demand_forecasting.keras')
            joblib.dump(self.models['lstm']['target_scaler'], f'{output_dir}/target_scaler.pkl')
            print("LSTM model and scaler saved.")
        
        # Save ensemble weights
        if 'ensemble' in self.models:
            with open(f'{output_dir}/ensemble_weights.json', 'w') as f:
                json.dump({
                    'approach': self.models['ensemble']['approach'],
                    'models': self.models['ensemble']['models']
                }, f)
            print("Ensemble weights saved.")
            
        # Save model metrics and results
        self.save_results(output_dir)
        
        print("All models saved successfully.")
    
    def save_results(self, output_dir='results'):
        """
        Save model metrics, logs, and comparison results to JSON and CSV files.
        
        Args:
            output_dir (str): Directory to save results
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics to JSON
        metrics_dict = {}
        for model_name, metrics in self.metrics.items():
            # Convert numpy values to Python native types for JSON serialization
            model_metrics = {}
            for metric_name, metric_value in metrics.items():
                if metric_name != 'history':  # Skip history which might contain non-serializable objects
                    if isinstance(metric_value, np.ndarray):
                        model_metrics[metric_name] = metric_value.tolist()
                    elif isinstance(metric_value, np.floating):
                        model_metrics[metric_name] = float(metric_value)
                    else:
                        model_metrics[metric_name] = metric_value
            metrics_dict[model_name] = model_metrics
        
        with open(f'{output_dir}/model_metrics.json', 'w') as f:
            json.dump(metrics_dict, f, indent=4)
        
        # Save comparison results to CSV
        if hasattr(self, 'comparison_df'):
            self.comparison_df.to_csv(f'{output_dir}/model_comparison.csv', index=False)
        
        # Create a detailed log of the training process
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_data = {
            'timestamp': timestamp,
            'dataset': self.data_path,
            'dataset_size': len(self.df) if self.df is not None else 0,
            'train_size': len(self.train_df) if self.train_df is not None else 0,
            'test_size': len(self.test_df) if self.test_df is not None else 0,
            'features': self.feature_columns,
            'metrics': metrics_dict,
            'best_model': self.get_best_model()
        }
        
        with open(f'{output_dir}/training_log_{timestamp}.json', 'w') as f:
            json.dump(log_data, f, indent=4)
        
        print(f"Results and logs saved to {output_dir}")
    
    def get_best_model(self):
        """
        Determine the best model based on MAE metric.
        
        Returns:
            dict: Information about the best model
        """
        if not self.metrics:
            return None
        
        best_model = None
        best_mae = float('inf')
        
        for model_name, metrics in self.metrics.items():
            if 'mae' in metrics and metrics['mae'] < best_mae:
                best_mae = metrics['mae']
                best_model = model_name
        
        return {
            'name': best_model,
            'metrics': self.metrics.get(best_model, {})
        }
    
    def visualize_results(self, sample_days=7):
        """
        Visualize forecasting results for all models.
        
        Args:
            sample_days (int): Number of days to visualize
        """
        print("\nVisualizing results...")
        
        # Set up the sample size and start point
        sample_size = sample_days * 24  
        
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
            plt.xlabel('Time Steps')
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
        
        self.load_data()
        
        self.preprocess_data()
        
        self.split_train_test()
        
        self.implement_naive_forecast()
        
        self.implement_xgboost()
        
        self.implement_lstm()
        
        self.create_ensemble()
        
        self.compare_models()
        
        self.save_models()
        
        print("Forecasting pipeline completed successfully!")


def main():
    """
    Main function to run the electricity demand forecasting system.
    """
    data_path = "./dataset/processed/samples/sample_10000_clean_merged_data.csv"
    
    forecaster = ElectricityDemandForecaster(data_path=data_path)
    
    forecaster.run_pipeline()


if __name__ == "__main__":
    main()
