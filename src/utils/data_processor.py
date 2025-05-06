"""
Data processing utilities for electricity demand forecasting.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import traceback

def load_dataset(data_path):
    """
    Load dataset from the given path.
    
    Args:
        data_path (str): Path to the dataset
        
    Returns:
        pd.DataFrame: Loaded DataFrame or None if error
    """
    try:
        df = pd.read_csv(data_path)
        # Check if the dataset has the required columns
        required_columns = ['timestamp', 'demand']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Dataset is missing required columns: {missing_columns}")
            return None
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print(traceback.format_exc())
        return None

def preprocess_data(df, start_date=None, end_date=None):
    """
    Preprocess data for modeling by extracting features and scaling.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        start_date (str, optional): Start date for filtering
        end_date (str, optional): End date for filtering
        
    Returns:
        tuple: Processed DataFrame and feature columns list
    """
    try:
        if df is None or df.empty:
            print("Error: Empty DataFrame provided for preprocessing")
            return pd.DataFrame(), []
            
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'])
        
        # Filter by date range if provided
        if start_date and end_date:
            try:
                start_date = pd.to_datetime(start_date)
                end_date = pd.to_datetime(end_date)
                df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]
                if df.empty:
                    print(f"Warning: No data available for the date range {start_date} to {end_date}")
                    return pd.DataFrame(), []
            except Exception as e:
                print(f"Error parsing date range: {e}")
                # Continue with full dataset if date parsing fails
        
        # Extract temporal features
        df['hour'] = df['datetime'].dt.hour
        df['dayofweek'] = df['datetime'].dt.dayofweek
        df['day'] = df['datetime'].dt.day
        df['month'] = df['datetime'].dt.month
        
        # Create cyclical features for time variables
        df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
        df['day_sin'] = np.sin(2 * np.pi * df['day']/31)
        df['day_cos'] = np.cos(2 * np.pi * df['day']/31)
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        
        # Create weekend indicator
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        
        # Add temperature binning feature
        if 'temperature' in df.columns:
            bins = [0, 40, 50, 60, 70, 80, 90, 100, 120]
            df['temperature_binned'] = pd.cut(df['temperature'], bins=bins, labels=False)
            df['temperature_binned'] = df['temperature_binned'].astype(float)
            df['temperature_binned'] = df['temperature_binned'].fillna(0)
        else:
            df['temperature_binned'] = 0
        
        # Add demand change features if demand column exists
        if 'demand' in df.columns:
            df = df.sort_values('datetime')
            df['demand_diff_1d'] = df['demand'].diff(24)
            df['demand_pct_change_1d'] = df['demand'].pct_change(24)
            df['demand_diff_1w'] = df['demand'].diff(168)
            df['demand_pct_change_1w'] = df['demand'].pct_change(168)
            # Fix for pandas FutureWarning - avoid inplace fillna on a copy
            for col in ['demand_diff_1d', 'demand_pct_change_1d', 'demand_diff_1w', 'demand_pct_change_1w']:
                df[col] = df[col].fillna(0)
        else:
            for col in ['demand_diff_1d', 'demand_pct_change_1d', 'demand_diff_1w', 'demand_pct_change_1w']:
                df[col] = 0
        
        # Scale weather features
        weather_features = ['temperature', 'humidity', 'windSpeed', 'pressure', 
                           'precipIntensity', 'precipProbability']
        
        scaler = MinMaxScaler()
        
        # Scale demand - this is needed for the XGBoost model
        if 'demand' in df.columns:
            if df['demand'].isna().any():
                print(f"Warning: NaN values found in demand, filling with mean")
                df['demand'] = df['demand'].fillna(df['demand'].mean())
            df['demand_scaled'] = scaler.fit_transform(df[['demand']])
        
        for feature in weather_features:
            if feature in df.columns:
                if df[feature].isna().any():
                    print(f"Warning: NaN values found in {feature}, filling with mean")
                    df[feature] = df[feature].fillna(df[feature].mean())
                df[f'{feature}_scaled'] = scaler.fit_transform(df[[feature]])
        
        # Define feature columns
        feature_columns = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 
                          'month_sin', 'month_cos', 'is_weekend',
                          'temperature_binned', 'demand_diff_1d', 'demand_pct_change_1d',
                          'demand_diff_1w', 'demand_pct_change_1w']
                          
        for feature in weather_features:
            if f'{feature}_scaled' in df.columns:
                feature_columns.append(f'{feature}_scaled')
        
        if 'demand_scaled' in df.columns:
            feature_columns.append('demand_scaled')
        
        return df, feature_columns
    except Exception as e:
        print(f"Error in preprocessing data: {e}")
        print(traceback.format_exc())
        return pd.DataFrame(), []

def create_features_targets(data, target_column='demand', lag_hours=24, window_size=7*24):
    """
    Create a dataset with lagged features for time series forecasting.
    
    Args:
        data (pd.DataFrame): DataFrame containing the data
        target_column (str): Target column name
        lag_hours (int): Number of hours to lag for forecasting
        window_size (int): Size of the window to consider for historical data
            
    Returns:
        tuple: Features DataFrame and target Series with NaN rows removed
    """
    try:
        if data is None or data.empty:
            print("Error: Empty DataFrame provided for feature creation")
            return pd.DataFrame(), pd.Series()
            
        if target_column not in data.columns:
            print(f"Error: Target column '{target_column}' not found in data")
            return pd.DataFrame(), pd.Series()
            
        data = data.copy()
        
        if data[target_column].isna().any():
            print(f"Warning: NaN values found in target column '{target_column}', filling with mean")
            data[target_column] = data[target_column].fillna(data[target_column].mean())
        
        available_hours = len(data)
        required_hours = lag_hours + window_size
        
        if available_hours < required_hours:
            print(f"Warning: Not enough data for requested window size. Have {available_hours} hours, need {required_hours} hours.")
            print(f"Adjusting window size from {window_size} to {max(24, available_hours - lag_hours - 24)}")
            window_size = max(24, available_hours - lag_hours - 24)
        
        lagged_features = pd.DataFrame()
        
        lag_step = 24
        if window_size < 72:
            lag_step = 12
        if window_size < 48:
            lag_step = 6
        
        for lag in range(lag_hours, lag_hours + window_size, lag_step):
            lagged_features[f'demand_lag_{lag}'] = data[target_column].shift(lag)
        
        for col in data.columns:
            if col.endswith('_scaled') or col in ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 
                                                 'month_sin', 'month_cos', 'is_weekend',
                                                 'temperature_binned', 'demand_diff_1d', 'demand_pct_change_1d',
                                                 'demand_diff_1w', 'demand_pct_change_1w']:
                lagged_features[col] = data[col]
        
        targets = data[target_column]
        
        valid_idx = ~lagged_features.isna().any(axis=1)
        
        if not valid_idx.any():
            print(f"Warning: No valid data after creating lagged features with lag_hours={lag_hours} and window_size={window_size}")
            return pd.DataFrame(), pd.Series()
            
        return lagged_features[valid_idx], targets[valid_idx]
    except Exception as e:
        print(f"Error creating features and targets: {e}")
        print(traceback.format_exc())
        return pd.DataFrame(), pd.Series()
