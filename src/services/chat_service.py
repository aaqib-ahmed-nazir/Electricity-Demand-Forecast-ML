"""
Chat service for Gemini AI integration.
"""
import google.generativeai as genai
from fastapi import HTTPException
import os
import pandas as pd

class ChatService:
    """Service for interacting with the Gemini generative AI model."""
    
    def __init__(self, api_key):
        """
        Initialize the chat service.
        
        Args:
            api_key (str): Gemini API key
        """
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.dataset_stats = self._load_dataset_stats()
    
    def _load_dataset_stats(self):
        """
        Load basic statistics about the dataset to provide more informed responses.
        
        Returns:
            dict: Statistics about the dataset
        """
        try:
            BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            DATA_PATH = os.path.join(BASE_DIR, "dataset/processed/samples/sample_10000_clean_merged_data.csv")
            
            if not os.path.exists(DATA_PATH):
                print("Warning: Sample dataset not found for chat context")
                return {}
                
            # Load dataset
            df = pd.read_csv(DATA_PATH)
            
            # Convert timestamp to datetime
            df['datetime'] = pd.to_datetime(df['timestamp'])
            
            # Calculate basic statistics
            stats = {
                "data_start_date": df['datetime'].min().strftime('%Y-%m-%d'),
                "data_end_date": df['datetime'].max().strftime('%Y-%m-%d'),
                "total_records": len(df),
                "cities": df['city'].unique().tolist(),
                "avg_demand": df['demand'].mean(),
                "min_demand": df['demand'].min(),
                "max_demand": df['demand'].max(),
                "avg_temperature": df['temperature'].mean(),
                "seasons": df['season'].unique().tolist(),
                "highest_demand_city": df.groupby('city')['demand'].mean().idxmax(),
                "lowest_demand_city": df.groupby('city')['demand'].mean().idxmin(),
                "hottest_city": df.groupby('city')['temperature'].mean().idxmax(),
                "coldest_city": df.groupby('city')['temperature'].mean().idxmin(),
            }
            
            return stats
        except Exception as e:
            print(f"Error loading dataset stats for chat: {e}")
            return {}
    
    def generate_response(self, query):
        """
        Generate a response from Gemini for the given query.
        
        Args:
            query (str): User query
            
        Returns:
            dict: Response from Gemini
        """
        try:
            # Set up the model
            model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')
            
            # Dataset statistics information
            stats_prompt = ""
            if self.dataset_stats:
                stats_prompt = f"""
                Dataset Information:
                - Date Range: {self.dataset_stats.get('data_start_date', 'Unknown')} to {self.dataset_stats.get('data_end_date', 'Unknown')}
                - Cities: {', '.join(self.dataset_stats.get('cities', ['Unknown']))}
                - Seasons: {', '.join(self.dataset_stats.get('seasons', ['Unknown']))}
                - Electricity Demand Range: {self.dataset_stats.get('min_demand', 'Unknown'):.2f} to {self.dataset_stats.get('max_demand', 'Unknown'):.2f} units
                - Average Demand: {self.dataset_stats.get('avg_demand', 'Unknown'):.2f} units
                - Highest Average Demand City: {self.dataset_stats.get('highest_demand_city', 'Unknown')}
                - Lowest Average Demand City: {self.dataset_stats.get('lowest_demand_city', 'Unknown')}
                - Hottest City on Average: {self.dataset_stats.get('hottest_city', 'Unknown')}
                - Coldest City on Average: {self.dataset_stats.get('coldest_city', 'Unknown')}
                """
            
            # System context about our application
            system_prompt = f"""
            You are an AI assistant specialized in electricity demand forecasting for our data mining project. 
            You can explain concepts related to time series forecasting, machine learning models, data processing,
            and help interpret results from our electricity demand forecasting system.
            
            The application uses:
            1. XGBoost model - A gradient boosting framework that delivers excellent performance for tabular data
            2. Ensemble model - A weighted combination of models for improved predictions
            3. K-means clustering for identifying electricity usage patterns
            
            Features in our dataset include:
            - timestamp: Date and time of the measurement
            - city: The city where electricity demand was recorded
            - demand: Electricity demand in units
            - temperature: Temperature in Fahrenheit
            - humidity: Relative humidity (0-1)
            - windSpeed: Wind speed in mph
            - pressure: Atmospheric pressure
            - precipIntensity: Precipitation intensity
            - precipProbability: Probability of precipitation (0-1)
            - hour, dayofweek, month, season: Temporal features
            - anomaly columns: Indicate unusual demand patterns
            
            {stats_prompt}
            
            Key limitations in our current system:
            1. Predictions are limited to the data range of {self.dataset_stats.get('data_start_date', '2018-07-01')} to {self.dataset_stats.get('data_end_date', '2020-05-31')}
            2. Models provide point forecasts (single values) rather than probabilistic forecasts
            3. Predictions work best for cities included in our training data
            
            Tailor your responses to be helpful, informative, and technically accurate. If the user asks about data outside our range, explain the limitations politely.
            """
            
            # Generate response
            chat = model.start_chat(history=[])
            response = chat.send_message(
                f"System: {system_prompt}\nUser: {query}"
            )
            
            return {
                "response": response.text,
                "model": "gemini-2.5-flash-preview"
            }
            
        except Exception as e:
            print(f"Chat error: {str(e)}")
            return {
                "response": "I'm sorry, I'm having trouble generating a response right now. Please try again later.",
                "error": str(e)
            }
