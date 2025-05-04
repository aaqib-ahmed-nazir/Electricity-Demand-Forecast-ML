"""
Chat service for Gemini AI integration.
"""
import google.generativeai as genai
from fastapi import HTTPException

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
            
            # System context about our application
            system_prompt = """
            You are an AI assistant specialized in electricity demand forecasting. 
            You can explain concepts related to time series forecasting, machine learning models (XGBoost and LSTM),
            ensemble techniques, data preprocessing, and clustering analysis.
            
            The application uses:
            1. XGBoost model - a gradient boosting framework for tabular data
            2. LSTM (Long Short-Term Memory) model - a recurrent neural network for sequence data
            3. Ensemble model - a weighted combination of XGBoost and LSTM
            4. K-means clustering for usage pattern identification
            
            Tailor your responses to be helpful, informative, and technically accurate.
            """
            
            # Generate response
            chat = model.start_chat(history=[])
            response = chat.send_message(
                f"System: {system_prompt}\nUser: {query}"
            )
            
            return {
                "response": response.text,
                "model": "gemini-1.5-pro"
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")
