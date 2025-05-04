"""
Pydantic models for the API request and response schemas.
"""
from typing import List, Dict, Optional, Any
from pydantic import BaseModel

class PredictionRequest(BaseModel):
    """Schema for electricity demand prediction requests."""
    city: str
    start_date: str
    end_date: str
    look_back_window: Optional[int] = 7*24  
    model: Optional[str] = "xgboost" 

class ClusteringRequest(BaseModel):
    """Schema for clustering analysis requests."""
    city: str
    num_clusters: int = 3
    features: Optional[List[str]] = None

class ChatRequest(BaseModel):
    """Schema for Gemini chat requests."""
    query: str
