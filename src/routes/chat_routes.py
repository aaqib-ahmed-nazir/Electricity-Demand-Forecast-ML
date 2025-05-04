"""
API routes for Gemini chatbot integration.
"""
from fastapi import APIRouter, HTTPException
from src.models.schemas import ChatRequest
from src.services.chat_service import ChatService
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Create router
router = APIRouter(prefix="/chat", tags=["Chatbot"])

# Get Gemini API key from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")

# Initialize chat service
chat_service = ChatService(GEMINI_API_KEY)

@router.post("") 
async def chat_with_gemini(request: ChatRequest):
    """
    Generate a response from Gemini AI for the given query.
    """
    try:
        response = chat_service.generate_response(request.query)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")
