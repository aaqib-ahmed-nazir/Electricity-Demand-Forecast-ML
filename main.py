import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.routes import prediction_routes, clustering_routes, chat_routes

app = FastAPI(
    title="Electricity Demand Forecasting API",
    description="API for predicting electricity demand using machine learning models",
    version="1.0.0",
    redirect_slashes=False,
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add routes
app.include_router(prediction_routes.router)
app.include_router(clustering_routes.router)
app.include_router(chat_routes.router)

@app.get("/")
def read_root():
    """Root endpoint providing application information."""
    return {
        "message": "Electricity Demand Forecasting API", 
        "status": "online",
        "endpoints": {
            "prediction": "/predict",
            "clustering": "/cluster",
            "chatbot": "/chat"
        },
        "documentation": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    # For local development
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
