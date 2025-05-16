import os
from fastapi import APIRouter, HTTPException
import traceback

from src.models.schemas import ClusteringRequest
from src.services.clustering_service import ClusteringService
from src.utils.data_processor import load_dataset, preprocess_data

# Create router
router = APIRouter(prefix="/cluster", tags=["Clustering"])

# Define constants
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "dataset/processed/samples")

@router.post("") 
async def cluster_data(request: ClusteringRequest):
    """
    Perform clustering analysis on electricity demand data.
    """
    try:
        # Load dataset
        data_path = os.path.join(DATA_DIR, "sample_10000_clean_merged_data.csv")
        df = load_dataset(data_path)
        
        if df is None:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        processed_df, _ = preprocess_data(df)
        
        if processed_df.empty:
            raise HTTPException(status_code=400, detail="No data available for processing")
        
        if request.features:
            cluster_features = [f for f in request.features if f in processed_df.columns]
            if not cluster_features:
                raise HTTPException(status_code=400, detail="None of the specified features exist in the dataset")
        else:
            cluster_features = ['demand', 'temperature_scaled', 'hour_sin', 'hour_cos']
            missing_features = [f for f in cluster_features if f not in processed_df.columns]
            if missing_features:
                cluster_features = [f for f in cluster_features if f in processed_df.columns]
                if 'demand' not in processed_df.columns:
                    raise HTTPException(status_code=400, detail="Required feature 'demand' is missing from the dataset")
                if len(cluster_features) < 3:
                    for feature in ['day_sin', 'day_cos', 'month_sin', 'month_cos', 'is_weekend']:
                        if feature in processed_df.columns and feature not in cluster_features:
                            cluster_features.append(feature)
                        if len(cluster_features) >= 4:
                            break
        
        # Perform clustering
        clustering_result = ClusteringService.perform_clustering(
            processed_df, 
            request.num_clusters, 
            cluster_features
        )
        
        return clustering_result
        
    except Exception as e:
        print(f"Clustering error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Clustering error: {str(e)}")
