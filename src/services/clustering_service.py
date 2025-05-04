"""
Clustering service for electricity demand data analysis.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class ClusteringService:
    """Service for clustering analysis of electricity demand data."""
    
    @staticmethod
    def perform_clustering(data, num_clusters, features):
        """
        Perform k-means clustering on the data.
        
        Args:
            data (pd.DataFrame): Input data
            num_clusters (int): Number of clusters to create
            features (list): Features to use for clustering
            
        Returns:
            dict: Clustering results
        """
        # Prepare data for clustering
        cluster_data = data[features].dropna()
        
        # Scale data for clustering
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_data)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_data)
        
        # Apply PCA for 2D visualization
        pca = PCA(n_components=2)
        pca_results = pca.fit_transform(scaled_data)
        
        # Prepare result
        clusters = []
        for i in range(num_clusters):
            cluster_indices = np.where(cluster_labels == i)[0]
            clusters.append({
                "cluster_id": int(i),
                "size": int(len(cluster_indices)),
                "center": kmeans.cluster_centers_[i].tolist(),
                "points": [
                    {
                        "x": float(pca_results[idx, 0]),
                        "y": float(pca_results[idx, 1]),
                        "demand": float(cluster_data.iloc[idx]['demand']),
                        "timestamp": data.iloc[cluster_data.index[idx]]['datetime'].strftime('%Y-%m-%d %H:%M:%S')
                    }
                    for idx in cluster_indices[:100]  # Limit to 100 points per cluster
                ]
            })
        
        # Calculate explained variance
        explained_variance = pca.explained_variance_ratio_.tolist()
        
        return {
            "clusters": clusters,
            "pca_components": {
                "x_label": f"PC1 ({explained_variance[0]:.2%} variance)",
                "y_label": f"PC2 ({explained_variance[1]:.2%} variance)"
            },
            "cluster_centers": kmeans.cluster_centers_.tolist(),
            "feature_importance": {feature: float(abs(pca.components_[0][i])) 
                                  for i, feature in enumerate(features)}
        }
