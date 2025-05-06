
import { ClusterPoint, ForecastDataPoint } from "@/types";

const API_BASE_URL = "http://localhost:8000";

// Interface for API responses
interface ClusterResponse {
  clusters: {
    cluster_id: number;
    size: number;
    center: number[];
    points: {
      x: number;
      y: number;
      demand: number;
      timestamp: string;
    }[];
  }[];
  pca_components: {
    x_label: string;
    y_label: string;
  };
  cluster_centers: number[][];
  feature_importance: Record<string, number>;
}

interface ForecastResponse {
  model: string;
  forecast: number[];
  actual: number[];
  timestamps: string[];
  confidence_bounds: {
    lower: number[];
    upper: number[];
  };
}

export async function fetchClusterData(
  city: string,
  numClusters: number
): Promise<{ clusterData: ClusterPoint[]; pcaLabels: { x: string; y: string } }> {
  try {
    const response = await fetch(`${API_BASE_URL}/cluster`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        city: city,
        num_clusters: numClusters,
        features: ["demand", "temperature_scaled", "hour_sin", "hour_cos"]
      }),
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    const data: ClusterResponse = await response.json();
    
    // Transform the API response into our application's expected format
    const clusterPoints: ClusterPoint[] = [];
    
    data.clusters.forEach(clusterGroup => {
      clusterGroup.points.forEach((point, index) => {
        clusterPoints.push({
          id: index,
          x: point.x,
          y: point.y,
          cluster: clusterGroup.cluster_id,
        });
      });
    });
    
    return {
      clusterData: clusterPoints,
      pcaLabels: {
        x: data.pca_components.x_label,
        y: data.pca_components.y_label
      }
    };
  } catch (error) {
    console.error("Failed to fetch cluster data:", error);
    throw error;
  }
}

export async function fetchForecastData(
  city: string,
  startDate: Date,
  endDate: Date,
  lookbackWindow: number,
  model: 'xgboost' | 'prophet' = 'xgboost'
): Promise<ForecastDataPoint[]> {
  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        city: city,
        start_date: formatDate(startDate),
        end_date: formatDate(endDate),
        look_back_window: lookbackWindow,
        model: model
      }),
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    const data: ForecastResponse = await response.json();
    
    // Transform the API response into our application's expected format
    const forecastData: ForecastDataPoint[] = data.timestamps.map((timestamp, index) => ({
      date: timestamp,
      actual: data.actual[index],
      predicted: data.forecast[index],
      lower: data.confidence_bounds.lower[index],
      upper: data.confidence_bounds.upper[index]
    }));
    
    return forecastData;
  } catch (error) {
    console.error("Failed to fetch forecast data:", error);
    throw error;
  }
}

// Helper function to format dates as YYYY-MM-DD
function formatDate(date: Date): string {
  return date.toISOString().split('T')[0];
}
