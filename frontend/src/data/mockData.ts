
import { City, ClusterPoint, ForecastDataPoint, ModelParams } from "@/types";

export const cities: City[] = [
  { id: "nyc", name: "New York City" },
  { id: "la", name: "Los Angeles" },
  { id: "dallas", name: "Dallas" },
  { id: "houston", name: "Houston" },
  { id: "philly", name: "Philadelphia" },
  { id: "phoenix", name: "Phoenix" },
  { id: "san_antonio", name: "San Antonio" },
  { id: "san_diego", name: "San Diego" },
  { id: "san_jose", name: "San Jose" },
  { id: "seattle", name: "Seattle" },
];

export const defaultParams: ModelParams = {
  k: 3,
  lookbackWindow: 7,
  smoothingFactor: 0.5,
};

// Generate random cluster points based on params.k
export const generateClusterData = (
  k: number,
  model: "kmeans" | "hierarchical"
): ClusterPoint[] => {
  const points: ClusterPoint[] = [];
  const pointsPerCluster = 20;
  
  for (let cluster = 0; cluster < k; cluster++) {
    const centerX = Math.random() * 8 - 4; // Center between -4 and 4
    const centerY = Math.random() * 8 - 4;
    
    const spread = model === "kmeans" ? 0.8 : 1.2; // Hierarchical more spread out
    
    for (let i = 0; i < pointsPerCluster; i++) {
      points.push({
        id: cluster * pointsPerCluster + i,
        x: centerX + (Math.random() - 0.5) * spread,
        y: centerY + (Math.random() - 0.5) * spread,
        cluster,
      });
    }
  }
  
  return points;
};

// Generate forecast data with date range and model params
export const generateForecastData = (
  from: Date,
  to: Date,
  model: "kmeans" | "hierarchical",
  lookbackWindow: number,
  smoothingFactor: number
): ForecastDataPoint[] => {
  const data: ForecastDataPoint[] = [];
  const days = Math.round((to.getTime() - from.getTime()) / (1000 * 60 * 60 * 24));
  
  let actual = 100 + Math.random() * 50;
  let trend = 0.1;
  let seasonality = model === "kmeans" ? 14 : 7; // Different seasonality for different models
  
  for (let i = 0; i <= days; i++) {
    const date = new Date(from.getTime() + i * (1000 * 60 * 60 * 24));
    
    // Add trend and seasonality components
    actual += trend + Math.sin(i / seasonality * Math.PI) * 10;
    
    // Add some noise
    actual += (Math.random() - 0.5) * 15;
    
    // Ensure value doesn't go below 50
    actual = Math.max(50, actual);
    
    // Calculate predicted with a lag based on lookback window and some error
    let predicted = actual;
    if (i > lookbackWindow) {
      const lag = data[i - lookbackWindow].actual;
      const error = (Math.random() - 0.5) * (10 / smoothingFactor);
      predicted = lag + trend * lookbackWindow + error;
    }
    
    data.push({
      date: date.toISOString().split('T')[0],
      actual: Math.round(actual),
      predicted: Math.round(predicted),
    });
  }
  
  return data;
};

// Documentation content
export const documentationContent = {
  usage: `
    Using the Dashboard
    
    1. Select a city from the dropdown in the sidebar
    2. Choose a date range to analyze
    3. Adjust model parameters to see how they affect clustering and forecasting
    4. Toggle between models using the checkbox to compare different approaches
    
    The dashboard will automatically update the visualizations based on your selections.
  `,
  approach: `
    Clustering and Forecasting Approach
    
    Temporal Clustering groups similar time periods based on demand patterns. This helps identify:
    - Weekly or seasonal patterns
    - Holiday effects
    - Anomalous periods
    
    Demand Forecasting uses historical data and the identified clusters to predict future demand. Our approach:
    1. Splits data into temporal clusters
    2. Trains separate forecasting models for each cluster
    3. Combines predictions for improved accuracy
  `,
  technical: `
    Technical Details
    
    Data Sources:
    - Historical demand data from city-specific databases
    - Weather and event data for contextual features
    
    Algorithms:
    - Clustering: K-means and Hierarchical clustering
    - Feature extraction: PCA for dimensionality reduction
    - Forecasting: SARIMA, Prophet, and custom sequence models
    
    Metrics:
    - Clustering: Silhouette score, Davies-Bouldin index
    - Forecasting: MAPE, MAE, RMSE
  `,
};
