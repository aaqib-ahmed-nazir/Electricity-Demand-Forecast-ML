
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

interface ModelPerformanceCardsProps {
  activeModel: 'kmeans' | 'hierarchical';
  loading: {
    clusters: boolean;
    forecast: boolean;
  };
  modelParams: {
    k: number;
    lookbackWindow: number;
    smoothingFactor: number;
  };
}

const ModelPerformanceCards = ({ 
  activeModel, 
  loading,
  modelParams 
}: ModelPerformanceCardsProps) => {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Model Performance</CardTitle>
        <CardDescription>
          Key metrics for the {activeModel === 'kmeans' ? 'K-means' : 'Hierarchical'} clustering model
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-secondary/50 p-4 rounded-md">
            <h3 className="font-medium mb-1">Silhouette Score</h3>
            <p className="text-2xl font-bold">
              {loading.clusters ? "..." : (0.65 + Math.random() * 0.3 - (modelParams.k > 5 ? 0.2 : 0)).toFixed(2)}
            </p>
            <p className="text-xs text-muted-foreground mt-1">
              Measure of how similar an object is to its own cluster compared to other clusters
            </p>
          </div>
          <div className="bg-secondary/50 p-4 rounded-md">
            <h3 className="font-medium mb-1">Mean Absolute Error</h3>
            <p className="text-2xl font-bold">
              {loading.forecast ? "..." : (5 + Math.random() * 8 - modelParams.lookbackWindow/10).toFixed(2)}
            </p>
            <p className="text-xs text-muted-foreground mt-1">
              Average absolute difference between actual and predicted values
            </p>
          </div>
          <div className="bg-secondary/50 p-4 rounded-md">
            <h3 className="font-medium mb-1">Forecast Accuracy</h3>
            <p className="text-2xl font-bold">
              {loading.forecast ? "..." : (90 - Math.random() * 15 + modelParams.smoothingFactor * 5).toFixed(1)}%
            </p>
            <p className="text-xs text-muted-foreground mt-1">
              Percentage of predictions within acceptable error margin
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default ModelPerformanceCards;
