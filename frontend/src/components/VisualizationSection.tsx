
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import ClusterVisualization from "./ClusterVisualization";
import ForecastChart from "./ForecastChart";
import { ClusterPoint, ForecastDataPoint } from "@/types";
import ModelPerformanceCards from "./ModelPerformanceCards";

interface VisualizationSectionProps {
  clusterData: ClusterPoint[];
  forecastData: ForecastDataPoint[];
  loading: {
    clusters: boolean;
    forecast: boolean;
  };
  error: {
    clusters?: string;
    forecast?: string;
  };
  pcaLabels?: { x: string; y: string };
  modelParams: {
    k: number;
    lookbackWindow: number;
    smoothingFactor: number;
  };
  activeModel: 'kmeans' | 'hierarchical';
}

const VisualizationSection = ({
  clusterData,
  forecastData,
  loading,
  error,
  pcaLabels,
  modelParams,
  activeModel
}: VisualizationSectionProps) => {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Cluster Visualization</CardTitle>
            <CardDescription>
              {activeModel === 'kmeans' ? 'K-means' : 'Hierarchical'} clustering with {modelParams.k} clusters
            </CardDescription>
          </CardHeader>
          <CardContent className="h-96 relative">
            {loading.clusters ? (
              <div className="absolute inset-0 flex items-center justify-center bg-background/50">
                <p className="text-lg font-medium">Loading clusters...</p>
              </div>
            ) : error.clusters ? (
              <div className="absolute inset-0 flex items-center justify-center">
                <p className="text-red-500">{error.clusters}</p>
              </div>
            ) : (
              <ClusterVisualization data={clusterData} k={modelParams.k} pcaLabels={pcaLabels} />
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Demand Forecast</CardTitle>
            <CardDescription>
              Actual vs. predicted demand with {modelParams.lookbackWindow}-day lookback window
            </CardDescription>
          </CardHeader>
          <CardContent className="h-96 relative">
            {loading.forecast ? (
              <div className="absolute inset-0 flex items-center justify-center bg-background/50">
                <p className="text-lg font-medium">Loading forecast...</p>
              </div>
            ) : error.forecast ? (
              <div className="absolute inset-0 flex items-center justify-center">
                <p className="text-red-500">{error.forecast}</p>
              </div>
            ) : (
              <ForecastChart data={forecastData} />
            )}
          </CardContent>
        </Card>
      </div>

      <ModelPerformanceCards 
        activeModel={activeModel} 
        loading={loading} 
        modelParams={modelParams} 
      />
    </div>
  );
};

export default VisualizationSection;
