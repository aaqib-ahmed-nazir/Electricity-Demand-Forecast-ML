
import { useState, useEffect } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import DocumentationSection from "./DocumentationSection";
import ChatSection from "./ChatSection";
import VisualizationSection from "./VisualizationSection";
import { ModelParams, SelectedData, ClusterPoint, ForecastDataPoint } from "@/types";
import { fetchClusterData, fetchForecastData } from "@/api/apiService";
import { useToast } from "@/hooks/use-toast";

interface MainDashboardProps {
  selectedData: SelectedData;
  modelParams: ModelParams;
  activeModel: 'kmeans' | 'hierarchical';
}

const MainDashboard = ({ 
  selectedData, 
  modelParams, 
  activeModel 
}: MainDashboardProps) => {
  const [activeTab, setActiveTab] = useState<string>("visualizations");
  const [clusterData, setClusterData] = useState<ClusterPoint[]>([]);
  const [forecastData, setForecastData] = useState<ForecastDataPoint[]>([]);
  const [pcaLabels, setPcaLabels] = useState<{ x: string; y: string } | undefined>(undefined);
  const [loading, setLoading] = useState<{clusters: boolean; forecast: boolean}>({
    clusters: false,
    forecast: false
  });
  const [error, setError] = useState<{clusters?: string; forecast?: string}>({});

  const { toast } = useToast();

  // Fetch cluster data when params change
  useEffect(() => {
    const getClusterData = async () => {
      setLoading(prev => ({ ...prev, clusters: true }));
      setError(prev => ({ ...prev, clusters: undefined }));
      
      try {
        const apiModel = activeModel === 'kmeans' ? 'xgboost' : 'prophet';
        const result = await fetchClusterData(selectedData.city, modelParams.k);
        
        setClusterData(result.clusterData);
        setPcaLabels(result.pcaLabels);
      } catch (err) {
        console.error("Error fetching cluster data:", err);
        setError(prev => ({ 
          ...prev, 
          clusters: "Failed to load cluster data. Please try again." 
        }));
        toast({
          title: "Error",
          description: "Failed to load cluster data. Please check your API connection.",
          variant: "destructive"
        });
      } finally {
        setLoading(prev => ({ ...prev, clusters: false }));
      }
    };
    
    getClusterData();
  }, [selectedData.city, modelParams.k, activeModel, toast]);

  // Fetch forecast data when params change
  useEffect(() => {
    const getForecastData = async () => {
      setLoading(prev => ({ ...prev, forecast: true }));
      setError(prev => ({ ...prev, forecast: undefined }));
      
      try {
        const apiModel = activeModel === 'kmeans' ? 'xgboost' : 'prophet';
        const result = await fetchForecastData(
          selectedData.city,
          selectedData.dateRange.from,
          selectedData.dateRange.to,
          modelParams.lookbackWindow,
          apiModel
        );
        
        setForecastData(result);
      } catch (err) {
        console.error("Error fetching forecast data:", err);
        setError(prev => ({ 
          ...prev, 
          forecast: "Failed to load forecast data. Please try again." 
        }));
        toast({
          title: "Error",
          description: "Failed to load forecast data. Please check your API connection.",
          variant: "destructive"
        });
      } finally {
        setLoading(prev => ({ ...prev, forecast: false }));
      }
    };
    
    getForecastData();
  }, [
    selectedData.city, 
    selectedData.dateRange.from, 
    selectedData.dateRange.to, 
    modelParams.lookbackWindow, 
    modelParams.smoothingFactor,
    activeModel,
    toast
  ]);

  return (
    <div className="flex-1 p-6 overflow-auto">
      <div className="mb-6">
        <h1 className="text-3xl font-bold">Temporal Clustering & Demand Forecasting</h1>
        <p className="text-muted-foreground mt-1">
          Interactive dashboard for analyzing demand patterns and predicting future trends
        </p>
      </div>

      <Tabs defaultValue="visualizations" value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="mb-4">
          <TabsTrigger value="visualizations">Visualizations</TabsTrigger>
          <TabsTrigger value="chat">Chat</TabsTrigger>
          <TabsTrigger value="documentation">Documentation</TabsTrigger>
        </TabsList>

        {/* Visualizations Tab */}
        <TabsContent value="visualizations">
          <VisualizationSection 
            clusterData={clusterData}
            forecastData={forecastData}
            loading={loading}
            error={error}
            pcaLabels={pcaLabels}
            modelParams={modelParams}
            activeModel={activeModel}
          />
        </TabsContent>

        {/* Chat Tab */}
        <TabsContent value="chat" className="h-[calc(100vh-220px)]">
          <ChatSection />
        </TabsContent>

        {/* Documentation Tab */}
        <TabsContent value="documentation">
          <DocumentationSection />
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default MainDashboard;
