
import { useState } from 'react';
import MainDashboard from '@/components/MainDashboard';
import Sidebar from '@/components/Sidebar';
import { SidebarProvider } from '@/components/ui/sidebar';
import { cities, defaultParams } from '@/data/mockData';
import { ModelParams, SelectedData } from '@/types';
import { ThemeToggle } from '@/components/ThemeToggle';

const Index = () => {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [selectedData, setSelectedData] = useState<SelectedData>({
    city: cities[0].id,
    dateRange: {
      from: new Date(2019, 0, 1), // January 1st, 2019
      to: new Date(2019, 1, 1),   // February 1st, 2019
    },
  });
  const [modelParams, setModelParams] = useState<ModelParams>(defaultParams);
  const [activeModel, setActiveModel] = useState<'kmeans' | 'hierarchical'>('kmeans');

  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };

  return (
    <SidebarProvider>
      <div className="min-h-screen flex w-full">
        <Sidebar 
          isOpen={sidebarOpen} 
          onToggle={toggleSidebar}
          selectedData={selectedData}
          setSelectedData={setSelectedData}
          modelParams={modelParams}
          setModelParams={setModelParams}
          activeModel={activeModel}
          setActiveModel={setActiveModel}
        />
        <div className="flex-1 flex flex-col">
          <div className="flex justify-end p-2">
            <ThemeToggle />
          </div>
          <MainDashboard 
            selectedData={selectedData}
            modelParams={modelParams}
            activeModel={activeModel}
          />
        </div>
      </div>
    </SidebarProvider>
  );
};

export default Index;
