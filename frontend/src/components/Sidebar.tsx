
import { CalendarIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Calendar } from "@/components/ui/calendar";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Switch } from "@/components/ui/switch";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";
import { format } from "date-fns";
import { cities } from "@/data/mockData";
import { ModelParams, SelectedData } from "@/types";
import {
  Sidebar as ShadcnSidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
  SidebarTrigger,
} from "@/components/ui/sidebar";

interface SidebarProps {
  isOpen: boolean;
  onToggle: () => void;
  selectedData: SelectedData;
  setSelectedData: React.Dispatch<React.SetStateAction<SelectedData>>;
  modelParams: ModelParams;
  setModelParams: React.Dispatch<React.SetStateAction<ModelParams>>;
  activeModel: 'kmeans' | 'hierarchical';
  setActiveModel: React.Dispatch<React.SetStateAction<'kmeans' | 'hierarchical'>>;
}

const Sidebar = ({
  isOpen,
  onToggle,
  selectedData,
  setSelectedData,
  modelParams,
  setModelParams,
  activeModel,
  setActiveModel,
}: SidebarProps) => {
  // Function to handle date selection ensuring dates are valid
  const handleDateChange = (field: 'from' | 'to', date: Date | undefined) => {
    if (!date) return;
    
    // Make a copy of the current date range
    const newDateRange = { ...selectedData.dateRange };
    
    // Update the selected field
    newDateRange[field] = date;
    
    // Ensure 'to' is not before 'from'
    if (field === 'from' && newDateRange.from > newDateRange.to) {
      newDateRange.to = new Date(newDateRange.from);
      newDateRange.to.setDate(newDateRange.to.getDate() + 7); // Default to 7 days later
    }
    
    // Ensure 'from' is not after 'to'
    if (field === 'to' && newDateRange.to < newDateRange.from) {
      newDateRange.from = new Date(newDateRange.to);
      newDateRange.from.setDate(newDateRange.from.getDate() - 7); // Default to 7 days earlier
    }
    
    // Update state with the new date range
    setSelectedData({
      ...selectedData,
      dateRange: newDateRange
    });
  };

  return (
    <ShadcnSidebar>
      <SidebarHeader className="px-6 py-3 flex items-center gap-2">
        <h2 className="text-xl font-semibold">Controls</h2>
      </SidebarHeader>

      <SidebarContent className="px-6 py-3">
        <div className="space-y-6">
          {/* City Selection */}
          <div className="space-y-2">
            <label className="text-sm font-medium">City</label>
            <Select
              value={selectedData.city}
              onValueChange={(city) => {
                setSelectedData({
                  ...selectedData,
                  city
                });
              }}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select city" />
              </SelectTrigger>
              <SelectContent>
                {cities.map((city) => (
                  <SelectItem key={city.id} value={city.id}>
                    {city.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Date Range */}
          <div className="space-y-2">
            <label className="text-sm font-medium">Date Range</label>
            <div className="flex flex-col gap-2">
              <Popover>
                <PopoverTrigger asChild>
                  <Button
                    variant="outline"
                    className={cn(
                      "justify-start text-left font-normal",
                      !selectedData.dateRange.from && "text-muted-foreground"
                    )}
                  >
                    <CalendarIcon className="mr-2 h-4 w-4" />
                    {selectedData.dateRange.from ? (
                      format(selectedData.dateRange.from, "PPP")
                    ) : (
                      <span>From date</span>
                    )}
                  </Button>
                </PopoverTrigger>
                <PopoverContent className="w-auto p-0" align="start">
                  <Calendar
                    mode="single"
                    selected={selectedData.dateRange.from}
                    onSelect={(date) => handleDateChange('from', date)}
                    initialFocus
                    className={cn("p-3 pointer-events-auto")}
                  />
                </PopoverContent>
              </Popover>

              <Popover>
                <PopoverTrigger asChild>
                  <Button
                    variant="outline"
                    className={cn(
                      "justify-start text-left font-normal",
                      !selectedData.dateRange.to && "text-muted-foreground"
                    )}
                  >
                    <CalendarIcon className="mr-2 h-4 w-4" />
                    {selectedData.dateRange.to ? (
                      format(selectedData.dateRange.to, "PPP")
                    ) : (
                      <span>To date</span>
                    )}
                  </Button>
                </PopoverTrigger>
                <PopoverContent className="w-auto p-0" align="start">
                  <Calendar
                    mode="single"
                    selected={selectedData.dateRange.to}
                    onSelect={(date) => handleDateChange('to', date)}
                    initialFocus
                    className={cn("p-3 pointer-events-auto")}
                  />
                </PopoverContent>
              </Popover>
            </div>
          </div>

          <Separator />

          {/* Model Parameters */}
          <div className="space-y-4">
            <h3 className="text-md font-medium">Model Parameters</h3>

            <div className="space-y-6">
              {/* Clusters (k) */}
              <div className="space-y-2">
                <div className="flex justify-between">
                  <label className="text-sm font-medium">Clusters (k)</label>
                  <span className="text-sm text-muted-foreground">{modelParams.k}</span>
                </div>
                <Slider
                  min={2}
                  max={10}
                  step={1}
                  value={[modelParams.k]}
                  onValueChange={(value) => {
                    setModelParams({
                      ...modelParams,
                      k: value[0]
                    });
                  }}
                />
              </div>

              {/* Lookback Window */}
              <div className="space-y-2">
                <div className="flex justify-between">
                  <label className="text-sm font-medium">Lookback Window</label>
                  <span className="text-sm text-muted-foreground">{modelParams.lookbackWindow} hours</span>
                </div>
                <Slider
                  min={24}
                  max={168}
                  step={24}
                  value={[modelParams.lookbackWindow]}
                  onValueChange={(value) => {
                    setModelParams({
                      ...modelParams,
                      lookbackWindow: value[0]
                    });
                  }}
                />
              </div>

              {/* Smoothing Factor */}
              <div className="space-y-2">
                <div className="flex justify-between">
                  <label className="text-sm font-medium">Smoothing Factor</label>
                  <span className="text-sm text-muted-foreground">{modelParams.smoothingFactor.toFixed(1)}</span>
                </div>
                <Slider
                  min={0.1}
                  max={1.0}
                  step={0.1}
                  value={[modelParams.smoothingFactor]}
                  onValueChange={(value) => {
                    setModelParams({
                      ...modelParams,
                      smoothingFactor: value[0]
                    });
                  }}
                />
              </div>
            </div>
          </div>

          <Separator />

          {/* Model Toggle */}
          <div className="space-y-2">
            <label className="text-sm font-medium">Model Algorithm</label>
            <div className="flex items-center justify-between">
              <span className={`text-sm ${activeModel === 'kmeans' ? 'font-medium' : 'text-muted-foreground'}`}>XGBoost</span>
              <Switch
                checked={activeModel === 'hierarchical'}
                onCheckedChange={(checked) => {
                  setActiveModel(checked ? 'hierarchical' : 'kmeans');
                }}
              />
              <span className={`text-sm ${activeModel === 'hierarchical' ? 'font-medium' : 'text-muted-foreground'}`}>Prophet</span>
            </div>
          </div>
        </div>
      </SidebarContent>

      <SidebarFooter className="px-6 py-3">
        <p className="text-xs text-muted-foreground">
          Connected to FastAPI backend at http://localhost:8000
        </p>
      </SidebarFooter>
    </ShadcnSidebar>
  );
};

export default Sidebar;
