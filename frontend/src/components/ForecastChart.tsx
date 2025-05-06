import { useState, useRef, useEffect } from "react";
import { ForecastDataPoint } from "@/types";
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend,
  ResponsiveContainer,
  TooltipProps,
  Area,
  ComposedChart,
  ReferenceArea
} from "recharts";
import { format, parseISO } from "date-fns";
import { NameType, ValueType } from "recharts/types/component/DefaultTooltipContent";
import { Maximize2, Minimize2, ZoomIn, ZoomOut, RotateCcw } from "lucide-react";
import { useTheme } from "next-themes";

interface ForecastChartProps {
  data: ForecastDataPoint[];
}

const ForecastChart = ({ data }: ForecastChartProps) => {
  const [isFullscreen, setIsFullscreen] = useState(false);
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const { theme, resolvedTheme } = useTheme();
  
  // Zooming state
  const [refAreaLeft, setRefAreaLeft] = useState<string | undefined>();
  const [refAreaRight, setRefAreaRight] = useState<string | undefined>();
  const [zoomedData, setZoomedData] = useState(data);
  const [isReset, setIsReset] = useState(false);
  const [zoomLevel, setZoomLevel] = useState(100); // Zoom level in percentage

  // Update zoomed data when original data changes
  useEffect(() => {
    setZoomedData(data);
    setIsReset(true);
    setZoomLevel(100);
  }, [data]);

  // Toggle fullscreen mode
  const toggleFullscreen = () => {
    if (!isFullscreen) {
      if (chartContainerRef.current?.requestFullscreen) {
        chartContainerRef.current.requestFullscreen();
      }
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen();
      }
    }
  };

  // Listen for fullscreen change events
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };

    document.addEventListener('fullscreenchange', handleFullscreenChange);
    
    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
    };
  }, []);

  // Zooming functions
  const zoom = () => {
    if (refAreaLeft === refAreaRight || !refAreaRight) {
      setRefAreaLeft(undefined);
      setRefAreaRight(undefined);
      return;
    }

    // Ensure left is always less than right
    const indexLeft = data.findIndex(d => d.date === refAreaLeft);
    const indexRight = data.findIndex(d => d.date === refAreaRight);
    
    if (indexLeft !== -1 && indexRight !== -1) {
      let [leftIndex, rightIndex] = [indexLeft, indexRight].sort((a, b) => a - b);
      
      const zoomed = data.slice(leftIndex, rightIndex + 1);
      const newData = zoomed.length > 1 ? zoomed : data;
      setZoomedData(newData);
      
      // Calculate zoom level percentage
      const zoomPercent = Math.round((data.length / newData.length) * 100);
      setZoomLevel(zoomed.length > 1 ? zoomPercent : 100);
    }

    setRefAreaLeft(undefined);
    setRefAreaRight(undefined);
    setIsReset(false);
  };

  const resetZoom = () => {
    setZoomedData(data);
    setIsReset(true);
    setZoomLevel(100);
  };

  // Handle slider zoom control
  const handleZoomChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newZoomLevel = parseInt(e.target.value);
    setZoomLevel(newZoomLevel);
    
    // Calculate how many data points to show based on zoom level
    const dataLength = data.length;
    const visiblePoints = Math.max(Math.round(dataLength / (newZoomLevel / 100)), 2);
    
    // Get middle point as reference for zooming
    const midPoint = Math.floor(dataLength / 2);
    const halfVisible = Math.floor(visiblePoints / 2);
    
    // Calculate start and end indices
    let startIdx = midPoint - halfVisible;
    let endIdx = midPoint + halfVisible;
    
    // Adjust if out of bounds
    if (startIdx < 0) {
      startIdx = 0;
      endIdx = Math.min(visiblePoints, dataLength);
    }
    if (endIdx >= dataLength) {
      endIdx = dataLength - 1;
      startIdx = Math.max(0, endIdx - visiblePoints);
    }
    
    const newData = data.slice(startIdx, endIdx + 1);
    setZoomedData(newData.length > 0 ? newData : data);
    setIsReset(newZoomLevel === 100);
  };

  // Mouse event handlers for drawing the zoom area
  const handleMouseDown = (e: any) => {
    if (!e || !e.activeLabel) return;
    setRefAreaLeft(e.activeLabel);
  };

  const handleMouseMove = (e: any) => {
    if (!e || !e.activeLabel || !refAreaLeft) return;
    setRefAreaRight(e.activeLabel);
  };

  // Custom tooltip component
  const CustomTooltip = ({ 
    active, 
    payload, 
    label 
  }: TooltipProps<ValueType, NameType>) => {
    if (active && payload && payload.length) {
      // Try to parse the date, but handle if it's already formatted
      const dateStr = typeof label === 'string' 
        ? (label.includes('T') || label.includes('-') ? label : null)
        : null;
      
      const displayDate = dateStr 
        ? format(parseISO(dateStr), 'MMM d, yyyy HH:mm') 
        : label;
      
      return (
        <div className="bg-white dark:bg-gray-800 p-3 border border-gray-200 dark:border-gray-700 rounded-md shadow-md">
          <p className="font-medium text-sm text-gray-900 dark:text-gray-100">{displayDate}</p>
          <div className="flex flex-col gap-1 mt-2">
            {payload[0]?.value !== undefined && (
              <p className="text-xs flex items-center text-gray-700 dark:text-gray-300">
                <span className="h-2 w-2 bg-blue-600 rounded-full mr-2"></span>
                Actual: {payload[0].value}
              </p>
            )}
            {payload[1]?.value !== undefined && (
              <p className="text-xs flex items-center text-gray-700 dark:text-gray-300">
                <span className="h-2 w-2 bg-rose-600 rounded-full mr-2"></span>
                Predicted: {payload[1].value}
              </p>
            )}
            {payload[2]?.value !== undefined && payload[3]?.value !== undefined && (
              <p className="text-xs flex items-center text-gray-700 dark:text-gray-300">
                <span className="h-2 w-2 bg-gray-300 dark:bg-gray-500 rounded-full mr-2"></span>
                Confidence: {payload[2].value} - {payload[3].value}
              </p>
            )}
          </div>
        </div>
      );
    }
  
    return null;
  };

  // Format X-axis tick
  const formatXTick = (value: string) => {
    try {
      if (value.includes('T') || value.includes('-')) {
        const date = parseISO(value);
        return format(date, 'MMM d, HH:mm');
      }
      return value;
    } catch (e) {
      return value;
    }
  };

  // Determine the current theme
  const currentTheme = resolvedTheme || theme;
  const isDarkMode = currentTheme === 'dark';

  return (
    <div 
      className={`w-full h-full relative ${isFullscreen ? (isDarkMode ? 'bg-gray-900 text-gray-100' : 'bg-white text-gray-900') : ''}`} 
      ref={chartContainerRef}
    >
      {/* Controls */}
      <div className="absolute top-2 right-2 flex space-x-2 z-10">
        <button
          onClick={toggleFullscreen}
          className="p-1.5 bg-gray-100 hover:bg-gray-200 dark:bg-gray-800 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-md shadow-sm transition-colors"
          title={isFullscreen ? "Exit Fullscreen" : "Full Screen"}
        >
          {isFullscreen ? <Minimize2 size={18} /> : <Maximize2 size={18} />}
        </button>
      </div>

      {/* Instructions */}
      <div className="absolute top-2 left-2 z-10">
        <p className="text-xs text-gray-500 dark:text-gray-400 bg-white/80 dark:bg-gray-800/80 rounded px-2 py-1 shadow-sm">
          Click and drag to zoom
        </p>
      </div>

      {/* Enhanced Zoom slider control */}
      <div className="absolute top-12 right-2 flex items-center gap-3 z-10 bg-white/90 dark:bg-gray-800/90 rounded-md px-3 py-2 shadow-sm backdrop-blur-sm">
        <button
          onClick={() => setZoomLevel(Math.max(zoomLevel - 100, 100))}
          className="p-1 rounded-full hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-400 transition-colors"
          title="Zoom Out"
        >
          <ZoomOut size={14} />
        </button>
        
        <div className="flex-1 relative w-32">
          <input
            type="range"
            min="100"
            max="1000"
            step="50"
            value={zoomLevel}
            onChange={handleZoomChange}
            className="w-full h-1.5 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
            title={`Zoom: ${zoomLevel}%`}
          />
          <div className="absolute -bottom-4 left-0 w-full flex justify-between text-[0.65rem] text-gray-400 dark:text-gray-500">
            <span>100%</span>
            <span>1000%</span>
          </div>
        </div>
        
        <button
          onClick={() => setZoomLevel(Math.min(zoomLevel + 100, 1000))}
          className="p-1 rounded-full hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-400 transition-colors"
          title="Zoom In"
        >
          <ZoomIn size={14} />
        </button>
        
        {!isReset && (
          <button
            onClick={resetZoom}
            className="p-1 rounded-full hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-400 transition-colors"
            title="Reset Zoom"
          >
            <RotateCcw size={14} />
          </button>
        )}
      </div>

      {/* Zoom level indicator */}
      <div className="absolute bottom-16 right-2 z-10">
        <p className="text-xs bg-gray-100/90 dark:bg-gray-800/90 text-gray-700 dark:text-gray-300 rounded px-2 py-1 shadow-sm">
          Zoom: {zoomLevel}%
        </p>
      </div>

      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart
          data={zoomedData}
          margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={zoom}
        >
          <CartesianGrid strokeDasharray="3 3" stroke={isDarkMode ? "#374151" : "#e5e7eb"} />
          <XAxis 
            dataKey="date" 
            tickFormatter={formatXTick}
            tick={{ fontSize: 12, fill: isDarkMode ? "#9ca3af" : "#4b5563" }}
            tickMargin={10}
            minTickGap={30}
            allowDataOverflow
            stroke={isDarkMode ? "#4b5563" : "#9ca3af"}
          />
          <YAxis
            tick={{ fontSize: 12, fill: isDarkMode ? "#9ca3af" : "#4b5563" }}
            tickMargin={10}
            domain={[
              (dataMin: number) => Math.floor(dataMin * 0.9),
              (dataMax: number) => Math.ceil(dataMax * 1.1)
            ]}
            allowDataOverflow
            stroke={isDarkMode ? "#4b5563" : "#9ca3af"}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend wrapperStyle={{ opacity: 0.9 }} />
          
          {/* Confidence Interval Area */}
          <Area
            type="monotone"
            dataKey="upper"
            stroke="transparent"
            fillOpacity={0.1}
            fill="#9333ea"
            name="Upper Bound"
          />
          <Area
            type="monotone"
            dataKey="lower"
            stroke="transparent"
            fillOpacity={0.1}
            fill="#9333ea"
            name="Lower Bound"
          />
          
          {/* Actual and Predicted Lines - with more contrasting colors */}
          <Line 
            type="monotone" 
            dataKey="actual" 
            stroke="#2563eb" 
            strokeWidth={2.5}
            dot={{ fill: '#2563eb', r: 3 }}
            activeDot={{ r: 6 }} 
            name="Actual"
            isAnimationActive={true}
          />
          <Line 
            type="monotone" 
            dataKey="predicted" 
            stroke="#e11d48" 
            strokeWidth={2}
            strokeDasharray="5 5"
            dot={{ fill: '#e11d48', r: 3 }}
            activeDot={{ r: 4 }}
            name="Predicted"
            isAnimationActive={true}
          />
          
          {/* Selection Area for Zooming */}
          {refAreaLeft && refAreaRight && (
            <ReferenceArea
              x1={refAreaLeft}
              x2={refAreaRight}
              strokeOpacity={0.3}
              fill={isDarkMode ? "#8b5cf6" : "#c4b5fd"}
              fillOpacity={0.3}
            />
          )}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
};

export default ForecastChart;