
import { useEffect, useRef } from "react";
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
  ComposedChart
} from "recharts";
import { format, parseISO } from "date-fns";
import { NameType, ValueType } from "recharts/types/component/DefaultTooltipContent";

interface ForecastChartProps {
  data: ForecastDataPoint[];
}

const ForecastChart = ({ data }: ForecastChartProps) => {
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
        <div className="bg-white p-3 border border-gray-200 rounded-md shadow-md">
          <p className="font-medium text-sm">{displayDate}</p>
          <div className="flex flex-col gap-1 mt-2">
            {payload[0]?.value !== undefined && (
              <p className="text-xs flex items-center">
                <span className="h-2 w-2 bg-chart-blue rounded-full mr-2"></span>
                Actual: {payload[0].value}
              </p>
            )}
            {payload[1]?.value !== undefined && (
              <p className="text-xs flex items-center">
                <span className="h-2 w-2 bg-chart-purple rounded-full mr-2"></span>
                Predicted: {payload[1].value}
              </p>
            )}
            {payload[2]?.value !== undefined && payload[3]?.value !== undefined && (
              <p className="text-xs flex items-center">
                <span className="h-2 w-2 bg-gray-300 rounded-full mr-2"></span>
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

  return (
    <div className="w-full h-full">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart
          data={data}
          margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis 
            dataKey="date" 
            tickFormatter={formatXTick}
            tick={{ fontSize: 12 }}
            tickMargin={10}
            minTickGap={30}
          />
          <YAxis
            tick={{ fontSize: 12 }}
            tickMargin={10}
            domain={[
              (dataMin: number) => Math.floor(dataMin * 0.9),
              (dataMax: number) => Math.ceil(dataMax * 1.1)
            ]}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend />
          
          {/* Confidence Interval Area */}
          <Area
            type="monotone"
            dataKey="upper"
            stroke="transparent"
            fillOpacity={0.1}
            fill="#8884d8"
            name="Upper Bound"
          />
          <Area
            type="monotone"
            dataKey="lower"
            stroke="transparent"
            fillOpacity={0.1}
            fill="#8884d8"
            name="Lower Bound"
          />
          
          {/* Actual and Predicted Lines */}
          <Line 
            type="monotone" 
            dataKey="actual" 
            stroke="#3b82f6" 
            strokeWidth={2}
            activeDot={{ r: 6 }} 
            name="Actual"
            isAnimationActive={true}
          />
          <Line 
            type="monotone" 
            dataKey="predicted" 
            stroke="#8b5cf6" 
            strokeWidth={2}
            activeDot={{ r: 4 }}
            name="Predicted"
            isAnimationActive={true}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
};

export default ForecastChart;
