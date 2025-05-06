
export interface City {
  id: string;
  name: string;
}

export interface DateRange {
  from: Date;
  to: Date;
}

export interface SelectedData {
  city: string;
  dateRange: DateRange;
}

export interface ModelParams {
  k: number;
  lookbackWindow: number;
  smoothingFactor: number;
}

export interface ClusterPoint {
  id: number;
  x: number;
  y: number;
  cluster: number;
}

export interface ForecastDataPoint {
  date: string;
  actual: number;
  predicted: number;
  lower?: number;
  upper?: number;
}
