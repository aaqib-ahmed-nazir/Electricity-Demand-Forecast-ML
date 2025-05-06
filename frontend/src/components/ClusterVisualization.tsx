import { useState, useRef, useEffect } from "react";
import { ClusterPoint } from "@/types";
import { Maximize2, Minimize2, ZoomIn, ZoomOut, RotateCcw } from "lucide-react";
import { useTheme } from "next-themes";

interface ClusterVisualizationProps {
  data: ClusterPoint[];
  k: number;
  pcaLabels?: { x: string; y: string };
}

const ClusterVisualization = ({ data, k, pcaLabels }: ClusterVisualizationProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const { theme, resolvedTheme } = useTheme();

  // Convert the 1-3 zoom scale to percentage for UI consistency
  const zoomPercentage = Math.round(zoomLevel * 100);

  // Define cluster colors
  const clusterColors = [
    '#3b82f6', // blue
    '#8b5cf6', // purple
    '#10b981', // green
    '#ef4444', // red
    '#f59e0b', // amber
    '#ec4899', // pink
    '#06b6d4', // cyan
    '#6366f1', // indigo
    '#84cc16', // lime
    '#14b8a6', // teal
  ];

  // Toggle fullscreen mode
  const toggleFullscreen = () => {
    if (!isFullscreen) {
      if (containerRef.current?.requestFullscreen) {
        containerRef.current.requestFullscreen();
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

  // Zoom in function
  const zoomIn = () => {
    setZoomLevel(prev => Math.min(prev + 0.2, 3));
  };

  // Zoom out function
  const zoomOut = () => {
    setZoomLevel(prev => Math.max(prev - 0.2, 0.5));
  };

  // Reset zoom and pan
  const resetView = () => {
    setZoomLevel(1);
    setPanOffset({ x: 0, y: 0 });
  };

  // Handle slider zoom control
  const handleZoomChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newZoomPercentage = parseInt(e.target.value);
    setZoomLevel(newZoomPercentage / 100); // Convert percentage back to scale
  };

  // Mouse event handlers for panning
  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX, y: e.clientY });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;
    
    const dx = e.clientX - dragStart.x;
    const dy = e.clientY - dragStart.y;
    
    setPanOffset(prev => ({
      x: prev.x + dx,
      y: prev.y + dy
    }));
    
    setDragStart({ x: e.clientX, y: e.clientY });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  // Handle touch events for mobile support
  const handleTouchStart = (e: React.TouchEvent) => {
    if (e.touches.length === 1) {
      setIsDragging(true);
      setDragStart({ 
        x: e.touches[0].clientX, 
        y: e.touches[0].clientY 
      });
    }
  };

  const handleTouchMove = (e: React.TouchEvent) => {
    if (!isDragging || e.touches.length !== 1) return;
    
    const dx = e.touches[0].clientX - dragStart.x;
    const dy = e.touches[0].clientY - dragStart.y;
    
    setPanOffset(prev => ({
      x: prev.x + dx,
      y: prev.y + dy
    }));
    
    setDragStart({ 
      x: e.touches[0].clientX, 
      y: e.touches[0].clientY 
    });
  };

  const handleTouchEnd = () => {
    setIsDragging(false);
  };

  // Handle mouse wheel for zooming
  const handleWheel = (e: React.WheelEvent) => {
    if (e.deltaY < 0) {
      setZoomLevel(prev => Math.min(prev + 0.1, 3));
    } else {
      setZoomLevel(prev => Math.max(prev - 0.1, 0.5));
    }
    e.preventDefault();
  };

  // Determine the current theme
  const currentTheme = resolvedTheme || theme;
  const isDarkMode = currentTheme === 'dark';

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas dimensions
    const resizeCanvas = () => {
      const container = canvas.parentElement;
      if (!container) return;
      
      canvas.width = container.clientWidth;
      canvas.height = container.clientHeight;
      
      drawScatterPlot();
    };

    // Draw the scatter plot
    const drawScatterPlot = () => {
      if (!ctx || !canvas) return;
      
      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Calculate canvas center and scale
      const centerX = canvas.width / 2 + panOffset.x;
      const centerY = canvas.height / 2 + panOffset.y;
      const scale = Math.min(canvas.width, canvas.height) / 10 * zoomLevel;
      
      // Colors based on theme
      const gridColor = isDarkMode ? '#374151' : '#e5e7eb';
      const axisColor = isDarkMode ? '#4b5563' : '#9ca3af';
      const textColor = isDarkMode ? '#d1d5db' : '#374151';
      
      // Draw axes
      ctx.strokeStyle = axisColor;
      ctx.lineWidth = 1;
      
      // X-axis
      ctx.beginPath();
      ctx.moveTo(0, centerY);
      ctx.lineTo(canvas.width, centerY);
      ctx.stroke();
      
      // Y-axis
      ctx.beginPath();
      ctx.moveTo(centerX, 0);
      ctx.lineTo(centerX, canvas.height);
      ctx.stroke();
      
      // Draw grid lines
      ctx.strokeStyle = gridColor;
      ctx.lineWidth = 0.5;
      
      // Grid lines - adjust the range based on zoom level
      const gridRange = Math.ceil(5 / zoomLevel);
      for (let i = -gridRange; i <= gridRange; i++) {
        if (i === 0) continue; // Skip center lines (already drawn)
        
        // Vertical grid lines
        ctx.beginPath();
        ctx.moveTo(centerX + i * scale, 0);
        ctx.lineTo(centerX + i * scale, canvas.height);
        ctx.stroke();
        
        // Horizontal grid lines
        ctx.beginPath();
        ctx.moveTo(0, centerY + i * scale);
        ctx.lineTo(canvas.width, centerY + i * scale);
        ctx.stroke();
      }
      
      // Draw points
      data.forEach((point) => {
        const x = centerX + point.x * scale;
        const y = centerY - point.y * scale; // Negate y for correct orientation
        
        // Skip points outside of visible area (with some padding)
        if (x < -20 || x > canvas.width + 20 || y < -20 || y > canvas.height + 20) {
          return;
        }
        
        // Point color based on cluster
        const clusterColor = clusterColors[point.cluster % clusterColors.length];
        
        // Draw point
        ctx.fillStyle = clusterColor;
        ctx.beginPath();
        ctx.arc(x, y, 6, 0, Math.PI * 2);
        ctx.fill();
        
        // Draw border
        ctx.strokeStyle = isDarkMode ? '#1f2937' : 'white';
        ctx.lineWidth = 1;
        ctx.stroke();
      });
      
      // Draw legend
      const legendX = 20;
      let legendY = 20;
      
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      
      for (let i = 0; i < k; i++) {
        const color = clusterColors[i % clusterColors.length];
        
        // Draw legend circle
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(legendX, legendY, 6, 0, Math.PI * 2);
        ctx.fill();
        
        // Draw legend border
        ctx.strokeStyle = isDarkMode ? '#1f2937' : 'white';
        ctx.lineWidth = 1;
        ctx.stroke();
        
        // Draw legend text
        ctx.fillStyle = textColor;
        ctx.fillText(`Cluster ${i + 1}`, legendX + 15, legendY);
        
        legendY += 20;
      }
      
      // Draw axis labels if provided
      if (pcaLabels) {
        ctx.fillStyle = textColor;
        ctx.font = '12px sans-serif';
        
        // X-axis label
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        ctx.fillText(pcaLabels.x, centerX, canvas.height - 20);
        
        // Y-axis label
        ctx.save();
        ctx.translate(20, centerY);
        ctx.rotate(-Math.PI/2);
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';
        ctx.fillText(pcaLabels.y, 0, 0);
        ctx.restore();
      }

      // Draw zoom level indicator
      const indicatorBgColor = isDarkMode ? 'rgba(31, 41, 55, 0.7)' : 'rgba(249, 250, 251, 0.7)';
      const indicatorTextColor = isDarkMode ? '#e5e7eb' : '#374151';
      
      ctx.fillStyle = indicatorBgColor;
      ctx.fillRect(canvas.width - 80, canvas.height - 30, 70, 20);
      ctx.fillStyle = indicatorTextColor;
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(`Zoom: ${zoomPercentage}%`, canvas.width - 15, canvas.height - 20);
    };

    // Initial draw
    resizeCanvas();
    
    // Add resize listener
    window.addEventListener('resize', resizeCanvas);
    
    // Cleanup
    return () => {
      window.removeEventListener('resize', resizeCanvas);
    };
  }, [data, k, clusterColors, pcaLabels, zoomLevel, panOffset, isDarkMode]);

  return (
    <div 
      ref={containerRef}
      className={`relative w-full h-full ${isFullscreen ? (isDarkMode ? 'bg-gray-900 text-gray-100' : 'bg-white text-gray-900') : ''}`}
      onWheel={handleWheel}
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
          Drag to pan, use mouse wheel to zoom
        </p>
      </div>

      {/* Zoom slider control */}
      <div className="absolute top-12 right-2 flex items-center gap-3 z-10 bg-white/90 dark:bg-gray-800/90 rounded-md px-3 py-2 shadow-sm backdrop-blur-sm">
        <button
          onClick={zoomOut}
          className="p-1 rounded-full hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-400 transition-colors"
          title="Zoom Out"
        >
          <ZoomOut size={14} />
        </button>
        
        <div className="flex-1 relative w-32">
          <input
            type="range"
            min="50"
            max="300"
            step="10"
            value={zoomPercentage}
            onChange={handleZoomChange}
            className="w-full h-1.5 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
            title={`Zoom: ${zoomPercentage}%`}
          />
          <div className="absolute -bottom-4 left-0 w-full flex justify-between text-[0.65rem] text-gray-400 dark:text-gray-500">
            <span>50%</span>
            <span>300%</span>
          </div>
        </div>
        
        <button
          onClick={zoomIn}
          className="p-1 rounded-full hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-400 transition-colors"
          title="Zoom In"
        >
          <ZoomIn size={14} />
        </button>
        
        <button
          onClick={resetView}
          className="p-1 rounded-full hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-400 transition-colors"
          title="Reset View"
        >
          <RotateCcw size={14} />
        </button>
      </div>

      <canvas 
        ref={canvasRef} 
        className="w-full h-full cursor-move"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onTouchStart={handleTouchStart}
        onTouchMove={handleTouchMove}
        onTouchEnd={handleTouchEnd}
      />
    </div>
  );
};

export default ClusterVisualization;
