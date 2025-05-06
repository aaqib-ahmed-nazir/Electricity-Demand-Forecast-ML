
import { useRef, useEffect } from "react";
import { ClusterPoint } from "@/types";

interface ClusterVisualizationProps {
  data: ClusterPoint[];
  k: number;
  pcaLabels?: { x: string; y: string };
}

const ClusterVisualization = ({ data, k, pcaLabels }: ClusterVisualizationProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

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
      const centerX = canvas.width / 2;
      const centerY = canvas.height / 2;
      const scale = Math.min(canvas.width, canvas.height) / 10;
      
      // Draw axes
      ctx.strokeStyle = '#e5e7eb'; // Light gray
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
      ctx.strokeStyle = '#f3f4f6'; // Lighter gray
      ctx.lineWidth = 0.5;
      
      // Grid lines
      for (let i = -4; i <= 4; i++) {
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
        
        // Point color based on cluster
        const clusterColor = clusterColors[point.cluster % clusterColors.length];
        
        // Draw point
        ctx.fillStyle = clusterColor;
        ctx.beginPath();
        ctx.arc(x, y, 6, 0, Math.PI * 2);
        ctx.fill();
        
        // Draw border
        ctx.strokeStyle = 'white';
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
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 1;
        ctx.stroke();
        
        // Draw legend text
        ctx.fillStyle = '#374151'; // Dark gray
        ctx.fillText(`Cluster ${i + 1}`, legendX + 15, legendY);
        
        legendY += 20;
      }
      
      // Draw axis labels if provided
      if (pcaLabels) {
        ctx.fillStyle = '#374151'; // Dark gray
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
    };

    // Initial draw
    resizeCanvas();
    
    // Add resize listener
    window.addEventListener('resize', resizeCanvas);
    
    // Cleanup
    return () => {
      window.removeEventListener('resize', resizeCanvas);
    };
  }, [data, k, clusterColors, pcaLabels]);

  return (
    <div className="relative w-full h-full">
      <canvas ref={canvasRef} className="w-full h-full" />
    </div>
  );
};

export default ClusterVisualization;
