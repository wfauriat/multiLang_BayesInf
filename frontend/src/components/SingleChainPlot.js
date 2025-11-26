import { useMemo } from "react";
import { listToChartData } from '../utils/helper.js';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Legend, ResponsiveContainer } from 'recharts';

export default function SingleChainPlot({ chainData, selectedDim1 }) {

  const chartData = useMemo(() => {
    // Safety checks
    if (!chainData || !Array.isArray(chainData) || chainData.length === 0) {
      console.log("No valid data to plot");
      return [];
    }
        
    const transformedData = listToChartData(chainData)

    return transformedData

  }, [chainData]);

  const getCol = (i) => `col${i}`;
    
  if (chartData.length === 0) {
    return (
      <div style={{ padding: '20px', textAlign: 'center' }}>
        <p>No data available to plot</p>
      </div>
    );
  };
  
  return (
    <div style={{ width: '100%', padding: '40px'}}>
      
      <ResponsiveContainer width="100%" height={450}>
        <LineChart data={chartData}>  
          <CartesianGrid strokeDasharray="3 3" />
          
          <XAxis 
            dataKey="index" 
            label={{ value: 'Index', position: 'insideBottom', offset: -5 }}
            domain={['auto', 'auto']}
          />
          
          <YAxis 
            label={{ value: 'Value', angle: -90, position: 'insideLeft' }}
            domain={['auto', 'auto']}
          />
          
          {/* <Tooltip /> */}
          <Legend />
          
          {/* Plot first column */}
          <Line 
            type="monotone"
            dataKey={getCol(selectedDim1)}
            name={getCol(selectedDim1)}
            stroke="#1c7e29ff" 
            strokeWidth={1}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
