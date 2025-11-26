import { useMemo } from "react";
import { listToChartData } from '../utils/helper.js';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Legend, ResponsiveContainer } from 'recharts';

export default function SingleChainPlot({ chainData, selectedDimR }) {

  const chartData = useMemo(() => {
    // Safety checks
    if (!chainData || !chainData.chains || !Array.isArray(chainData.chains) || chainData.chains.length === 0) {
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
      
      {/* ResponsiveContainer makes it adapt to parent size */}
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
            dataKey={getCol(selectedDimR)}
            name={getCol(selectedDimR)}
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
