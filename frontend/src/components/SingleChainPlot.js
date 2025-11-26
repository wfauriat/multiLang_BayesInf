import { useMemo } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Legend, ResponsiveContainer } from 'recharts';

export default function SingleChainPlot({ chainData, selectedDimR }) {

  const chartData = useMemo(() => {
    // Safety checks
    if (!chainData || !chainData.chains || !Array.isArray(chainData.chains) || chainData.chains.length === 0) {
      console.log("No valid data to plot");
      return [];
    }
        
    // const transformedData = chainData.chains.map((row, index) => ({
    //   index: index,
    //   col1: row[0],  // First column
    //   col2: row[1],  // Second column (optional)
    //   col3: row[2],  // Third column (optional)
    //   col4: row[3],   // Fourth column (optional)
    //   col5: row[4] // TO BE ADDED
    //   }));

    const transformedData = chainData.chains.map((row, index) => {
      const dynamicCols = row.reduce((acc, colValue, colIndex) => {
        const colName = `col${colIndex + 1}`;
          acc[colName] = colValue;
          return acc;
      }, {});
      return {
        index: index,
        ...dynamicCols
        };
      });

    return transformedData

  }, [chainData]);

  const getCol = (i) => `col${i+1}`;
    
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
          />
          
          <YAxis 
            label={{ value: 'Value', angle: -90, position: 'insideLeft' }}
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
