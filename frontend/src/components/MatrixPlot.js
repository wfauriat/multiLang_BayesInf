import { useMemo } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Legend, ResponsiveContainer } from 'recharts';

export default function MatrixPlot({ chainData, selectedDimR }) {

  // Transform the data: extract first column from the 1000x4 array
  const chartData = useMemo(() => {
    // Safety checks
    if (!chainData || !chainData.chains || !Array.isArray(chainData.chains) || chainData.chains.length === 0) {
      console.log("No valid data to plot");
      return [];
    }
    
    // console.log(`Processing ${chainData.chains.length} rows`);
    
    // Map each row to a chart-friendly format
    return chainData.chains.map((row, index) => ({
      index: index,
      col1: row[0],  // First column
      col2: row[1],  // Second column (optional)
      col3: row[2],  // Third column (optional)
      col4: row[3]   // Fourth column (optional)
    }));
  }, [chainData]);

  const getCol = (i) => `col${i+1}`;
  // console.log(getCol(selectedDimR));
    
  // If no data, show a message
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
            name="Column 1"
            stroke="#1c7e29ff" 
            strokeWidth={1}
            dot={false}
            isAnimationActive={false}
          />
          
          {/* Uncomment to plot additional columns */}
          {/* <Line 
            type="monotone"
            dataKey="col2" 
            name="Column 2"
            stroke="#2563eb" 
            strokeWidth={1}
            dot={false}
            isAnimationActive={false}
          /> */}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
