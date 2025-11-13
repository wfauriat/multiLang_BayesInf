import { useState } from "react";
import { LineChart, Line, XAxis, YAxis} from 'recharts';


export default function MatrixPlot() {
  const [data, setData] = useState([]);

  const fetchAndPlot = async () => {
    try {
      const response = await fetch('compute');
      const jsonData = await response.json();
      
      // jsonData is N x 4 matrix: [[a1, b1, c1, d1], [a2, b2, c2, d2], ...]
      // Extract first column and format for Recharts
      const chartData = jsonData.chain.map((row, index) => ({
        index: index,           // X-axis (0, 1, 2, ...)
        value: row[0]          // Y-axis (first column)
      }));
      
      setData(chartData);
    } catch (error) {
      console.error('Error:', error);
    }
  };

  return (
    <div>
      <button 
            onClick={fetchAndPlot}>
        Fetch and Plot
      </button>
      
      {data.length > 0 && (
        <LineChart width={800} height={400} data={data}>
          <XAxis 
            dataKey="index" 
            label={{ value: 'Index', position: 'insideBottom', offset: -5 }}
          />
          <YAxis 
            label={{ value: 'Value', angle: -90, position: 'insideLeft' }}
          />
        <Line 
        type="linear"  // Instead of "monotone" - faster
        dataKey="value" 
        stroke="#8884d8" 
        strokeWidth={1}  // Thinner = faster
        dot={false}
        isAnimationActive={false}
        />
        </LineChart>
      )}
    </div>
  );
}