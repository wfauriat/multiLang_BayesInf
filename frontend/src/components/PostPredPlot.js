import { useMemo } from "react";
import { listToChartData } from '../utils/helper.js';
import { LineChart, Line, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Legend, ResponsiveContainer } from 'recharts';


export default function PostPredPlot({ xmes, yobs, postY, postYeps, selectedDimR }) {


    const chartData = xmes.map((row, i) => ({
    index: i,
    ...row.reduce((acc, val, dimIndex) => {
        acc[`xmes_${dimIndex}`] = val;
        return acc;
    }, {}),
    yobs: yobs[i],
    postY: postY[i],
    postYeps: postYeps[i]
    }));

//   const chartData = useMemo(() => {
//     // Safety checks
//     if (!chainData || !Array.isArray(chainData) || chainData.length === 0) {
//       console.log("No valid data to plot");
//       return [];
//     }
        
//     const transformedData = listToChartData(chainData)

//     return transformedData

//   }, [chainData]);

    const getCol = (i) => `col${i}`;
    
//   if (chartData.length === 0) {
//     return (
//       <div style={{ padding: '20px', textAlign: 'center' }}>
//         <p>No data available to plot</p>
//       </div>
//     );
//   };
  
  return (
    <div style={{ width: '100%', padding: '40px'}}>
      
      <ResponsiveContainer width="100%" height={450}>
        <ScatterChart width={800} height={600}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="x" type="number" name="X" />
        <YAxis dataKey="y" type="number" name="Y" />
        <Legend />
        
        <Scatter 
            name="Y Observed" 
            data={chartData.map(d => ({ x: d[`xmes_${selectedDimR}`], y: d.yobs }))} 
            fill="#8884d8" 
        />
        {/* <Scatter 
            name="Post Y" 
            data={chartData.map(d => ({ x: d.xmes_1, y: d.postY }))} 
            fill="#82ca9d" 
        />
        <Scatter 
            name="Post Y Eps" 
            data={chartData.map(d => ({ x: d.xmes_1, y: d.postYeps }))} 
            fill="#ffc658" 
        /> */}
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}