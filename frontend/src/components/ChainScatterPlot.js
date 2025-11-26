import { useMemo } from "react";
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Legend, ResponsiveContainer, Cell } from 'recharts';
import { listToChartData } from '../utils/helper.js';

export default function ChainScatterPlot({chainData, selectedDim1, selectedDim2}) {

  const chartData = useMemo(() => {
    // Safety checks
    if (!chainData || !chainData.chains || !Array.isArray(chainData.chains) || chainData.chains.length === 0) {
      console.log("No valid data to plot");
      return [];
    }
      
    const limitedData = {chains: [...chainData.chains].slice(0, 300)}
    const transformedData = listToChartData(limitedData)
    // const transformedData = listToChartData(chainData)

    return transformedData

  }, [chainData]);

  const getCol = (i) => `col${i}`;


const getJetColor = (value, min, max) => {
  // Normalize value to 0-1 range
  const normalized = (value - min) / (max - min);
  let r, g, b;
  if (normalized < 0.125) {
    r = 0;
    g = 0;
    b = 0.5 + normalized / 0.125 * 0.5;
  } else if (normalized < 0.375) {
    r = 0;
    g = (normalized - 0.125) / 0.25;
    b = 1;
  } else if (normalized < 0.625) {
    r = (normalized - 0.375) / 0.25;
    g = 1;
    b = 1 - (normalized - 0.375) / 0.25;
  } else if (normalized < 0.875) {
    r = 1;
    g = 1 - (normalized - 0.625) / 0.25;
    b = 0;
  } else {
    r = 1 - (normalized - 0.875) / 0.125 * 0.5;
    g = 0;
    b = 0;
  }
  const toHex = (val) => {
    const hex = Math.round(val * 255).toString(16);
    return hex.length === 1 ? '0' + hex : hex;
  };
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
};

  // const zMin = Math.min(...chartData.map(d => d.col0));
  // const zMax = Math.max(...chartData.map(d => d.col0));

  // console.log(chartData)
  // console.log(zMin)
  // console.log(zMax)

        // <Scatter name="Data Points" data={data}>
        //   {data.map((entry, index) => (
        //     <Cell key={`cell-${index}`} fill={getJetColor(entry.z, zMin, zMax)} />
        //   ))}
        // </Scatter>



  // Example data: 2D coordinates (x, y) with associated color values
  // const xValues = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
  // const yValues = [2, 4, 3, 7, 5, 8, 6, 9, 7, 10];
  // const colorValues = [0.1, 0.3, 0.2, 0.7, 0.5, 0.8, 0.6, 0.9, 0.7, 1.0];
  
  // const getColor = (value) => {
  //   // Simple gradient from blue to red
  //   const r = Math.floor(value * 255);
  //   const b = Math.floor((1 - value) * 255);
  //   return `rgb(${r}, 100, ${b})`;
  // };
  
  // const CustomDot = (props) => {
  //   const { cx, cy, fill } = props;
  //   return (
  //     <circle cx={cx} cy={cy} r={6} fill={fill} stroke="#fff" strokeWidth={1} />
  //   );
  // };

  // SYNTAX FOR COLOR DOTS BELOW
  //   <Scatter 
  //   name="Data Points" 
  //   data={chartData} 
  //   shape={<CustomDot />}
  // />

  // const data = xValues.map((x, i) => ({
  //   x: x,
  //   y: yValues[i],
  //   colorValue: colorValues[i],
  //   fill: getColor(colorValues[i])
  // }));
  


  return (
    <div>     
        <ResponsiveContainer width="100%" height={400}>
          <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              type="number" 
              dataKey={getCol(selectedDim1)} 
              name="X Axis" 
              label={{ value: 'X Coordinate', position: 'insideBottom', offset: -10 }}
              domain={['auto', 'auto']}
            />
            <YAxis 
              type="number" 
              dataKey={getCol(selectedDim2)} 
              name="Y Axis"
              label={{ value: 'Y Coordinate', angle: -90, position: 'insideLeft' }}
              domain={['auto', 'auto']}
            />
            {/* <ZAxis range={[60, 60]} /> */}
            {/* <Tooltip 
              cursor={{ strokeDasharray: '3 3' }}
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const data = payload[0].payload;
                  return (
                    <div className="bg-white p-3 border border-gray-300 rounded shadow-md">
                      <p className="font-semibold">Point Data:</p>
                      <p>X: {data.x}</p>
                      <p>Y: {data.y}</p>
                      <p>Color Value: {data.colorValue.toFixed(2)}</p>
                    </div>
                  );
                }
                return null;
              }}
            /> */}
            <Legend />
            <Scatter 
              name="Data Points" 
              data={chartData} 
              fill={"blue"}
            />
          </ScatterChart>
        </ResponsiveContainer>
        
        {/* Color scale legend */}
        {/* <div className="mt-6">
          <p className="text-sm font-semibold mb-2 text-gray-700">Color Scale:</p>
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-600">0.0</span>
            <div className="flex-1 h-6 rounded" 
                 style={{
                   background: 'linear-gradient(to right, rgb(0, 100, 255), rgb(128, 100, 128), rgb(255, 100, 0))'
                 }}
            />
            <span className="text-xs text-gray-600">1.0</span>
          </div>
        </div> */}
    </div>
  );
}