

import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Legend, ResponsiveContainer } from 'recharts';


export default function ChainScatterPlot() {
  // Example data: 2D coordinates (x, y) with associated color values
  const xValues = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
  const yValues = [2, 4, 3, 7, 5, 8, 6, 9, 7, 10];
  const colorValues = [0.1, 0.3, 0.2, 0.7, 0.5, 0.8, 0.6, 0.9, 0.7, 1.0];
  
  const getColor = (value) => {
    // Simple gradient from blue to red
    const r = Math.floor(value * 255);
    const b = Math.floor((1 - value) * 255);
    return `rgb(${r}, 100, ${b})`;
  };
  
  const CustomDot = (props) => {
    const { cx, cy, fill } = props;
    return (
      <circle cx={cx} cy={cy} r={6} fill={fill} stroke="#fff" strokeWidth={1} />
    );
  };

  const data = xValues.map((x, i) => ({
    x: x,
    y: yValues[i],
    colorValue: colorValues[i],
    fill: getColor(colorValues[i])
  }));
  


  return (
    <div>     
        <ResponsiveContainer width="100%" height={400}>
          <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              type="number" 
              dataKey="x" 
              name="X Axis" 
              label={{ value: 'X Coordinate', position: 'insideBottom', offset: -10 }}
            />
            <YAxis 
              type="number" 
              dataKey="y" 
              name="Y Axis"
              label={{ value: 'Y Coordinate', angle: -90, position: 'insideLeft' }}
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
              data={data} 
              shape={<CustomDot />}
            />
          </ScatterChart>
        </ResponsiveContainer>
        
        {/* Color scale legend */}
        <div className="mt-6">
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
        </div>
    </div>
  );
}