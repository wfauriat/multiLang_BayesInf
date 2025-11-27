import { BarChart, Bar, XAxis, YAxis, CartesianGrid, ResponsiveContainer } from 'recharts';
import { generateHistogramData, listToChartData} from '../utils/helper.js';


export default function ChainDistPlot({ chainData, selectedDim1}) {

    const binNb = 20;

    const chartData = listToChartData(chainData);
    const singleData = chartData.map(d => d[`col${selectedDim1}`]);
    const histogramData = generateHistogramData(singleData, binNb);

  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart 
        data={histogramData} 
        margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
        barGap={0} // Makes the bars touch to visually represent a histogram
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis 
          dataKey="bin" 
          label={{ value: 'Score Range (Bin)', position: 'bottom' }} 
        />
        <YAxis 
          label={{ value: 'Frequency', angle: -90, position: 'insideLeft' }} 
        />
        {/* <Tooltip /> */}
        <Bar dataKey="frequency" fill="#8884d8" />
      </BarChart>
    </ResponsiveContainer>
  );
};
