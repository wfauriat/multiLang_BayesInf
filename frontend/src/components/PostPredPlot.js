import { useMemo } from "react";
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Legend, ResponsiveContainer } from 'recharts';


export default function PostPredPlot({ xmes, yobs, postMAP, postY, postYeps, yregPred, selectedDimR }) {

    // const transpose = arr => arr[0].map((_, i) => arr.map(row => row[i]));
    // const t_postY = transpose(postY)
    // const t_postYeps = transpose(postYeps)

    const limitedXmes = xmes.slice(0,30);

    const chartData = limitedXmes.map((row, i) => ({
    index: i,
    ...row.reduce((acc, val, dimIndex) => {
        acc[`xmes_${dimIndex}`] = val;
        return acc;
    }, {}),
    yobs: yobs[i],
    postMAP: postMAP[i],
    postY: postY[i],
    postYeps: postYeps[i],
    yregPred: yregPred[i]
    }));



  // const customDot = (props) => {
  //   const { cx, cy, fill } = props;
  //   return (
  //     <circle cx={cx} cy={cy} r={8} fill={"red"} />  // r controls the radius
  //   );
  // };

//   const chartData = useMemo(() => {
//     // Safety checks
//     if (!chainData || !Array.isArray(chainData) || chainData.length === 0) {
//       console.log("No valid data to plot");
//       return [];
//     }
        
//     const transformedData = listToChartData(chainData)

//     return transformedData

//   }, [chainData]);

    // const getCol = (i) => `col${i}`;
    
    const scatterData = useMemo(() => ({
      postY: chartData.map(d => ({ x: d[`xmes_${selectedDimR}`], y: d.postY[0] })),
      postMAP: chartData.map(d => ({ x: d[`xmes_${selectedDimR}`], y: d.postMAP })),
      yobs: chartData.map(d => ({ x: d[`xmes_${selectedDimR}`], y: d.yobs })),
      postYeps0: chartData.map(d => ({ x: d[`xmes_${selectedDimR}`], y: d.postYeps[0] })),
      postYeps: [0,2,4,6,8,10].map(i => 
        chartData.map(d => ({ x: d[`xmes_${selectedDimR}`], y: d.postYeps[i] }))
      ),
      yregPred: chartData.map(d => ({ x: d[`xmes_${selectedDimR}`], y: d.yregPred }))
    }), [chartData, selectedDimR]);


  return (
    <div style={{ width: '100%', padding: '40px'}}>
      
      <ResponsiveContainer width="100%" height={450}>
        <ScatterChart width={800} height={600}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="x" type="number" name="X" domain={['auto', 'auto']}/>
        <YAxis dataKey="y" type="number" name="Y" domain={['auto', 'auto']}/>
        <Legend />
        <Scatter 
            name="Post MAP" 
            data={scatterData.postMAP} 
            fill="green" shape="cross"  isAnimationActive={false}
        />
          {scatterData.postYeps.map((data, i) => (
            <Scatter key={i} data={data} fill="blue" legendType="none" fillOpacity={0.2} />
          ))
          }

        <Scatter 
            name="Post Y Eps" 
            data={scatterData.postYeps0} 
            fill="blue" fillOpacity={0.2} isAnimationActive={false}
        />
        <Scatter 
            name="Regression Pred" 
            data={scatterData.yregPred} 
            fill="orange" fillOpacity={1} shape="diamond"  isAnimationActive={false}
        />
        <Scatter 
            name="Y Observed" 
            data={scatterData.yobs} 
            fill="red" fillOpacity={1} shape="diamond"  isAnimationActive={false}
        />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}