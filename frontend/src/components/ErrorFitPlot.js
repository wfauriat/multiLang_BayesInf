import { useMemo } from "react";
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Legend, ResponsiveContainer } from 'recharts';

export default function ErrorFitPlot({ xmes, yobs, postMAP, yregPred, selectedDimR }) {

    const limitedXmes = xmes.slice(0,30);

    const chartData = limitedXmes.map((row, i) => ({
    index: i,
    ...row.reduce((acc, val, dimIndex) => {
        acc[`xmes_${dimIndex}`] = val;
        return acc;
    }, {}),
    errMAP: yobs[i] - postMAP[i],
    errReg: yobs[i] - yregPred[i],
    }));

    const scatterData = useMemo(() => ({
        errMAP: chartData.map(d => ({ x: d[`xmes_${selectedDimR}`], y: d.errMAP })),
        errReg: chartData.map(d => ({ x: d[`xmes_${selectedDimR}`], y: d.errReg })),
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
                        data={scatterData.errMAP} 
                        fill="green" shape="cross"  isAnimationActive={false}
                    />
                    <Scatter 
                        name="Regression Pred" 
                        data={scatterData.errReg} 
                        fill="orange" fillOpacity={1} shape="diamond"  isAnimationActive={false}
                    />
                    </ScatterChart>
                  </ResponsiveContainer>
        </div>
    )

};