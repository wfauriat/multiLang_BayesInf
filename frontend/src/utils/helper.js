export const handlePost = async (val, key, endpoint) => {
    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ [key] : val }),
      });
      if (response.ok) {
        console.log("POST SUCCESSFUL", val)
      }
    } catch (error) {
      console.log(error.message);
    }
  };

export function ConfigField({ label, value, onChange, onSend, endpoint }) {
  return (
    <div>
      <label style={{ minWidth: "80px", display: "inline-block"}}>
          {label}
      </label>
      <input 
          type="text" 
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={label}
          style={{width: "100px", display: "inline-block"}}
          required
      />
      <button 
          onClick={() => onSend(parseFloat(value), label, endpoint)}>
          Send
      </button>
    </div>
  );
}

export const listToChartData = (listData) => {
      const transformedData = listData.map((row, index) => {
      const dynamicCols = row.reduce((acc, colValue, colIndex) => {
        const colName = `col${colIndex}`;
          acc[colName] = colValue;
          return acc;
      }, {});
      return {
        index: index,
        ...dynamicCols
        };
      });

    // const transformedData = chainData.chains.map((row, index) => ({
    //   index: index,
    //   col1: row[0],  // First column
    //   col2: row[1],  // Second column (optional)
    //   col3: row[2],  // Third column (optional)
    //   col4: row[3],   // Fourth column (optional)
    //   col5: row[4] // TO BE ADDED
    //   }));

    return transformedData
}


export const generateHistogramData = (data, binCount = 5) => {
  if (!data || data.length === 0) return [];

  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min;
  
  const binWidth = range === 0 ? 1 : range / binCount;

  const bins = [];
  const frequencies = new Array(binCount).fill(0);
  
  const precision = 2; 

  for (let i = 0; i < binCount; i++) {
    const lowerBound = min + i * binWidth;
    let upperBound = lowerBound + binWidth;

    if (Number.isInteger(min) && Number.isInteger(max) && binWidth >= 1) {
        upperBound = lowerBound + binWidth - (i === binCount - 1 && max > lowerBound + binWidth - 1 ? 0 : 1);
        
         const lastUpper = max;
        if (i === binCount - 1) {
             bins.push(`${lowerBound}-${lastUpper}`);
        } else {
             bins.push(`${lowerBound}-${upperBound}`);
        }
    } 
    else {
      const displayLower = lowerBound.toFixed(precision);
      const displayUpper = upperBound.toFixed(precision);
      bins.push(`${displayLower} to < ${displayUpper}`);
    }
  }

  for (const value of data) {
    let binIndex = Math.floor((value - min) / binWidth);

    if (binIndex >= binCount) {
      binIndex = binCount - 1; 
    }
    frequencies[binIndex] += 1;
  }
  const histogramData = bins.map((binLabel, index) => ({
    bin: binLabel,
    frequency: frequencies[index],
  }));

  return histogramData;
};