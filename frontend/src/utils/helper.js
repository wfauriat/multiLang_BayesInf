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

export const handlePostCustomCase = async (data, endpoint) => {
 try {
  const response = await fetch(endpoint, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({"data": data}),
  });
    if (response.ok) {
        console.log("POST SUCCESSFUL")
    }
 } catch (error) {
      console.log(error.message);
    }
};

export function ConfigField({ label, value, onChange, onSend, endpoint }) {
  return (
    <div className="space-y-2">
      <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">
        {label}
      </label>
      <div className="flex gap-2">
        <input
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={label}
          className="flex-1 bg-white dark:bg-slate-700 border border-slate-300 dark:border-slate-600 text-slate-900 dark:text-slate-100 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          required
        />
        <button
          onClick={() => onSend(parseFloat(value), label, endpoint)}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md font-medium text-sm transition-colors"
        >
          Send
        </button>
      </div>
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

export function simpleCsvParser(csvString) {
  const rows = csvString.trim().split('\n');
  const matrix = rows
    .filter(line => line.trim() !== '')
    .map(row => row.split(',').map(cell => parseFloat(cell.trim())));
  return matrix;
}


export function matrixToCsv(matrix){
  return matrix.map(row => 
    row.join(',')
  ).join('\n');
};