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
      const transformedData = listData.chains.map((row, index) => {
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