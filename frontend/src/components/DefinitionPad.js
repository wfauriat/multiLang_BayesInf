import { useState } from 'react';
import styles from './DefinitionPad.module.css'
import { handlePost, handlePostCustomCase, ConfigField, simpleCsvParser, matrixToCsv } from '../utils/helper.js';

export default function DefinitionPad({ 
    selectedCase, setSelectedCase, 
    selectedModReg, setSelectedModReg,
    dimChain, 
    selectedDimM, setSelectedDimM,
    selectedDistType, setSelectedDistType,
    paramMLow, paramMHigh,
    setParamMLow, setParamMHigh,
    NMCMC, setNMCMC,
    Nthin, setNthin,
    Nburn, setNburn,
    handleCompute, setIsComputed,
    handleFit,
    endpoint,
    chainData
    }) {

  const [file, setFile] = useState(null);
  const [fileContent, setFileContent] = useState('');


  const handleFileChange = (e) => {
    const uploadedFile = e.target.files[0];
    if (!uploadedFile) {
      return;
    }
  
    setFile(uploadedFile);
    setFileContent('');

    const reader = new FileReader();
    if (uploadedFile.name.endsWith('.csv')) {
      reader.onload = (e) => {
        const csvString = e.target.result;
        const parsedData = simpleCsvParser(csvString);
        setFileContent(parsedData);
        setSelectedCase("Custom")
        console.log(parsedData)
        handlePostCustomCase(parsedData, endpoint + "case/customData")
      };
      reader.readAsText(uploadedFile);
    }
    else {
      setFileContent('');
    }   
  };

  const handleSelectCase = async (e) => {
      const value = e.target.value;
      setSelectedCase(value);
      handlePost(value, "selectedItem", endpoint + "case/select");
      setIsComputed(false);
  };

  const handleSelectReg = async (e) => {
      const value = e.target.value;
      setSelectedModReg(value);
      handlePost(value, "selectedItem", endpoint + "regr/select")
  };

  const handleSelectDist= async () => {
  // const value = e.target.value;
    try {
      const response = await fetch(endpoint + "modelBayes/select", {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ "selected_dist" : selectedDistType,
          "paramMLow" : parseFloat(paramMLow), "paramMHigh": parseFloat(paramMHigh) }),
      });
      if (response.ok) {
        console.log("POST SUCCESSFUL", selectedDistType)
      }
    } catch (error) {
      console.log(error.message);
    }
  }; 

  const handleSelectCurrentM = async (e) => {
      const value = parseInt(e.target.value);
      handlePost(value, "selectedItem", endpoint + "modelBayes/current");
  };

  const configFields = [
      {label: 'NMCMC', value: NMCMC, onChange: setNMCMC,
      endpoint: endpoint + "inf/NMCMC"},
      {label: 'Nthin', value: Nthin, onChange: setNthin,
      endpoint: endpoint + "inf/Nthin"},
      {label: 'Nburn', value: Nburn, onChange: setNburn,
      endpoint: endpoint + "inf/Nburn"}
  ];

  
  const DimSelectFieldX = ({selectDimField, setterDimField, dimChain}) => {
    const optionsArrayX = Array.from({ length: dimChain }, (v, i) => parseInt(i));
    return (
            <select value={selectDimField} 
              onChange={(e) => {
                setterDimField(parseInt(e.target.value));
                handleSelectCurrentM(e);
              }}
              onClick={(e) => {
                setterDimField(parseInt(e.target.value));
                handleSelectCurrentM(e);
              }}
              style={{width:"80px", display:"inline", marginLeft:"1em"}}>
              {optionsArrayX.map((index) => (
                <option key={index} value={index}>{index}</option>
                ))}
            </select>
    );
  };

  const handleExport = () => {
      const csvContent = matrixToCsv(chainData);
      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
      const url = URL.createObjectURL(blob);

      const link = document.createElement("a");
      link.href = url;
      link.download = "exported_data.csv";
      link.click();
      // link.setAttribute("href", url);
      // link.setAttribute("download", "exported_data.csv");
      // document.body.appendChild(link);
      // link.click();
      // document.body.removeChild(link);
      URL.revokeObjectURL(url);
  };
  
     return (
        // DEFINITION PAD
        <div className={styles.DefinitionPad}>

            {/* DATA CASE SELECTION PAD */} 
            <div className={styles.DefSubPad}>
              <h3>Data Case Selection</h3>
              <div>
                <select value={selectedCase} onChange={handleSelectCase} onClick={handleSelectCase}>
                  <option value="Polynomial">Polynomial</option>
                  <option value="Housing">Housing</option>
                  {fileContent && <option value="Custom">Custom</option>}
                </select>
                <div>
                  {/* <button>Custom</button> */}
                  <div style={{display:"flex", flexDirection:"column"}}>
                    <label style={{width:"200px", margin:"auto", textAlign:"center", padding:"0.2em"}}>Import custom</label>
                    <input type="file" style={{margin:"0.2em auto", width:"200px"}}
                    accept='.csv' onChange={handleFileChange}/>
                    {/* {file && <p>Selected: {file.name}</p>} */}
                  </div>
                  <input type="number" placeholder="Data Set Size"></input>
                  <input type="number" placeholder="Cross Validation"></input>
                </div>
              </div>
            </div>

            {/* REGRESSION MODEL PAD */} 
            <div className={styles.DefSubPad}>    
                <h3>Regressor Selection</h3>
                <div>
                    <select value={selectedModReg} onChange={handleSelectReg} onClick={handleSelectReg}>
                        <option value="Linear Polynomial">Linear Polynomial</option>
                        <option value="ElasticNet">Elastic Net</option>
                        <option value="SVR">SVR</option>
                        <option value="RandomForest">Random Forest</option>
                    </select>
                    <div className={styles.TwoButtonArray}>
                        <button>Parameter</button>
                        <button onClick={handleFit}>Fit Model</button>
                    </div>
                </div>
            </div>

            {/* BAYESIAN MODEL PAD */} 
            <div className={styles.DefSubPad}>
                <h3>Bayesian Model Selection</h3>
                <div>
                  <p style={{display:"inline-block", width:"150px", margin:"0.5em 0"}}>Dimension selection</p>
                  <DimSelectFieldX selectDimField={selectedDimM} setterDimField={setSelectedDimM} dimChain={dimChain}/>
                  {/* <select value={selectedDistType} onChange={handleSelectDist} onClick={handleSelectDist}> */}
                  <select value={selectedDistType} onChange={(e) => {setSelectedDistType(e.target.value)}} >
                    <option value="Normal">Normal</option>
                    <option value="Uniform">Uniform</option>
                    <option value="Half-Normal">Half-Normal</option>
                  </select>
                  <div style={{display:"flex", justifyContent:"center", flexDirection:"row"}}>
                    <input type="text" placeholder="LowValue" value={paramMLow}
                    onChange={(e) => {setParamMLow(e.target.value)}}
                    style={{width:"100px", margin:"0.2em 1em"}}/>
                    <input type="text" placeholder="HighValue" value={paramMHigh}
                    onChange={(e) => {setParamMHigh(e.target.value)}}
                    style={{width:"100px", margin:"0.2em 1em"}}/>
                  </div>
                  <button style={{display:"block", width:"150px", margin:"0 auto"}} 
                    onClick={() => {handleSelectDist();}}>Send</button>
                </div>
            </div>

            {/* INFERENCE PAD */} 
            <div className={styles.DefSubPad}>
                <h3>Inference Configuration</h3>
                {configFields.map((field) => (
                    <ConfigField
                    key={field.label}
                    label={field.label}
                    value={field.value}
                    onChange={field.onChange}
                    onSend={handlePost}
                    endpoint={field.endpoint}
                    />
                ))}
              <div>
                <button style={{margin: "0 auto", display:"block", width:"100px", marginTop:"10px"}}
                onClick={handleCompute}>
                  Compute
              </button>
              <button style={{margin: "0 auto", display:"block", width:"100px", marginTop:"10px"}}
              onClick={handleExport}>
                Export to CSV
              </button>
              </div>
            </div>
          </div>
    )

}

