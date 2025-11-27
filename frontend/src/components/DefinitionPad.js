import styles from './DefinitionPad.module.css'
import { handlePost, ConfigField } from '../utils/helper.js';

export default function DefinitionPad({ 
    selectedCase, setSelectedCase, 
    selectedModReg, setSelectedModReg,
    dimChain, 
    selectedDimM, setSelectedDimM,
    selectedDistType, setSelectedDistType,
    NMCMC, setNMCMC,
    Nthin, setNthin,
    Nburn, setNburn,
    handleCompute, setIsComputed,
    handleFit,
    endpoint
    }) {

    const handleSelectCase = async (e) => {
        const value = e.target.value;
        setSelectedCase(value);
        handlePost(value, "selectedItem", endpoint + "case/select")
        setIsComputed(false)
    };

    const handleSelectReg = async (e) => {
        const value = e.target.value;
        setSelectedModReg(value);
        handlePost(value, "selectedItem", endpoint + "regr/select")
    };

    const handleSelectDist= async (e) => {
        const value = e.target.value;
        setSelectedDistType(value);
        handlePost(value, "selectedItem", endpoint + "modelBayes/select")
    };    
    
    const handleSelectCurrentM= async (e) => {
        const value = parseInt(e.target.value);
        handlePost(value, "selectedItem", endpoint + "modelBayes/current")
    };

    const configFields = [
        {label: 'NMCMC', value: NMCMC, onChange: setNMCMC,
        endpoint: endpoint + "inf/NMCMC"},
        {label: 'Nthin', value: Nthin, onChange: setNthin,
        endpoint: endpoint + "inf/Nthin"},
        {label: 'Nburn', value: Nburn, onChange: setNburn,
        endpoint: endpoint + "inf/Nburn"}
    ];

    const optionsArrayX = Array.from({ length: dimChain }, (v, i) => parseInt(i));
    const DimSelectFieldX = ({selectDimField, setterDimField}) => {
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
    }
  
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
                </select>
                <div className={styles.FourButtonArray}>
                  <button>Custom</button>
                  <button>Import Data</button>
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
                  <DimSelectFieldX selectDimField={selectedDimM} setterDimField={setSelectedDimM}/>
                  <select value={selectedDistType} onChange={handleSelectDist} onClick={handleSelectDist}>
                    <option value="Normal">Normal</option>
                    <option value="Uniform">Uniform</option>
                    <option value="Half-Normal">Half-Normal</option>
                  </select>
                  <div style={{display:"flex", justifyContent:"center", flexDirection:"row"}}>
                    <input type="text" placeholder="LowValue" 
                    style={{width:"100px", margin:"0.2em 1em"}}/>
                    <input type="text" placeholder="HighValue" 
                    style={{width:"100px", margin:"0.2em 1em"}}/>
                  </div>
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
              </div>
            </div>
          </div>
    )

}

