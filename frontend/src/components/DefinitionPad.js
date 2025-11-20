import styles from './DefinitionPad.module.css'
import { handlePost, ConfigField } from '../utils/helper.js';

export default function DefinitionPad({ 
    selectedCase, setSelectedCase, 
    NMCMC, setNMCMC,
    Nthin, setNthin,
    Nburn, setNburn,
    handleCompute, setIsComputed,
    endpoint
    }) {

    const handleSelectCase = async (e) => {
        const value = e.target.value;
        setSelectedCase(value);
        handlePost(value, "selectedItem", endpoint + "case/select")
        setIsComputed(false)
    };

    const configFields = [
        {label: 'NMCMC', value: NMCMC, onChange: setNMCMC,
        endpoint: endpoint + "inf/NMCMC"},
        {label: 'Nthin', value: Nthin, onChange: setNthin,
        endpoint: endpoint + "inf/Nthin"},
        {label: 'Nburn', value: Nburn, onChange: setNburn,
        endpoint: endpoint + "inf/Nburn"}
    ];
  
    return (
        // DEFINITION PAD
        <div className={styles.DefinitionPad}>

            {/* DATA CASE SELECTION PAD */} 
            <div className={styles.DefSubPad}>
              <h3>Data Case Selection</h3>
              <div>
                <select value={selectedCase} onChange={handleSelectCase}>
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
                    <select>
                        <option>Linear Polynomial</option>
                        <option>Elastic Net</option>
                        <option>SVR</option>
                        <option>Random Forest</option>
                    </select>
                    <div className={styles.TwoButtonArray}>
                        <button>Parameter</button>
                        <button>Fit Model</button>
                    </div>
                </div>
            </div>

            {/* BAYESIAN MODEL PAD */} 
            <div className={styles.DefSubPad}>
                <h3>Bayesian Model Selection</h3>
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

