import styles from './DefinitionPad.module.css'
import { handlePost } from '../utils/helper.js';

export default function DefinitionPad({ 
    selectedCase, setSelectedCase, 
    NMCMC, setNMCMC,
    Nthin, setNthin,
    Ntune, setNtune,
    handleCompute,
    endpoint
    }) {

    const handleSelectCase = async (e) => {
        const value = e.target.value;
        setSelectedCase(value);
        handlePost(value, "selectedItem", endpoint + "case/select")
    };
  
    return (
        <div className={styles.DefinitionPad}>

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

            <div className={styles.DefSubPad}>
              <h3>Bayesian Model Selection</h3>
            </div>

            <div className={styles.DefSubPad}>
              <h3>Inference Configuration</h3>
              <div className={styles.OneLineField}>
                <label style={{paddingRight: "1em"}}>
                NMCMC
                </label>
                <input type="text" value={NMCMC} style={{marginRight: "1em", width:"30%"}}
                  placeholder="NMCMC" onChange={(e) => setNMCMC(e.target.value)} required>
                </input>
                <button onClick={() => handlePost(parseFloat(NMCMC), "NMCMC", endpoint + "inf/NMCMC")} >
                  Send
                </button>
              </div>
              <div className={styles.OneLineField}>
                <label style={{paddingRight: "1em"}}>
                Nthin
                </label>
                <input type="text" value={Nthin} style={{marginRight: "1em", width:"30%"}} 
                  onChange={(e) => setNthin(e.target.value)}
                  placeholder="Nthin">
                </input>
                <button onClick={() => handlePost(parseFloat(Nthin), "NMCMC", endpoint + "inf/Nthin")} >
                  Send
                </button>
              </div>
              <div className={styles.OneLineField}>
                <label style={{paddingRight: "1em"}}>
                Nthin
                </label>
                <input type="text" value={Ntune} style={{marginRight: "1em", width:"30%"}}
                  onChange={(e) => setNtune(e.target.value)}
                  placeholder="Ntune">
                </input>
                <button onClick={() => handlePost(parseFloat(Ntune), "NMCMC", endpoint + "inf/Ntune")} >
                  Send
                </button>
              </div>
              <div>
                <button
                onClick={handleCompute}>
                  Compute
              </button>
              </div>
            </div>
          </div>
    )

}

