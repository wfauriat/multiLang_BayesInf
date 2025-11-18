import {useState} from "react";
import styles from './App.module.css'
import MatrixPlot from "./components/MatrixPlot";

function App() {

  const ENDPOINT = "http://127.0.0.1:5000/" // not used if proxy set to localhost:5000
  const [value, setValue] = useState('');

  const handleSubmit = async () => {
    try {
      // const response = await fetch("inf/NMCMC", {
      const response = await fetch(ENDPOINT + "inf/NMCMC", {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ "NMCMC": parseFloat(value) }),
      });
      if (response.ok) {
        console.log("POST SUCCESFUL")
        console.log(value)
      }
    } catch (error) {
      console.log(error.message);
    }
  };

  const handleCompute = async () => {
    const response = await fetch(ENDPOINT + "compute");
    const data = await response.json();
    console.log(data.chain[0])
  };

    return (
      <div>
        <h2 className={styles.HeaderApp}>Bayesian Inference</h2>

        <div className={styles.MainContent}>

          <div className={styles.DefinitionPad}>

            <div className={styles.DefSubPad}>
              <h3>Data Case Selection</h3>
              <div>
                <select>
                  <option>Polynomial</option>
                  <option>Housing</option>
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
              <label style={{paddingRight: "1em"}}>
              NMCMC
              </label>
              <input type="text" value={value} style={{marginRight: "1em"}}
              placeholder="NMCMC" onChange={(e) => setValue(e.target.value)} required>
              </input>
              <button onClick={handleSubmit} >
                Send
              </button>
              <button
              onClick={handleCompute}>
              Compute
              </button>
            </div>
          </div>

          <div className={styles.DisplayPad}>
            <div className={styles.CanvasPad}>
              <h2>Canvas Pad</h2>
              <div>
                <MatrixPlot />
              </div> 
            </div>
            <div className={styles.ControlPad}>
              <h2>Control Pad</h2>
            </div>
          </div>

      </div> 
    </div>

    );
}

export default App