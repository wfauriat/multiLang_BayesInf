import {useState} from "react";
import styles from './App.module.css'
import MatrixPlot from "./components/MatrixPlot";

function App() {

  // const ENDPOINT = "http://127.0.0.1:5000/inf/NMCMC" # not used if proxy set to localhost:5000
  const [value, setValue] = useState('');

  const handleSubmit = async () => {
    try {
      const response = await fetch("inf/NMCMC", {
      // const response = await fetch(ENDPOINT, {
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
    const response = await fetch("compute");
    const data = await response.json();
    console.log(data.chain[0])
  };

    return (
      <div>
        <h2 className={styles.HeaderApp}>BI front-end</h2>
        <div>
          <label style={{paddingRight: "1em"}}>
            NMCMC
          </label>
          <input type="text" value={value} style={{marginRight: "1em"}}
            placeholder="NMCMC" onChange={(e) => setValue(e.target.value)} required>
          </input>
          <button onClick={handleSubmit} >
            Send
          </button>
        </div>
        <div>
          <button style={{marginTop:"3em", marginLeft:"2em"}}
          onClick={handleCompute}>
            Compute
          </button>
        </div>
        <div>
          With Recharts
        </div>
        <MatrixPlot />
      </div>
    );
}

export default App