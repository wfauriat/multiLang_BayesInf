import { useState } from 'react';
import styles from './App.module.css';
import DefinitionPad from './components/DefinitionPad';
import CanvasPad from './components/CanvasPad';
import ControlPad from './components/ControlPad';

function App() {

  const ENDPOINT = "http://127.0.0.1:5000/" // not used if proxy set to localhost:5000

  const [NMCMC, setNMCMC] = useState('');
  const [Nthin, setNthin] = useState('');
  const [Ntune, setNtune] = useState('');
  const [selectedCase, setSelectedCase] = useState('');
  const [selectedDimR, setSelectedDimR] = useState(parseInt(0));
  const [chainData, setChainData] = useState(null);

  const handleCompute = async () => {
        const response = await fetch(ENDPOINT + "compute");
        const data = await response.json();
        setChainData(data)
        console.log(chainData)
  };
 
    return (
      <div>
        <h1 className={styles.HeaderApp}>Bayesian Inference</h1>
        <div className={styles.MainContent}>
          <DefinitionPad  
            selectedCase={selectedCase} setSelectedCase={setSelectedCase}
            NMCMC={NMCMC} setNMCMC={setNMCMC} 
            Nthin={Nthin} setNthin={setNthin}
            Ntune={Ntune} setNtune={setNtune}
            handleCompute={handleCompute}
            endpoint={ENDPOINT}
          />

          <div className={styles.DisplayPad}>
            <CanvasPad 
              chainData={chainData} selectedDimR={selectedDimR} 
            />
            <ControlPad 
              selectedDimR={selectedDimR} setSelectedDimR={setSelectedDimR}
            />
          </div>
      </div> 
    </div>

    );
}

export default App