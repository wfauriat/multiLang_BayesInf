import { useState, useEffect } from 'react';
import styles from './App.module.css';
import DefinitionPad from './components/DefinitionPad';
import CanvasPad from './components/CanvasPad';
import ControlPad from './components/ControlPad';
// import ResizableDashboard from './components/ResizableDashboard';

function App() {

  const ENDPOINT = "http://127.0.0.1:5000/" // not used if proxy set to localhost:5000

  const [NMCMC, setNMCMC] = useState('');
  const [Nthin, setNthin] = useState('');
  const [Nburn, setNburn] = useState('');
  const [selectedCase, setSelectedCase] = useState('');
  const [selectedDimR, setSelectedDimR] = useState(parseInt(0));
  const [isComputed, setIsComputed] = useState(false);
  const [isDisplayed, setIsDisplayed] = useState(false);
  const [chainData, setChainData] = useState(null);
  const [dimChain, setDimChain] = useState(parseInt(0))

  useEffect(() => {
      if (!isComputed) return;
      else {
        fetchChainData();
        setIsDisplayed(true)
      }
  }, [isComputed, isDisplayed, selectedCase]);


  const fetchChainData = async () => {
      try {
          const response = await fetch(ENDPOINT + "chains");
          if (!response.ok) {
              throw new Error(`HTTP error! status: ${response.status}`);
          }
          const data = await response.json();
          setChainData(data);
          setDimChain(parseInt(data.chains[0].length))
      } catch (err) {
          setChainData(null);
          console.log(err)
      }
  };

  const handleCompute = async () => {
    try{
        const response = await fetch(ENDPOINT + "compute");
        const data = await response.json();
        console.log(data.message)
        setIsComputed(true)
        setIsDisplayed(false)
    } catch (err) {
      console.log(err)
      setIsComputed(false);
    }

  };
 
    return (
      <div>
          <h1 className={styles.HeaderApp}>Bayesian Inference</h1>
          <div className={styles.MainContent}>
            <DefinitionPad  
              selectedCase={selectedCase} setSelectedCase={setSelectedCase}
              NMCMC={NMCMC} setNMCMC={setNMCMC} 
              Nthin={Nthin} setNthin={setNthin}
              Nburn={Nburn} setNburn={setNburn}
              handleCompute={handleCompute} setIsComputed={setIsComputed}
              endpoint={ENDPOINT}
            />
            <div className={styles.DisplayPad}>
                <CanvasPad 
                  chainData={chainData} selectedDimR={selectedDimR} 
                />
                <ControlPad 
                  selectedDimR={selectedDimR} setSelectedDimR={setSelectedDimR}
                  dimChain={dimChain}
                />
              <div>
              <button onClick={()=>{console.log(chainData.chains[0].length)}}>
                Test
              </button>
            </div>
            </div>
        </div>
        {/* <div>
          <ResizableDashboard />
        </div>  */}
    </div>
    );
}

export default App