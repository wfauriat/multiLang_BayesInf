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
  const [selectedModReg, setSelectedModReg] = useState('');
  const [selectedDimR, setSelectedDimR] = useState(parseInt(0));
  const [selectedDimM, setSelectedDimM] = useState(parseInt(0));
  const [selectedDistType, setSelectedDistType] = useState('Normal');
  const [selectedDim1, setSelectedDim1] = useState(parseInt(0));
  const [selectedDim2, setSelectedDim2] = useState(parseInt(1));
  const [isComputed, setIsComputed] = useState(false);
  const [isDisplayed, setIsDisplayed] = useState(false);
  const [chainData, setChainData] = useState(null);
  const [dimChain, setDimChain] = useState(parseInt(0));
  const [MCsortData, setMCsortData] = useState(null);
  const [LLsortData, setLLsortData] = useState(null);
  const [xmes, setXmes] = useState(null);
  const [dimX, setDimX] = useState(parseInt(0));
  const [yobs, setYobs] = useState(null);
  const [postMAP, setPostMAP] = useState(null);
  const [postY, setPostY] = useState(null);
  const [postYeps, setPostYeps] = useState(null);
  const [yregPred, setYRegPred] = useState(null);


  useEffect(() => {
      if (!isComputed) return;
      else {
        fetchChainData();
        setIsDisplayed(true)
      }
  }, [isComputed, isDisplayed, selectedCase]);


  const fetchChainData = async () => {
      try {
          const response = await fetch(ENDPOINT + "results");
          if (!response.ok) {
              throw new Error(`HTTP error! status: ${response.status}`);
          }
          const data = await response.json();
          setChainData(data.chains);
          setDimChain(parseInt(data.chains[0].length));
          setMCsortData(data.MCsort);
          setLLsortData(data.LLsort);
          setXmes(data.xmes);
          setDimX(parseInt(data.xmes[0].length));
          setYobs(data.obs);
          setPostMAP(data.postMAP);
          setPostY(data.postY);
          setPostYeps(data.postYeps);
          setYRegPred(data.yregPred);
      } catch (err) {
          setChainData(null);
          setMCsortData(null);
          setLLsortData(null);
          setXmes(null);
          setYobs(null);
          setPostMAP(null);
          setPostY(null);
          setPostYeps(null);
          setYRegPred(null);
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

  const handleFit = async () => {
    try{
        const fitresponse = await fetch(ENDPOINT + "regr/fit");
        const response1 = await fitresponse.json()
        const response = await fetch(ENDPOINT + "regr/reg_pred");
        const data = await response.json();
        // console.log(data.yreg_pred)
        console.log(response1.message)
        setYRegPred(data.yreg_pred)
    } catch (err) {
      console.log(err)
    }
  };
 
    return (
      <div>
          <h1 className={styles.HeaderApp}>Bayesian Inference</h1>
          <div className={styles.MainContent}>
            <DefinitionPad  
              selectedCase={selectedCase} setSelectedCase={setSelectedCase}
              selectedModReg={selectedModReg} setSelectedModReg={setSelectedModReg}
              dimChain={dimChain} 
              selectedDimM={selectedDimM} setSelectedDimM={setSelectedDimM} 
              selectedDistType={selectedDistType} setSelectedDistType={setSelectedDistType}
              NMCMC={NMCMC} setNMCMC={setNMCMC} 
              Nthin={Nthin} setNthin={setNthin}
              Nburn={Nburn} setNburn={setNburn}
              handleCompute={handleCompute} setIsComputed={setIsComputed}
              handleFit={handleFit}
              endpoint={ENDPOINT}
            />
            <div className={styles.DisplayPad}>
                <CanvasPad 
                  chainData={chainData} selectedDimR={selectedDimR} 
                  selectedDim1={selectedDim1} selectedDim2={selectedDim2}
                  MCsortData={MCsortData} LLsortData={LLsortData}
                  xmes={xmes} yobs={yobs} postMAP={postMAP} postY={postY} postYeps={postYeps}
                  yregPred={yregPred}
                />
                <ControlPad 
                  selectedDimR={selectedDimR} setSelectedDimR={setSelectedDimR}
                  selectedDim1={selectedDim1} setSelectedDim1={setSelectedDim1}
                  selectedDim2={selectedDim2} setSelectedDim2={setSelectedDim2}
                  dimChain={dimChain} dimX={dimX}
                />
              <div>
              <button onClick={()=>{console.log(yregPred)}}>
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