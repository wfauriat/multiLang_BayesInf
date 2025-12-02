import { useState, useEffect } from 'react';
import styles from './App.module.css';
import DefinitionPad from './components/DefinitionPad';
import CanvasPad from './components/CanvasPad';
import ControlPad from './components/ControlPad';
// import ResizableDashboard from './components/ResizableDashboard';

// Theme icons
const SunIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
  </svg>
);

const MoonIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
  </svg>
);

function App() {

  const ENDPOINT = process.env.REACT_APP_API_URL || 'http://localhost:5000/' || 'http://127.0.0.1:5000/'
  // const ENDPOINT = "http://127.0.0.1:5000/" // not used if proxy set to localhost:5000

  const [NMCMC, setNMCMC] = useState('');
  const [Nthin, setNthin] = useState('');
  const [Nburn, setNburn] = useState('');
  const [selectedCase, setSelectedCase] = useState('');
  const [selectedModReg, setSelectedModReg] = useState('');
  const [selectedDimR, setSelectedDimR] = useState(parseInt(0));
  const [selectedDimM, setSelectedDimM] = useState(parseInt(0));
  const [paramMLow, setParamMLow] = useState(parseFloat(0))
  const [paramMHigh, setParamMHigh] = useState(parseFloat(0))
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
  const [taskId, setTaskId] = useState(null);
  const [computeProgress, setComputeProgress] = useState(0);
  const [computeStatus, setComputeStatus] = useState('');
  const [isComputing, setIsComputing] = useState(false);
  const [darkMode, setDarkMode] = useState(true);


  useEffect(() => {
      if (!isComputed) return;
      else {
        fetchChainData(); 
        setIsDisplayed(true)
      }
  }, [isComputed, isDisplayed, selectedCase]);

  useEffect(() => {
    fetchCase();
    fetchDimChain();
    setSelectedDimM(parseInt(0));
    fetchCurrentDist();
    console.log("changed case")
  }, [selectedCase])

  useEffect(() => {
    fetchCurrentDist();
    fetchCurrParam();
    fetchCurrentDist();
    console.log("changed dimM")
  }, [selectedDimM])


    const fetchCase = async () => {
      const response = await fetch(ENDPOINT + "case/select");
      const data = await response.json();
      setSelectedCase(data.selected_case);
    };

    const fetchDimChain = async () => {
      const response = await fetch(ENDPOINT + "case/dimChain");
      const data = await response.json();
      setDimChain(data.dimChain);
    };

    const fetchCurrentDist = async () => {
      const response = await fetch(ENDPOINT + "modelBayes/select");
      const data = await response.json();
      setSelectedDistType(data.distType);
    };

    const fetchCurrParam = async () => {
      const response = await fetch(ENDPOINT + "modelBayes/paramM");
      const data = await response.json();
      if (data.hasOwnProperty('lowM')) {
        setParamMLow(parseFloat(data.lowM));
        setParamMHigh(parseFloat(data.highM));
      } 
      else {
        setParamMLow(parseFloat(0));
        setParamMHigh(parseFloat(data.param));
      }
    }


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
          setSelectedDimM(parseInt(0))
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
    try {
      setIsComputing(true);
      setComputeProgress(0);
      setComputeStatus('Starting computation...');

      // Start computation
      const response = await fetch(ENDPOINT + "compute", {
        method: 'POST'
      });
      const data = await response.json();

      if (response.status === 202) {
        setTaskId(data.task_id);
        pollTaskStatus(data.task_id);
      } else {
        throw new Error('Failed to start computation');
      }
    } catch (err) {
      console.log(err);
      setIsComputed(false);
      setIsComputing(false);
      setComputeStatus('Failed to start computation');
    }
  };

  const pollTaskStatus = async (task_id) => {
    const interval = setInterval(async () => {
      try {
        const response = await fetch(ENDPOINT + `task/${task_id}/status`);
        const data = await response.json();

        setComputeStatus(data.status || '');
        setComputeProgress(data.progress || 0);

        if (data.state === 'SUCCESS') {
          clearInterval(interval);

          // Fetch results
          const resultResponse = await fetch(ENDPOINT + `task/${task_id}/result`);
          const resultData = await resultResponse.json();

          // Update state with results
          setChainData(resultData.chains);
          setDimChain(parseInt(resultData.chains[0].length));
          setMCsortData(resultData.MCsort);
          setLLsortData(resultData.LLsort);
          setXmes(resultData.xmes);
          setDimX(parseInt(resultData.xmes[0].length));
          setYobs(resultData.obs);
          setPostMAP(resultData.postMAP);
          setPostY(resultData.postY);
          setPostYeps(resultData.postYeps);
          setYRegPred(resultData.yregPred);
          setSelectedDimM(parseInt(0));

          setIsComputed(true);
          setIsDisplayed(false);
          setIsComputing(false);
          setComputeStatus('Computation completed!');

        } else if (data.state === 'FAILURE') {
          clearInterval(interval);
          setIsComputed(false);
          setIsComputing(false);
          setComputeStatus('Computation failed: ' + (data.error || 'Unknown error'));

        } else if (data.state === 'RUNNING') {
          // Continue polling - task is still running
        }

      } catch (err) {
        console.log(err);
        clearInterval(interval);
        setIsComputed(false);
        setIsComputing(false);
        setComputeStatus('Error checking task status');
      }
    }, 1000); // Poll every second
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
      <div className={darkMode ? 'dark' : ''}>
        <div className="min-h-screen bg-white dark:bg-slate-900 transition-colors">
          {/* Header */}
          <header className="border-b border-slate-200 dark:border-slate-800 bg-white/95 dark:bg-slate-900/95 backdrop-blur supports-[backdrop-filter]:bg-white/60 dark:supports-[backdrop-filter]:bg-slate-900/60 sticky top-0 z-50">
            <div className="max-w-7xl mx-auto px-8 py-4">
              <div className="flex items-center justify-between">
                <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-100">
                  Bayesian Inference
                </h1>

                {/* Theme Toggle */}
                <button
                  onClick={() => setDarkMode(!darkMode)}
                  className="p-2 rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors"
                  aria-label="Toggle theme"
                >
                  {darkMode ? <SunIcon /> : <MoonIcon />}
                </button>
              </div>
            </div>
          </header>

          {/* Main Content */}
          <div className="max-w-7xl mx-auto px-8 py-6">
            <div className="grid grid-cols-[380px_1fr] gap-6">
              {/* Sidebar */}
              <aside className="sticky top-24 self-start">
                <DefinitionPad
                  selectedCase={selectedCase} setSelectedCase={setSelectedCase}
                  selectedModReg={selectedModReg} setSelectedModReg={setSelectedModReg}
                  dimChain={dimChain}
                  selectedDimM={selectedDimM} setSelectedDimM={setSelectedDimM}
                  selectedDistType={selectedDistType} setSelectedDistType={setSelectedDistType}
                  paramMLow={paramMLow} paramMHigh={paramMHigh}
                  setParamMLow={setParamMLow} setParamMHigh={setParamMHigh}
                  NMCMC={NMCMC} setNMCMC={setNMCMC}
                  Nthin={Nthin} setNthin={setNthin}
                  Nburn={Nburn} setNburn={setNburn}
                  handleCompute={handleCompute} setIsComputed={setIsComputed}
                  handleFit={handleFit}
                  endpoint={ENDPOINT}
                  chainData={chainData}
                  isComputing={isComputing}
                  computeProgress={computeProgress}
                  computeStatus={computeStatus}
                />
              </aside>

              {/* Main content area */}
              <main className="min-w-0">
                <CanvasPad
                  chainData={chainData} selectedDimR={selectedDimR}
                  selectedDim1={selectedDim1} selectedDim2={selectedDim2}
                  MCsortData={MCsortData} LLsortData={LLsortData}
                  xmes={xmes} yobs={yobs} postMAP={postMAP} postY={postY} postYeps={postYeps}
                  yregPred={yregPred}
                />
                <div className="mt-6">
                  <ControlPad
                    selectedDimR={selectedDimR} setSelectedDimR={setSelectedDimR}
                    selectedDim1={selectedDim1} setSelectedDim1={setSelectedDim1}
                    selectedDim2={selectedDim2} setSelectedDim2={setSelectedDim2}
                    dimChain={dimChain} dimX={dimX}
                  />
                </div>
              </main>
            </div>
          </div>
        </div>
      </div>
    );
}

export default App