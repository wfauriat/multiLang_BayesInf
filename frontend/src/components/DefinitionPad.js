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
    chainData,
    isComputing,
    computeProgress,
    computeStatus
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
            <select
              value={selectDimField}
              onChange={(e) => {
                setterDimField(parseInt(e.target.value));
                handleSelectCurrentM(e);
              }}
              onClick={(e) => {
                setterDimField(parseInt(e.target.value));
                handleSelectCurrentM(e);
              }}
              className="w-full bg-white dark:bg-slate-700 border border-slate-300 dark:border-slate-600 text-slate-900 dark:text-slate-100 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              {optionsArrayX.map((index) => (
                <option key={index} value={index}>Dimension {index}</option>
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
        <div className="space-y-5">
            {/* DATA CASE SELECTION */}
            <div className="bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 p-5 shadow-md">
              <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-4 pb-3 border-b border-slate-200 dark:border-slate-700">
                Data Case Selection
              </h3>
              <div className="space-y-4">
                <div className="space-y-2">
                  <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">
                    Select Case
                  </label>
                  <select
                    value={selectedCase}
                    onChange={handleSelectCase}
                    onClick={handleSelectCase}
                    className="w-full bg-white dark:bg-slate-700 border border-slate-300 dark:border-slate-600 text-slate-900 dark:text-slate-100 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="Polynomial">Polynomial</option>
                    <option value="Housing">Housing</option>
                    {fileContent && <option value="Custom">Custom</option>}
                  </select>
                </div>

                {/* File Upload */}
                <div className="space-y-2">
                  <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">
                    Import Custom Dataset
                  </label>
                  <div className="relative">
                    <input
                      type="file"
                      id="file-upload"
                      className="sr-only"
                      onChange={handleFileChange}
                      accept=".csv"
                    />
                    <label
                      htmlFor="file-upload"
                      className="flex items-center justify-center w-full px-4 py-3 border-2 border-dashed border-slate-300 dark:border-slate-600 rounded-lg cursor-pointer hover:border-blue-500 dark:hover:border-blue-500 hover:bg-slate-50 dark:hover:bg-slate-700/50 transition-all duration-200"
                    >
                      <div className="text-center">
                        <svg className="mx-auto h-8 w-8 text-slate-400 dark:text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                        </svg>
                        <p className="mt-2 text-sm text-slate-600 dark:text-slate-400">
                          {file ? (
                            <span className="text-blue-600 dark:text-blue-400 font-medium">{file.name}</span>
                          ) : (
                            <>
                              <span className="font-medium text-blue-600 dark:text-blue-400">Click to upload</span>
                              <span className="text-slate-500 dark:text-slate-500"> CSV file</span>
                            </>
                          )}
                        </p>
                      </div>
                    </label>
                  </div>
                </div>
              </div>
            </div>

            {/* REGRESSION MODEL */}
            <div className="bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 p-5 shadow-md">
                <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-4 pb-3 border-b border-slate-200 dark:border-slate-700">
                  Regression Model
                </h3>
                <div className="space-y-4">
                    <div className="space-y-2">
                      <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">
                        Select Model
                      </label>
                      <select
                        value={selectedModReg}
                        onChange={handleSelectReg}
                        onClick={handleSelectReg}
                        className="w-full bg-white dark:bg-slate-700 border border-slate-300 dark:border-slate-600 text-slate-900 dark:text-slate-100 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      >
                        <option value="Linear Polynomial">Linear Polynomial</option>
                        <option value="ElasticNet">Elastic Net</option>
                        <option value="SVR">SVR</option>
                        <option value="RandomForest">Random Forest</option>
                      </select>
                    </div>
                    <div className="grid grid-cols-2 gap-3">
                        <button className="px-4 py-2 bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 text-slate-700 dark:text-slate-200 rounded-md font-medium text-sm transition-colors">
                          Parameters
                        </button>
                        <button
                          onClick={handleFit}
                          className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md font-medium text-sm transition-colors"
                        >
                          Fit Model
                        </button>
                    </div>
                </div>
            </div>

            {/* BAYESIAN MODEL */}
            <div className="bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 p-5 shadow-md">
                <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-4 pb-3 border-b border-slate-200 dark:border-slate-700">
                  Bayesian Model
                </h3>
                <div className="space-y-4">
                  <div className="space-y-2">
                    <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">
                      Parameter Dimension
                    </label>
                    <DimSelectFieldX selectDimField={selectedDimM} setterDimField={setSelectedDimM} dimChain={dimChain}/>
                  </div>

                  <div className="space-y-2">
                    <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">
                      Prior Distribution
                    </label>
                    <select
                      value={selectedDistType}
                      onChange={(e) => {setSelectedDistType(e.target.value)}}
                      className="w-full bg-white dark:bg-slate-700 border border-slate-300 dark:border-slate-600 text-slate-900 dark:text-slate-100 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    >
                      <option value="Normal">Normal</option>
                      <option value="Uniform">Uniform</option>
                      <option value="Half-Normal">Half-Normal</option>
                    </select>
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    <div className="space-y-2">
                      <label className="block text-xs font-medium text-slate-600 dark:text-slate-400">
                        Low Value
                      </label>
                      <input
                        type="text"
                        placeholder="Low"
                        value={paramMLow}
                        onChange={(e) => {setParamMLow(e.target.value)}}
                        className="w-full bg-white dark:bg-slate-700 border border-slate-300 dark:border-slate-600 text-slate-900 dark:text-slate-100 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="block text-xs font-medium text-slate-600 dark:text-slate-400">
                        High Value
                      </label>
                      <input
                        type="text"
                        placeholder="High"
                        value={paramMHigh}
                        onChange={(e) => {setParamMHigh(e.target.value)}}
                        className="w-full bg-white dark:bg-slate-700 border border-slate-300 dark:border-slate-600 text-slate-900 dark:text-slate-100 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      />
                    </div>
                  </div>

                  <button
                    onClick={() => {handleSelectDist();}}
                    className="w-full px-4 py-2 bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 text-slate-700 dark:text-slate-200 rounded-md font-medium text-sm transition-colors"
                  >
                    Update Prior
                  </button>
                </div>
            </div>

            {/* INFERENCE CONFIGURATION */}
            <div className="bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 p-5 shadow-md">
                <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-4 pb-3 border-b border-slate-200 dark:border-slate-700">
                  Inference Configuration
                </h3>
                <div className="space-y-4">
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
                </div>
              </div>

              {/* ACTION BUTTONS */}
              <div className="space-y-3">
                <button
                  onClick={handleCompute}
                  disabled={isComputing}
                  className="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-2.5 px-4 rounded-md transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-blue-600"
                >
                  {isComputing ? (
                    <span className="flex items-center justify-center">
                      <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
                      </svg>
                      Computing...
                    </span>
                  ) : 'Compute'}
                </button>

                {isComputing && (
                  <div className="bg-slate-50 dark:bg-slate-700/50 rounded-lg p-4 border border-slate-200 dark:border-slate-600">
                    <div className="flex items-center justify-between mb-2">
                      <p className="text-sm font-medium text-slate-700 dark:text-slate-200">
                        {computeStatus}
                      </p>
                      <span className="text-xs font-semibold text-blue-600 dark:text-blue-400">
                        {computeProgress}%
                      </span>
                    </div>
                    <div className="w-full bg-slate-200 dark:bg-slate-600 rounded-full h-2 overflow-hidden">
                      <div
                        className="bg-gradient-to-r from-blue-500 to-blue-400 h-full rounded-full transition-all duration-300 ease-out"
                        style={{ width: `${computeProgress}%` }}
                      />
                    </div>
                  </div>
                )}

                <button
                  onClick={handleExport}
                  className="w-full bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 text-slate-700 dark:text-slate-200 font-medium py-2.5 px-4 rounded-md transition-colors duration-200"
                >
                  Export to CSV
                </button>
              </div>
          </div>
    )

}

