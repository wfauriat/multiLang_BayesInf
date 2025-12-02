export default function ControlPad({selectedDimR, setSelectedDimR,
                                    selectedDim1, setSelectedDim1,
                                    selectedDim2, setSelectedDim2, dimChain, dimX}) {
    const optionsArrayR = Array.from({ length: dimChain }, (v, i) => parseInt(i));
    const optionsArrayX = Array.from({ length: dimX }, (v, i) => parseInt(i));

    const DimSelectFieldR = ({selectDimField, setterDimField}) => {
      return (
              <select
                value={selectDimField}
                onChange={(e) => setterDimField(parseInt(e.target.value))}
                className="w-full bg-white dark:bg-slate-700 border border-slate-300 dark:border-slate-600 text-slate-900 dark:text-slate-100 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                {optionsArrayR.map((index) => (
                  <option key={index} value={index}>Dimension {index}</option>
                  ))}
              </select>
      );
    }

    const DimSelectFieldX = ({selectDimField, setterDimField}) => {
      return (
              <select
                value={selectDimField}
                onChange={(e) => setterDimField(parseInt(e.target.value))}
                className="w-full bg-white dark:bg-slate-700 border border-slate-300 dark:border-slate-600 text-slate-900 dark:text-slate-100 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                {optionsArrayX.map((index) => (
                  <option key={index} value={index}>Dimension {index}</option>
                  ))}
              </select>
      );
    }

    return (
        <div className="bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 p-6 shadow-md">
            <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-5 pb-3 border-b border-slate-200 dark:border-slate-700">
              Visualization Controls
            </h2>

            <div className="grid grid-cols-3 gap-6">
              <div className="space-y-2">
                <label className="flex items-center gap-2 text-sm font-medium text-slate-700 dark:text-slate-300">
                  <svg className="w-4 h-4 text-slate-400 dark:text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01" />
                  </svg>
                  Display Dimension
                </label>
                <DimSelectFieldX selectDimField={selectedDimR} setterDimField={setSelectedDimR}/>
              </div>

              <div className="space-y-2">
                <label className="flex items-center gap-2 text-sm font-medium text-slate-700 dark:text-slate-300">
                  <span className="flex items-center justify-center w-5 h-5 bg-blue-500/20 text-blue-600 dark:text-blue-400 rounded text-xs font-bold">
                    1
                  </span>
                  Chain Dimension 1
                </label>
                <DimSelectFieldR selectDimField={selectedDim1} setterDimField={setSelectedDim1}/>
              </div>

              <div className="space-y-2">
                <label className="flex items-center gap-2 text-sm font-medium text-slate-700 dark:text-slate-300">
                  <span className="flex items-center justify-center w-5 h-5 bg-green-500/20 text-green-600 dark:text-green-400 rounded text-xs font-bold">
                    2
                  </span>
                  Chain Dimension 2
                </label>
                <DimSelectFieldR selectDimField={selectedDim2} setterDimField={setSelectedDim2}/>
              </div>
            </div>
        </div>
    )
}