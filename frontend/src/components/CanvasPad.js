import { useState } from 'react';
import SingleChainPlot from './SingleChainPlot';
import ChainScatterPlot from './ChainScatterPlot';
import PostPredPlot from './PostPredPlot';
import ChainDistPlot from './ChainDistPlot';
import ErrorFitPlot from './ErrorFitPlot';

export default function CanvasPad({chainData, selectedDimR, selectedDim1, selectedDim2,
                                    MCsortData, LLsortData,
                                    xmes, yobs, postMAP, postY, postYeps,
                                    yregPred}) {

    const [activeTab, setActiveTab] = useState(0);

    const tabs = [
    { label: 'Posterior Prediction'},
    { label: 'Posterior Parameters'},
    { label: 'Fit / Error'},
    { label: 'Chains'},
    { label: 'Dist 1D'}
    ];

    const renderTabContent = () => {
    switch(activeTab) {
        case 0:
            return chainData && <PostPredPlot xmes={xmes} yobs={yobs} postMAP={postMAP}
                                    postY={postY} postYeps={postYeps} yregPred={yregPred}
                                    selectedDimR={selectedDimR} />
        case 1:
            return chainData && <ChainScatterPlot
                                selectedDim1={selectedDim1} selectedDim2={selectedDim2}
                                MCsortData={MCsortData} LLsortData={LLsortData}/>
        case 2:
            return chainData && <ErrorFitPlot xmes={xmes} yobs={yobs} postMAP={postMAP}
                                                yregPred={yregPred} selectedDimR={selectedDimR}/>
        case 3:
            return chainData && <SingleChainPlot chainData={chainData} selectedDim1={selectedDim1}/>
        case 4:
            return chainData && <ChainDistPlot chainData={chainData} selectedDim1={selectedDim1}/>
        default:
        return null;
    }
    };

    return (
    <div>
        {/* Tab Navigation */}
        <div className="bg-white dark:bg-slate-800 rounded-t-lg border border-slate-200 dark:border-slate-700 border-b-0">
          <nav className="flex space-x-1 p-1" aria-label="Tabs">
            {tabs.map((tab, index) => (
                <button
                    key={index}
                    onClick={()=>setActiveTab(index)}
                    className={`
                      flex-1 px-4 py-2.5 text-sm font-medium rounded-md transition-all duration-200
                      ${activeTab === index
                        ? 'bg-slate-100 dark:bg-slate-700 text-slate-900 dark:text-white shadow-sm'
                        : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-700/50'
                      }
                    `}
                >
                  {tab.label}
                </button>
            ))}
          </nav>
        </div>

        {/* Content Area */}
        <div className="bg-white dark:bg-slate-800 rounded-b-lg border border-slate-200 dark:border-slate-700 p-6 min-h-[500px]">
            {renderTabContent()}
        </div>
    </div>
    )
}

