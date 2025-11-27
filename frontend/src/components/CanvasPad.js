import { useState } from 'react';
import SingleChainPlot from './SingleChainPlot';
import ChainScatterPlot from './ChainScatterPlot';
import PostPredPlot from './PostPredPlot';
import ChainDistPlot from './ChainDistPlot';
import ErrorFitPlot from './ErrorFitPlot';
import styles from './CanvasPad.module.css'

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
    <div className={styles.CanvasPad}>
        <div className={styles.containerTabs}>
            {tabs.map((tab, index) => (
                <button
                    key={index}
                    onClick={()=>setActiveTab(index)}
                    className={`${styles.tabButton}
                                ${activeTab === index ? styles.tabButtonActive : ''}`}
                    >{tab.label}
                </button>))}
        </div>
        <div className={styles.CanvasView}>
            {renderTabContent()}
        </div>
    </div>
    )
}

