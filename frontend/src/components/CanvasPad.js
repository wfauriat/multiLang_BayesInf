import { useState } from 'react';
import SingleChainPlot from './SingleChainPlot';
import ChainScatterPlot from './ChainScatterPlot';
import PostPredPlot from './PostPredPlot';
import styles from './CanvasPad.module.css'

export default function CanvasPad({chainData, selectedDimR, selectedDim1, selectedDim2,
                                    MCsortData, LLsortData,
                                    xmes, yobs, postY, postYeps}) {

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
            // return <h2>Hello</h2>
            return chainData && <PostPredPlot xmes={xmes} yobs={yobs} postY={postY} postYeps={postYeps}
                                    selectedDimR={selectedDimR} />
        case 1:
            return chainData && <ChainScatterPlot
                                selectedDim1={selectedDim1} selectedDim2={selectedDim2}
                                MCsortData={MCsortData} LLsortData={LLsortData}/>
        case 2:
            return <h2>Hello 2</h2>;
        case 3:
            return chainData && <SingleChainPlot chainData={chainData} selectedDim1={selectedDim1}/>
        case 4:
            return <h2>Hello 4</h2>;
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

