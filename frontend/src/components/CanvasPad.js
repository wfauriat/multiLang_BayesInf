// import { useState } from 'react';
import MatrixPlot from './MatrixPlot';
import styles from './CanvasPad.module.css'

export default function CanvasPad({chainData, selectedDimR}) {

    return (
    <div className={styles.CanvasPad}>
        <h2>Canvas Pad</h2>
        <div className={styles.CanvasView}>
            {chainData && <MatrixPlot chainData={chainData} selectedDimR={selectedDimR}/>}
        </div>
    </div>
    )
}