import styles from './ControlPad.module.css'

export default function ControlPad({selectedDimR, setSelectedDimR}) {
            
    return (
        <div className={styles.ControlPad}>
            <h2>Control Pad</h2>
            <div>
              Dimension selection
              <select value={selectedDimR} 
                onChange={(e) => setSelectedDimR(parseInt(e.target.value))}
                style={{width:"30%", display:"inline"}}>
                <option value={0}>0</option>
                <option value={1}>1</option>
                <option value={2}>2</option>
                <option value={3}>3</option>
              </select>
            </div>
        </div>
    )
}