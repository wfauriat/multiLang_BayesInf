import styles from './ControlPad.module.css'
// import { v4 as uuidv4 } from 'uuid';

export default function ControlPad({selectedDimR, setSelectedDimR, dimChain}) {
    const optionsArray = Array.from({ length: dimChain }, (v, i) => parseInt(i));
    // console.log(optionsArray)

    return (
        <div className={styles.ControlPad}>
            <h2>Control Pad</h2>
            <div>
              Dimension selection
              <select value={selectedDimR} 
                onChange={(e) => setSelectedDimR(parseInt(e.target.value))}
                style={{width:"30%", display:"inline"}}>
                {optionsArray.map((index) => (
                  <option value={index}>{index}</option>
                  ))}
              </select>
            </div>
        </div>
    )
}