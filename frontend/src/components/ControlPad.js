import styles from './ControlPad.module.css'

export default function ControlPad({selectedDimR, setSelectedDimR, 
                                    selectedDim1, setSelectedDim1,
                                    selectedDim2, setSelectedDim2, dimChain}) {
    const optionsArray = Array.from({ length: dimChain }, (v, i) => parseInt(i));
    // console.log(optionsArray)

    const DimSelectField = ({selectDimField, setterDimField}) => {
      return (
              <select value={selectDimField} 
                onChange={(e) => setterDimField(parseInt(e.target.value))}
                style={{width:"80px", display:"inline", marginLeft:"1em"}}>
                {optionsArray.map((index) => (
                  <option value={index}>{index}</option>
                  ))}
              </select>
      );
    }

    return (
        <div className={styles.ControlPad}>
            <h2>Control Pad</h2>
            <div>
              <p style={{display:"inline-block", width:"150px", margin:"0.5em 0"}}>Dimension selection</p>
              <DimSelectField selectDimField={selectedDimR} setterDimField={setSelectedDimR}/>
            </div>
            <div>
              <p style={{display:"inline-block", width:"150px", margin:"0.5em 0"}}>Chain 1 selection</p>
              <DimSelectField selectDimField={selectedDim1} setterDimField={setSelectedDim1}/>
            </div>
            <div>
              <p style={{display:"inline-block", width:"150px", margin:"0.5em 0"}}>Chain 2 selection</p>
              <DimSelectField selectDimField={selectedDim2} setterDimField={setSelectedDim2}/>
            </div>
        </div>
    )
}