export const handlePost = async (val, key, endpoint) => {
    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ [key] : val }),
      });
      if (response.ok) {
        console.log("POST SUCCESSFUL", val)
      }
    } catch (error) {
      console.log(error.message);
    }
  };

export function ConfigField({ label, value, onChange, onSend, endpoint }) {
  return (
    <div>
      <label style={{ minWidth: "80px", display: "inline-block"}}>
          {label}
      </label>
      <input 
          type="text" 
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={label}
          style={{width: "100px", display: "inline-block"}}
          required
      />
      <button 
          onClick={() => onSend(parseFloat(value), label, endpoint)}>
          Send
      </button>
    </div>
  );
}

