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

