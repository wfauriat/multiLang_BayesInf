# Bayesian Inference demonstrator App (Web and Desktop)

## Full-stack Web App

- Bayesian Inference core (core in **Python**, and some small attemps in C++ (not yet implemented because the gain w/r to numpy/python is not significant)),
    - In the style of **probabilistic programming paradigm**, i.e. with objects for random variables (observed or priors)
- Flask (**Python**) Back-end
- React (**Javascript**) Front-end

> Run using the script "*run_local_server.sh*" with a python virtual env that includes *gunicorn* (see requirements.txt) 



### Detailed install from git repo (to run locally both the front-end web UI and the back-end server)

``` bash
gh repo clone wfauriat/BayesInfApp          ## Clone the repo
cd BayesInfApp                       
python3 -m venv .venv                       ## Create python virtual env
source .venv/bin/activate                   ## Activate the virtual env
pip install -r requirements.txt             ## Install required python dependencies locally
cd frontend                                 
npm install                                 ## Install the front-end dependencies (npm must be installed)
npm run build                               ## Build the React front-end (and using CreateReactApp environnement)
```

Then run :

``` bash
cd ..                                       ## Move back to app root directory
chmod u+rx run_local_server.sh              ## Authorize script execution locally
./run_local_server.sh                       ## Run the gunicorn server (back-end API and serving front-end)
## Or run directly (with virtual env activated) : gunicorn -b 127.0.0.1:5000 Flask_app:app
```

> Open front-end (and API server) at endpoint http://127.0.0.1:5000

Also, hosted on Render @ https://bi-webapp.onrender.com/

---

## Desktop App

*(also enclosed in the repository)*

- Same **Python** Core (as above)
- GUI in PyQt5

> Run "python Qt_App.py" (in a virtual env that includes PyQt5 : not in requirement.txt, must be added)

``` bash
gh repo clone wfauriat/BayesInfApp   
cd BayesInfApp                       
python3 -m venv .venv                       
source .venv/bin/activate                   
pip install -r requirements.txt 

pip install PyQt5

python Qt_app.py
```

---

### *Developped from mid to end 2025 (W. Fauriat)* :
#### Bayesian Inference in Python (using only numpy for core computation) + PyQt GUI + Flask-React full-stack web app

