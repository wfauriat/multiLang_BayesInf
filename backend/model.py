from flask import Blueprint, jsonify, request

from UIcomps.componentsGUI import ModelUI

from pyBI.inference import MHalgo
from pyBI.base import UnifVar, NormVar, HalfNormVar

import numpy as np
from sklearn.model_selection import train_test_split

model = ModelUI()
model.currentM = 0
model.currentDistType = "Normal"

bp_inf = Blueprint('inf', __name__, url_prefix='/inf')
bp_case = Blueprint('case', __name__, url_prefix='/case')
bp_visu = Blueprint('visu', __name__, url_prefix='/visu')
bp_regr = Blueprint('regr', __name__, url_prefix='/regr')
bp_modelBayes = Blueprint('modelBayes', __name__, url_prefix='/modelBayes')
bp_comp = Blueprint('comp', __name__, url_prefix='') 



@bp_inf.route('/<string:Nparam>', methods=['GET'])
def get_NParam(Nparam):
    value = getattr(model, Nparam)
    if value is None:
        return jsonify({"error": f"Configuration parameter '{Nparam}' not found."}), 404
    return jsonify({Nparam: value}), 200

@bp_inf.route('/<string:Nparam>', methods=['POST'])
def set_NParam(Nparam):
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400
    data = request.get_json()
    print(data)
    value = data.get(Nparam)
    if value is None:
        return jsonify({"error": f"Parameter '{Nparam}' not provided in request"}), 400
    if not hasattr(model, Nparam):
        return jsonify({"error": f"Configuration parameter '{Nparam}' not found on model"}), 404
    setattr(model, Nparam, value)
    return jsonify({Nparam: getattr(model, Nparam)}), 200


@bp_case.route('/select', methods=['POST'])
def handle_select_case():
    data = request.get_json()
    selected_item = data.get('selectedItem')

    print(f"Received: {selected_item}")
    model.data_selected_case = selected_item
    model.load_case()
    
    return jsonify({'message': 'Success', 'received': selected_item}), 200

@bp_case.route('/customData', methods=['POST'])
def handle_select_custom_case():
    data = request.get_json()
    input_data = data.get('data')

    # print(f"Received: {input_data[0:2]}")
    
    XX = np.array(input_data)[:,:-1]
    yy = np.array(input_data)[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(
        XX, yy, test_size=0.95, random_state=42)
    model.custom_case.xmes = X_train
    model.custom_case.ymes = y_train
    model.load_case()
    
    return jsonify({'message': 'Success'}), 200

@bp_case.route('/select', methods=['GET'])
def get_select_case():
    value = getattr(model, "data_selected_case")
    return jsonify({'message': 'Success', 'selected_case': value}), 200

@bp_case.route('/dimChain', methods=['GET'])
def get_dimChain():
    value = getattr(model, "bstart")
    return jsonify({'message': 'Success', 'dimChain': value.shape[0]}), 200

@bp_modelBayes.route('/select', methods=['POST'])
def handle_select_dist():
    data = request.get_json()
    selected_dist = data.get('selected_dist')
    paramMLow = data.get('paramMLow')
    paramMHigh = data.get('paramMHigh')
    model.currentDistType = selected_dist
    if model.currentM < len(model.rndUs):
        if model.currentDistType == "Normal":
            model.rndUs[model.currentM] = NormVar([paramMLow,paramMHigh])
        elif model.currentDistType == "Uniform":
            model.rndUs[model.currentM] = UnifVar([paramMLow,paramMHigh])
        elif model.currentDistType == "Half-Normal":
            model.rndUs[model.currentM] = HalfNormVar(paramMHigh)
    else: model.rnds = HalfNormVar(paramMHigh)

    print(f"Received: {selected_dist, paramMLow, paramMHigh}")   
    return jsonify({'message': 'Success', 'received': selected_dist}), 200

@bp_modelBayes.route('/current', methods=['POST'])
def handle_select_currentM():
    data = request.get_json()
    selected_item = data.get('selectedItem')
    model.currentM = selected_item
    if model.currentM < len(model.rndUs):
        if isinstance(model.rndUs[model.currentM], NormVar):
            model.currentDistType = "Normal"
        elif isinstance(model.rndUs[model.currentM], UnifVar):
            model.currentDistType = "Uniform"
        elif isinstance(model.rndUs[model.currentM], HalfNormVar):
            model.currentDistType = "Half-Normal"
    else: model.currentDistType = "Half-Normal"
    print(f"Received: {selected_item}")   
    return jsonify({'message': 'Success', 'received': selected_item}), 200

@bp_modelBayes.route('/current', methods=['GET'])
def get_select_currentM():
    return jsonify({'message': 'Success', 'currentM': model.currentM})

@bp_modelBayes.route('/select', methods=['GET'])
def get_select_handle_select_dist():
    return jsonify({'message': 'Success', 'distType': model.currentDistType})

@bp_modelBayes.route('/paramM', methods=['GET'])
def get_MParam():
    if model.currentM > len(model.rndUs)-1:
        value = getattr(model.rnds, "param")
        return jsonify({"param": value}), 200
    else:
        value = getattr(model.rndUs[model.currentM], "param")
        return jsonify({"lowM": value[0], "highM": value[1]}), 200


@bp_comp.route('/compute')
def compute():
    model.MCalgo = MHalgo(N=model.NMCMC,
                                    Nthin=model.Nthin,
                                    Nburn=model.Nburn,
                                    is_adaptive=True,
                                    verbose=model.verbose)
    model.MCalgo.initialize(model.obsvar,
                                     model.rndUs, 
                                     model.rnds)
    model.MCalgo.MCchain[0] = model.bstart
    model.MCalgo.state(0, set_state=True)
    model.MCalgo.runInference()
    model.regr_fit()
    model.postpar = model.MCalgo.cut_chain
    model.MCsort = model.MCalgo.idx_chain
    model.LLsort = model.MCalgo.cut_llchain[model.MCalgo.sorted_indices]
    model.post_treat_chains()
    return jsonify({'message': 'Computation Succeded'}), 200

@bp_comp.route('/results')
def get_chains():
    try: 
        return jsonify({'chains': model.MCalgo.cut_chain.tolist(),
                         'MCsort': model.MCsort.tolist(),
                         'LLsort': model.LLsort.tolist(),
                         'xmes': model.data_case.xmes.tolist(),
                         'obs': model.data_case.ymes.tolist(),
                         'postMAP': model.postMAP.tolist(),
                         'postY': model.postY.tolist(),
                         'postYeps': model.postYeps.tolist(),
                         'yregPred': model.yreg_pred.tolist()},
                         ), 200
    except Exception:
        return jsonify({'message': 'Computation has not been made' }), 200


@bp_regr.route('/reg_pred')
def get_regpred():
    try: 
        return jsonify({'yreg_pred': model.yreg_pred.tolist()}), 200
    except Exception:
        return jsonify({'message': 'Computation has not been made' }), 200
    
@bp_regr.route('/fit')
def fit_reg():
    model.regr_fit()
    return jsonify({'message': 'Regression performed' }), 200

@bp_regr.route('/select', methods=['POST'])
def handle_select_regr():
    data = request.get_json()
    selected_item = data.get('selectedItem')

    print(f"Received: {selected_item}")
    model.selected_model = selected_item
    # model.regr_fit()
    
    return jsonify({'message': 'Success', 'received': selected_item}), 200