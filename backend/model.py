from flask import Blueprint, jsonify, request

from UIcomps.componentsGUI import ModelUI

from pyBI.inference import MHalgo

model = ModelUI()

bp_inf = Blueprint('inf', __name__, url_prefix='/inf')
bp_case = Blueprint('case', __name__, url_prefix='/case')
bp_visu = Blueprint('visu', __name__, url_prefix='/visu')
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
                         'LLsort': model.LLsort.tolist()}), 200
    except Exception:
        return jsonify({'message': 'Computation has not been made' }), 200

    
