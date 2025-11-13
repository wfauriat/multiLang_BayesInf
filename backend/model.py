from flask import Blueprint, jsonify, request

from UIcomps.componentsGUI import ModelUI

from pyBI.inference import MHalgo

model = ModelUI()

bp_inf = Blueprint('inf', __name__, url_prefix='/inf')
bp_comp = Blueprint('comp', __name__, url_prefix='') 

@bp_inf.route('/NMCMC', methods=['GET'])
def get_NMCMC():
    return jsonify({'NMCMC': model.NMCMC}), 200

@bp_inf.route('/NMCMC', methods=['POST'])
def set_NMCMC():
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400
    data = request.get_json()
    print(data)
    NMCMC = data.get('NMCMC')
    model.NMCMC = NMCMC
    return jsonify({'NMCMC': model.NMCMC}), 200


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
    return jsonify({'chain': model.MCalgo.cut_chain.tolist()}), 200

