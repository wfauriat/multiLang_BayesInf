from flask import Flask
from flask_cors import CORS
from flask import send_file, render_template


from . import model

import os

template_dir = os.path.abspath('./frontend/build') 
static_dir = os.path.abspath('./frontend/build/static')


def create_app():
    app = Flask(__name__,
    static_folder=static_dir,
    template_folder=template_dir)
    CORS(app) # proxy can be used on the front side to prevent CORS issue
    # put "proxy": "http://127.0.0.1:5000", in package.json (on the front size)

    app.register_blueprint(model.bp_inf)
    app.register_blueprint(model.bp_comp)
    app.register_blueprint(model.bp_case)
    app.register_blueprint(model.bp_visu)
    app.register_blueprint(model.bp_regr)
    app.register_blueprint(model.bp_modelBayes)

    # @app.route('/')
    # def home():
    #     return "Hello, BI_Flask!\n"
    
    @app.route('/', defaults={'path': ''})
    @app.route('/<path:path>')
    def serve_frontend(path):
        """Serve the main index.html file for any non-API route."""
        return render_template('index.html')
    
    return app