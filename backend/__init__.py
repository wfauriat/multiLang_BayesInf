from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    # CORS(app) # proxy can be used on the front side to prevent CORS issue

    from . import model
    app.register_blueprint(model.bp_inf)
    app.register_blueprint(model.bp_comp)

    @app.route('/')
    def home():
        return "Hello, BI_Flask!\n"
    
    return app