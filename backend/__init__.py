from flask import Flask

def create_app():
    app = Flask(__name__)

    from . import model
    app.register_blueprint(model.bp_inf)
    app.register_blueprint(model.bp_comp)

    @app.route('/')
    def home():
        return "Hello, BI_Flask!\n"
    
    return app