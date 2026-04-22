from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
import os

# Cargar variables de entorno
load_dotenv()

def create_app():
    app = Flask(__name__)
    CORS(app)
    
    # Necesario para las sesiones de Flask requeridas por OAuth2
    app.secret_key = os.getenv("FLASK_SECRET_KEY", "super_secret_default_key")

    # Registrar el blueprint de la API
    from presentation.routes import api_bp
    app.register_blueprint(api_bp)

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, port=5000)