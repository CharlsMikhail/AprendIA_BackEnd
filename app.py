from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO
from dotenv import load_dotenv
import os

# Cargar variables de entorno
load_dotenv()

from presentation.extensions import socketio

def create_app():
    app = Flask(__name__)
    CORS(app)
    
    # Necesario para las sesiones de Flask requeridas por OAuth2
    app.secret_key = os.getenv("FLASK_SECRET_KEY", "super_secret_default_key")

    # Ya no usamos api_bp, todo está distribuido en controladores de la carpeta api/
    from presentation.api.colab_webhooks_controller import colab_bp
    app.register_blueprint(colab_bp)

    from presentation.api.auth_controller import auth_bp, infra_bp
    app.register_blueprint(auth_bp)
    app.register_blueprint(infra_bp)

    from presentation.api.users_controller import users_bp
    app.register_blueprint(users_bp)

    from presentation.api.course_controller import courses_bp, community_bp
    app.register_blueprint(courses_bp)
    app.register_blueprint(community_bp)

    from presentation.api.course_generation_controller import course_gen_bp
    app.register_blueprint(course_gen_bp)

    # Conectar la app con SocketIO
    socketio.init_app(app)

    return app

if __name__ == "__main__":
    app = create_app()
    # Usar socketio.run en lugar de app.run para soportar WebSockets
    socketio.run(app, debug=True, port=5000)