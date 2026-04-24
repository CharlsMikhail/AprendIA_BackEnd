from flask import Blueprint, request, jsonify

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')
infra_bp = Blueprint('infrastructure', __name__)

@auth_bp.route('/google', methods=['POST'])
def google_auth():
    """
    Inicia sesión o registra un usuario parcialmente con el token de Google.
    """
    data = request.json
    google_token = data.get('google_token')
    
    if not google_token:
        return jsonify({"error": "google_token is required"}), 400
        
    # TODO: Validar google_token y upsert en DB
    return jsonify({
        "access_token": "mock-access-token",
        "refresh_token": "mock-refresh-token",
        "needs_onboarding": True
    }), 200

@auth_bp.route('/refresh', methods=['POST'])
def refresh_token():
    """
    Renueva un access_token expirado.
    """
    # TODO: Validar refresh token de headers o cookies
    return jsonify({
        "access_token": "nuevo-mock-access-token"
    }), 200

@infra_bp.route('/health', methods=['GET'])
def healthcheck():
    """
    Verifica estado del servidor y DB.
    """
    # TODO: Ping real a DB
    return jsonify({
        "status": "healthy",
        "api_version": "1.0.0",
        "database": "connected"
    }), 200
