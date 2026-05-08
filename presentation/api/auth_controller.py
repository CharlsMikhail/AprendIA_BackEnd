from flask import Blueprint, request, jsonify, g
from application.auth_service import AuthService
from infrastructure.db.user_repository import UserRepository
from presentation.middlewares.auth_middleware import require_jwt
import logging

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')
infra_bp = Blueprint('infrastructure', __name__)

auth_service = AuthService()
user_repo = UserRepository()


@auth_bp.route('/google', methods=['POST'])
def google_auth():
    """
    Login con Google OAuth.
    Recibe el id_token emitido por Google Sign-In del frontend,
    lo valida con Google, y devuelve un JWT de AprendIA.
    """
    data = request.json or {}
    google_id_token = data.get('id_token') or data.get('google_token')

    if not google_id_token:
        return jsonify({"error": "id_token is required"}), 400

    try:
        result = auth_service.login_with_google_token(google_id_token)
        return jsonify(result), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 401
    except Exception as e:
        logging.error(f"Error en /auth/google: {e}", exc_info=True)
        return jsonify({"error": "Internal server error during authentication"}), 500


@auth_bp.route('/logout', methods=['POST'])
@require_jwt
def logout():
    """
    Cerrar sesión.
    El token JWT es stateless: el cliente debe eliminarlo de su almacenamiento local.
    Este endpoint solo confirma que la operación fue recibida.
    """
    return jsonify({"message": "Sesión cerrada exitosamente"}), 200


@auth_bp.route('/me', methods=['GET'])
@require_jwt
def get_me():
    """
    Obtener usuario autenticado.
    Requiere: Authorization: Bearer <token>
    """
    user_id = g.current_user_id
    user = user_repo.get_user_by_id(user_id)

    if not user:
        return jsonify({"error": "Usuario no encontrado"}), 404

    return jsonify({
        "id": user.id,
        "google_id": user.google_id,
        "name": user.name,
        "email": user.email,
        "picture": user.picture,
        "created_at": user.created_at.isoformat() if user.created_at else None
    }), 200


@infra_bp.route('/health', methods=['GET'])
def healthcheck():
    """Verifica estado del servidor y DB."""
    from infrastructure.db.database import DatabaseConnection
    try:
        db = DatabaseConnection()
        db.execute_query("SELECT 1")
        db_status = "connected"
    except Exception:
        db_status = "error"

    return jsonify({
        "status": "healthy" if db_status == "connected" else "degraded",
        "api_version": "2.0.0",
        "database": db_status
    }), 200
