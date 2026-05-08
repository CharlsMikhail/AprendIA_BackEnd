from flask import Blueprint, request, jsonify, g
from infrastructure.db.user_repository import UserRepository
from presentation.middlewares.auth_middleware import require_jwt
import logging

users_bp = Blueprint('users', __name__, url_prefix='/users')
user_repo = UserRepository()


@users_bp.route('/me/profile', methods=['GET'])
@require_jwt
def get_profile():
    """
    Obtener el perfil del usuario autenticado.
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


@users_bp.route('/me/profile', methods=['PATCH'])
@require_jwt
def update_profile():
    """
    Actualiza el perfil del usuario (ej: nombre de usuario, teléfono).
    """
    user_id = g.current_user_id
    data = request.json or {}

    try:
        updated = user_repo.update_user_profile(user_id, data)
        if updated:
            return jsonify({"message": "Perfil actualizado exitosamente"}), 200
        else:
            return jsonify({"message": "No se enviaron campos válidos para actualizar"}), 400
    except Exception as e:
        return jsonify({"error": "Error al actualizar el perfil", "detail": str(e)}), 500


@users_bp.route('/me/stats', methods=['GET'], strict_slashes=False)
@require_jwt
def get_user_stats():
    """
    Obtener estadísticas del usuario.
    """
    user_id = g.current_user_id
    stats = user_repo.get_user_stats(user_id)
    return jsonify(stats), 200


@users_bp.route('/me/courses/recent', methods=['GET'], strict_slashes=False)
@require_jwt
def get_recent_courses():
    """
    Obtiene los cursos recientemente completados por el usuario.
    """
    user_id = g.current_user_id
    limit = request.args.get('limit', 5, type=int)
    courses = user_repo.get_recent_courses(user_id, limit=limit)
    return jsonify({"data": courses}), 200


@users_bp.route('/me/courses', methods=['GET'], strict_slashes=False)
@require_jwt
def list_user_courses():
    """
    Lista todos los cursos del usuario con filtros y paginación.
    Query Params: ?visibility=all|completed|failed&sort=recent|rating&q=texto&page=1&limit=10
    """
    user_id = g.current_user_id
    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 10, type=int)
    status_filter = request.args.get('visibility', 'all')
    sort = request.args.get('sort', 'recent')
    q = request.args.get('q', None)

    result = user_repo.get_user_courses(
        user_id,
        page=page,
        limit=limit,
        status_filter=status_filter,
        sort=sort,
        q=q
    )
    return jsonify(result), 200
