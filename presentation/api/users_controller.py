from flask import Blueprint, request, jsonify

users_bp = Blueprint('users', __name__, url_prefix='/users')

@users_bp.route('/complete-profile', methods=['PUT'])
def complete_profile():
    """
    Finaliza el registro guardando los datos adicionales solicitados en el onboarding.
    """
    data = request.json
    # TODO: Auth middleware validation
    # TODO: Update DB
    return jsonify({
        "message": "Perfil completado exitosamente"
    }), 200

@users_bp.route('/me', methods=['GET'])
def get_me():
    """
    Recupera los datos completos del usuario autenticado.
    """
    # TODO: Auth middleware validation
    # TODO: Select from DB
    return jsonify({
        "id_usuario": 1,
        "nombre_usuario": "GrokMaster99",
        "correo": "usuario@gmail.com",
        "nombre_completo": "Juan Pérez",
        "foto_perfil_url": "https://lh3.googleusercontent.com/...",
        "esta_activo": True
    }), 200
