import os
import jwt
import logging
from functools import wraps
from flask import request, jsonify, g

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "aprendia_super_secret_key_change_in_production")
JWT_ALGORITHM = "HS256"


def require_jwt(f):
    """
    Decorador que protege un endpoint con JWT Bearer Token.
    Decodifica el token e inyecta el user_id en flask.g.current_user_id.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get("Authorization")

        if not auth_header or not auth_header.startswith("Bearer "):
            return jsonify({"error": "Unauthorized", "message": "Missing or invalid Authorization header"}), 401

        token = auth_header.split(" ")[1]

        try:
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            g.current_user_id = payload.get("user_id")
            if not g.current_user_id:
                return jsonify({"error": "Unauthorized", "message": "Invalid token payload"}), 401
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Unauthorized", "message": "Token has expired"}), 401
        except jwt.InvalidTokenError as e:
            logging.warning(f"JWT inválido: {e}")
            return jsonify({"error": "Unauthorized", "message": "Invalid token"}), 401

        return f(*args, **kwargs)
    return decorated_function


def require_colab_token(f):
    """
    Decorador para asegurar que la petición entrante viene autorizada desde Colab.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        expected_token = os.getenv("COLAB_SECRET_TOKEN", "AprendiaSecret2026")
        auth_header = request.headers.get("Authorization")

        if not auth_header or not auth_header.startswith("Bearer "):
            return jsonify({"error": "Unauthorized", "message": "Missing or invalid Authorization header"}), 401

        token = auth_header.split(" ")[1]

        if token != expected_token:
            return jsonify({"error": "Unauthorized", "message": "Invalid token"}), 401

        return f(*args, **kwargs)
    return decorated_function
