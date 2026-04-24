from functools import wraps
from flask import request, jsonify
import os

def require_colab_token(f):
    """
    Decorador para asegurar que la petición entrante viene autorizada desde Colab.
    Verifica que el header Authorization tenga el Bearer token correcto.
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
