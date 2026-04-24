from flask import Blueprint, request, jsonify, session
from application.auth_service import AuthService
from application.course_pipeline import CoursePipeline
from infrastructure.db.user_repository import UserRepository
import json
from dataclasses import asdict

api_bp = Blueprint('api', __name__)
auth_service = AuthService()
course_pipeline = CoursePipeline()
user_repo = UserRepository()

@api_bp.route("/auth/google/login")
def login():
    return auth_service.login()

@api_bp.route("/callback")
def callback():
    return auth_service.callback()

@api_bp.route("/auth/logout")
def logout():
    return auth_service.logout()

@api_bp.route("/user/history", methods=["GET"])
def get_history():
    user_id = auth_service.get_current_user()
    if not user_id:
        return jsonify({"error": "No autorizado"}), 401
    
    history = user_repo.get_user_history(user_id)
    return jsonify(history)

@api_bp.route("/solicitar_cursos", methods=["POST"])
def solicitar_cursos():
    try:
        data = request.json
        prompt = data.get("prompt", "").strip()

        if not prompt:
            return jsonify({"error": "El prompt no puede estar vacío"}), 400

        user_id = auth_service.get_current_user()

        # Inicia el proceso de forma asíncrona devolviendo el job_id
        job_id = course_pipeline.iniciar_generacion(prompt, user_id)

        return jsonify({
            "message": "Generación de curso en progreso",
            "job_id": job_id,
            "status": "processing"
        }), 202

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@api_bp.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "architecture": "clean"})
