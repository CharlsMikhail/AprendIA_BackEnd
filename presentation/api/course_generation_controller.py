from flask import Blueprint, request, jsonify
from application.course_pipeline import CoursePipeline
from application.auth_service import AuthService

course_gen_bp = Blueprint('course_gen', __name__, url_prefix='/courses')
course_pipeline = CoursePipeline()
auth_service = AuthService()

@course_gen_bp.route("/generate", methods=["POST"])
def generate_course():
    """
    Inicia la generación de un curso asíncrono (Fase 1: Full Mock)
    """
    try:
        data = request.json
        prompt = data.get("prompt", "").strip()

        if not prompt:
            return jsonify({"error": "El prompt no puede estar vacío"}), 400

        # TODO: Auth middleware validation instead of this direct call
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
