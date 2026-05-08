from flask import Blueprint, request, jsonify, g
from application.course_pipeline import CoursePipeline
from infrastructure.db.course_job_repository import CourseJobRepository
from presentation.middlewares.auth_middleware import require_jwt
import logging

course_gen_bp = Blueprint('course_gen', __name__, url_prefix='/courses')

course_pipeline = CoursePipeline()
job_repo = CourseJobRepository()


# =============================================================================
# GENERACIÓN DE CURSOS
# =============================================================================

@course_gen_bp.route('/generate', methods=['POST'])
@require_jwt
def generate_course():
    """
    Inicia la generación de un curso con IA de forma asíncrona.
    Devuelve un job_id para monitorear el progreso via WebSocket.
    Body: { "prompt": "string" }
    """
    try:
        data = request.json or {}
        prompt = data.get("prompt", "").strip()

        if not prompt:
            return jsonify({"error": "El prompt no puede estar vacío"}), 400

        if len(prompt) > 500:
            return jsonify({"error": "El prompt no puede superar los 500 caracteres"}), 400

        job_id = course_pipeline.iniciar_generacion(prompt=prompt, user_id=g.current_user_id)

        return jsonify({
            "message": "Generación de curso en progreso",
            "job_id": job_id,
            "status": "processing",
            "websocket_room": f"job_{job_id}"
        }), 202

    except Exception as e:
        logging.error(f"Error iniciando generación: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@course_gen_bp.route('/generate/<string:job_id>/cancel', methods=['POST'])
@require_jwt
def cancel_generation(job_id: str):
    """
    Cancela una generación en curso.
    """
    job = job_repo.get_job(job_id)

    if not job:
        return jsonify({"error": "Job no encontrado"}), 404

    if job.user_id != g.current_user_id:
        return jsonify({"error": "No autorizado para cancelar este job"}), 403

    if job.status in ["completed", "failed"]:
        return jsonify({"error": f"El job ya terminó con status: {job.status}"}), 409

    job.status = "failed"
    job.error_message = "Cancelado por el usuario"
    job_repo.save_job(job)

    return jsonify({"message": "Generación cancelada exitosamente", "job_id": job_id}), 200


# =============================================================================
# PREVIEW DEL CURSO GENERADO
# =============================================================================

@course_gen_bp.route('/preview/<string:generation_id>', methods=['GET'])
@require_jwt
def get_preview(generation_id: str):
    """
    Obtiene el preview del curso generado.
    Solo disponible cuando el job está en status 'completed'.
    """
    job = job_repo.get_job(generation_id)

    if not job:
        return jsonify({"error": "Generación no encontrada"}), 404

    if job.user_id != g.current_user_id:
        return jsonify({"error": "No autorizado"}), 403

    if job.status == "processing":
        return jsonify({"message": "El curso aún está siendo generado", "status": "processing"}), 202

    if job.status == "failed":
        return jsonify({
            "error": "La generación falló",
            "detail": job.error_message,
            "status": "failed"
        }), 422

    # Construir el preview limpio (sin transcripciones completas)
    sections_preview = []
    for section in (job.sections_with_candidates or []):
        candidates_preview = []
        for c in section.get("candidates", []):
            candidates_preview.append({
                "title": c.get("title"),
                "url": c.get("url"),
                "score": c.get("score"),
                "views": c.get("views"),
                "likes": c.get("likes"),
                "sentiment": c.get("sentiment"),
                "has_transcript": bool(c.get("transcript")),
                "rag_validation": c.get("rag_validation")
            })
        sections_preview.append({
            "title": section.get("title"),
            "description": section.get("description"),
            "candidates": candidates_preview
        })

    return jsonify({
        "generation_id": job.job_id,
        "status": job.status,
        "course_outline": job.course_outline,
        "sections": sections_preview
    }), 200


@course_gen_bp.route('/preview/<string:generation_id>/save', methods=['POST'])
@require_jwt
def save_preview(generation_id: str):
    """
    Guarda el curso generado como curso permanente del usuario.
    """
    job = job_repo.get_job(generation_id)

    if not job:
        return jsonify({"error": "Generación no encontrada"}), 404

    if job.user_id != g.current_user_id:
        return jsonify({"error": "No autorizado"}), 403

    if job.status != "completed":
        return jsonify({"error": "Solo se pueden guardar generaciones completadas"}), 422

    return jsonify({
        "message": "Curso guardado exitosamente",
        "course_id": job.job_id,
        "title": (job.course_outline or {}).get("title", "Sin título")
    }), 201


@course_gen_bp.route('/preview/<string:generation_id>', methods=['DELETE'])
@require_jwt
def discard_preview(generation_id: str):
    """
    Descarta el preview de un curso generado.
    """
    job = job_repo.get_job(generation_id)

    if not job:
        return jsonify({"error": "Generación no encontrada"}), 404

    if job.user_id != g.current_user_id:
        return jsonify({"error": "No autorizado"}), 403

    job.status = "failed"
    job.error_message = "Preview descartado por el usuario"
    job_repo.save_job(job)

    return jsonify({"message": "Preview descartado exitosamente"}), 200
