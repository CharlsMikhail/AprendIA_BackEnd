from flask import Blueprint, request, jsonify
from presentation.middlewares.auth_middleware import require_colab_token
from application.course_pipeline import CoursePipeline

colab_bp = Blueprint('colab_webhooks', __name__, url_prefix='/colab')
pipeline = CoursePipeline()

@colab_bp.route('/entregar_esquema', methods=['POST'])
@require_colab_token
def entregar_esquema():
    """
    Colab #1 llama a este endpoint cuando termina de generar el esquema (JSON) del curso.
    """
    data = request.json
    job_id = data.get("job_id")
    course_outline = data.get("course_outline")
    
    if not job_id or not course_outline:
        return jsonify({"error": "Faltan datos requeridos (job_id, course_outline)"}), 400

    # Iniciar procesamiento en background (buscar videos) sin bloquear la respuesta a Colab
    # Idealmente, esto se encola en Celery o ThreadPool, pero para simplificar lo disparamos directo
    # ya que buscar en YouTube es relativamente rápido y no bloquea el GPU.
    import threading
    t = threading.Thread(target=pipeline.procesar_esquema, args=(job_id, course_outline))
    t.start()

    return jsonify({"message": "Esquema recibido. Procesando videos candidatos..."}), 202

@colab_bp.route('/entregar_ranking', methods=['POST'])
@require_colab_token
def entregar_ranking():
    """
    Colab #2 llama a este endpoint cuando termina de rankear los videos con IA.
    """
    data = request.json
    job_id = data.get("job_id")
    ranked_sections = data.get("ranked_sections")

    if not job_id or not ranked_sections:
        return jsonify({"error": "Faltan datos requeridos (job_id, ranked_sections)"}), 400

    # Ensamblar curso final
    course = pipeline.procesar_ranking(job_id, ranked_sections)
    
    if course:
        return jsonify({"message": "Curso generado exitosamente", "course_id": course.id}), 200
    else:
        return jsonify({"error": "No se pudo ensamblar el curso."}), 500
