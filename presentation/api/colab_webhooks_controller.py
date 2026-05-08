from flask import Blueprint, request, jsonify
from presentation.middlewares.auth_middleware import require_colab_token
from application.course_pipeline import CoursePipeline

colab_bp = Blueprint('colab_webhooks', __name__, url_prefix='/colab')
pipeline = CoursePipeline()

import json
import os

@colab_bp.route('/entregar_esquema', methods=['POST'])
@require_colab_token
def entregar_esquema():
    """Colab #1 llama a este endpoint cuando termina."""
    data = request.json
    
    # --- DEBUG DUMP ---
    try:
        with open("debug_webhook_1_esquema.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print("Error al guardar debug_webhook_1:", e)
    # ------------------

    job_id = data.get("job_id")
    course_outline = data.get("course_outline")
    
    if not job_id or not course_outline:
        return jsonify({"error": "Faltan datos requeridos"}), 400

    import threading
    t = threading.Thread(target=pipeline.procesar_esquema, args=(job_id, course_outline))
    t.start()

    return jsonify({"message": "Esquema recibido. Procesando..."}), 202

@colab_bp.route('/entregar_ranking', methods=['POST'])
@require_colab_token
def entregar_ranking():
    """Colab #2 llama a este endpoint."""
    data = request.json
    
    # --- DEBUG DUMP ---
    try:
        with open("debug_webhook_2_ranking.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print("Error al guardar debug_webhook_2:", e)
    # ------------------

    job_id = data.get("job_id")
    ranked_sections = data.get("ranked_sections")

    if not job_id or not ranked_sections:
        return jsonify({"error": "Faltan datos requeridos"}), 400

    course = pipeline.procesar_ranking(job_id, ranked_sections)
    
    if course:
        return jsonify({"message": "Curso generado", "course_id": course.id}), 200
    else:
        return jsonify({"error": "No se pudo ensamblar."}), 500
