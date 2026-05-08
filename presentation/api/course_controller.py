from flask import Blueprint, request, jsonify, g
from infrastructure.db.course_job_repository import CourseJobRepository
from presentation.middlewares.auth_middleware import require_jwt
import logging

courses_bp = Blueprint('courses', __name__, url_prefix='/courses')
explore_bp = Blueprint('explore', __name__, url_prefix='/explore')

job_repo = CourseJobRepository()


# =============================================================================
# DETALLE DE CURSO
# =============================================================================

@courses_bp.route('/<string:course_id>', methods=['GET'])
@require_jwt
def get_course_detail(course_id: str):
    """Obtiene el detalle completo de un curso generado."""
    job = job_repo.get_job(course_id)

    if not job:
        return jsonify({"error": "Curso no encontrado"}), 404

    if job.user_id != g.current_user_id:
        return jsonify({"error": "No autorizado"}), 403

    outline = job.course_outline or {}
    sections = []
    for i, section in enumerate(job.sections_with_candidates or []):
        candidates = section.get("candidates", [])
        candidate = candidates[0] if candidates else {}
        sections.append({
            "indice": i + 1,
            "titulo": section.get("title", ""),
            "descripcion": section.get("description", ""),
            "video": {
                "url": candidate.get("url"),
                "titulo": candidate.get("title"),
                "score": candidate.get("score"),
                "views": candidate.get("views"),
                "has_transcript": bool(candidate.get("transcript"))
            } if candidate else None
        })

    return jsonify({
        "course_id": job.job_id,
        "titulo": outline.get("title", job.prompt),
        "nivel": outline.get("level", "N/A"),
        "descripcion": outline.get("level_description", ""),
        "objetivos": outline.get("learningOutcomes", []),
        "requisitos": outline.get("requirements", []),
        "secciones": sections,
        "status": job.status,
        "created_at": job.created_at
    }), 200


# =============================================================================
# EDICIÓN, VISIBILIDAD Y ELIMINACIÓN
# =============================================================================

@courses_bp.route('/<string:course_id>', methods=['PATCH'])
@require_jwt
def update_course(course_id: str):
    """
    Edita información del curso.
    Body: { "title": "...", "description": "..." }
    """
    job = job_repo.get_job(course_id)
    if not job:
        return jsonify({"error": "Curso no encontrado"}), 404
    if job.user_id != g.current_user_id:
        return jsonify({"error": "No autorizado"}), 403

    data = request.json or {}
    outline = job.course_outline or {}

    if "title" in data:
        outline["title"] = data["title"]
    if "description" in data:
        outline["level_description"] = data["description"]

    job.course_outline = outline
    job_repo.save_job(job)
    return jsonify({"message": "Curso actualizado correctamente"}), 200


@courses_bp.route('/<string:course_id>/visibility', methods=['PATCH'])
@require_jwt
def update_visibility(course_id: str):
    """
    Cambia la visibilidad del curso.
    Body: { "es_publico": true|false }
    """
    job = job_repo.get_job(course_id)
    if not job:
        return jsonify({"error": "Curso no encontrado"}), 404
    if job.user_id != g.current_user_id:
        return jsonify({"error": "No autorizado"}), 403

    data = request.json or {}
    es_publico = data.get("es_publico", False)

    outline = job.course_outline or {}
    outline["_es_publico"] = es_publico
    job.course_outline = outline
    job_repo.save_job(job)

    return jsonify({"message": "Visibilidad actualizada", "es_publico": es_publico}), 200


@courses_bp.route('/<string:course_id>', methods=['DELETE'])
@require_jwt
def delete_course(course_id: str):
    """Borrado lógico del curso."""
    job = job_repo.get_job(course_id)
    if not job:
        return jsonify({"error": "Curso no encontrado"}), 404
    if job.user_id != g.current_user_id:
        return jsonify({"error": "No autorizado"}), 403

    job.status = "failed"
    job.error_message = "Eliminado por el usuario"
    job_repo.save_job(job)
    return jsonify({"message": "Curso eliminado correctamente"}), 200


# =============================================================================
# PROGRESO DEL CURSO
# =============================================================================

@courses_bp.route('/<string:course_id>/progress', methods=['PATCH'])
@require_jwt
def update_progress(course_id: str):
    """
    Actualiza el progreso del usuario en un curso.
    Body: { "part_index": 1, "watched_seconds": 120 }
    """
    job = job_repo.get_job(course_id)
    if not job:
        return jsonify({"error": "Curso no encontrado"}), 404

    data = request.json or {}
    part_index = data.get("part_index")
    watched_seconds = data.get("watched_seconds", 0)

    if part_index is None:
        return jsonify({"error": "part_index es requerido"}), 400

    # TODO: persistir en tabla progreso_usuario cuando exista en Neon
    logging.info(f"Progreso: user={g.current_user_id}, curso={course_id}, parte={part_index}, seg={watched_seconds}")
    return jsonify({
        "message": "Progreso actualizado",
        "part_index": part_index,
        "watched_seconds": watched_seconds
    }), 200


# =============================================================================
# EXPLORAR CURSOS (CATÁLOGO PÚBLICO)
# =============================================================================

@explore_bp.route('/courses', methods=['GET'])
def explore_courses():
    """
    Explora el catálogo de cursos públicos.
    Query params: ?sort=recent|top_rated&q=...&page=1
    """
    from infrastructure.db.database import DatabaseConnection
    db = DatabaseConnection()

    page = request.args.get('page', 1, type=int)
    limit = 10
    offset = (page - 1) * limit
    q = request.args.get('q', None)

    conditions = ["status = 'completed'"]
    params = []

    if q:
        conditions.append("prompt ILIKE %s")
        params.append(f"%{q}%")

    where = " AND ".join(conditions)
    query = f"""
        SELECT job_id, prompt, created_at, course_outline
        FROM course_jobs WHERE {where}
        ORDER BY created_at DESC LIMIT %s OFFSET %s
    """
    params += [limit, offset]

    try:
        rows = db.execute_query(query, tuple(params))
        courses = []
        for row in rows:
            outline = row[3] or {}
            if outline.get("_es_publico", False):
                courses.append({
                    "job_id":     row[0],
                    "titulo":     outline.get("title", row[1]),
                    "nivel":      outline.get("level", "N/A"),
                    "created_at": row[2].isoformat() if row[2] else None
                })
        return jsonify({"data": courses, "page": page}), 200
    except Exception as e:
        logging.error(f"Error explorando cursos: {e}")
        return jsonify({"data": [], "page": page}), 200


@explore_bp.route('/courses/top-rated', methods=['GET'])
def top_rated_courses():
    """Cursos mejor puntuados. (Requiere tabla de valoraciones)"""
    return jsonify({"message": "Próximamente. Requiere tabla de valoraciones.", "data": []}), 501


@explore_bp.route('/tags/popular', methods=['GET'])
def popular_tags():
    """Tags populares. (Requiere columna de tags en DB)"""
    return jsonify({"message": "Próximamente.", "data": []}), 501
