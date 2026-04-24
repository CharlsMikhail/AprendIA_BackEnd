from flask import Blueprint, request, jsonify

courses_bp = Blueprint('courses', __name__, url_prefix='/courses')
community_bp = Blueprint('community', __name__, url_prefix='/courses') # comparte prefijo para /explore y /rate

# ==== GESTIÓN DE MIS CURSOS ====

@courses_bp.route('/me', methods=['GET'])
def get_my_courses():
    """
    Devuelve la lista resumida de los cursos creados por el usuario.
    """
    # TODO: Obtener user_id de token, query a DB
    return jsonify({
        "data": [
            {
                "id_curso": 42,
                "titulo": "Python Data Science",
                "estado": True,
                "es_publico": False,
                "etiquetas": ["Python", "Datos", "Principiante"],
                "video_introduccion_url": "https://youtube.com/watch?v=ID_VIDEO"
            }
        ],
        "meta": {"total_items": 1, "current_page": 1, "total_pages": 1}
    }), 200

@courses_bp.route('/<int:id_curso>', methods=['GET'])
def get_course_detail(id_curso):
    """
    Recupera la información anidada completa de un curso.
    """
    return jsonify({
        "id_curso": id_curso,
        "titulo": "Python Data Science",
        "descripcion": "Curso estructurado por IA...",
        "estado": True,
        "es_publico": False,
        "duracion_total": "05:30:00",
        "valoracion": 4.5,
        "etiquetas": ["Python", "Datos"],
        "secciones": [
            {
                "id_seccion": 101,
                "titulo": "Módulo 1: Introducción a Datos",
                "descripcion": "Conceptos fundamentales",
                "indice_orden": 1,
                "video": {
                    "id_video_youtube": "dQw4w9WgXcQ",
                    "titulo": "Python Básico en 15 minutos",
                    "nombre_canal": "Programación Ya",
                    "duracion": "00:15:00",
                    "puntuacion_final": 98.5
                }
            }
        ]
    }), 200

@courses_bp.route('/<int:id_curso>', methods=['PATCH'])
def update_course(id_curso):
    """
    Permite modificar información base (título, descripción).
    """
    data = request.json
    return jsonify({"message": "Curso actualizado correctamente"}), 200

@courses_bp.route('/<int:id_curso>/visibility', methods=['PATCH'])
def update_course_visibility(id_curso):
    """
    Alterna el estado de publicación en la comunidad.
    """
    data = request.json
    return jsonify({"message": "Visibilidad del curso actualizada"}), 200

@courses_bp.route('/<int:id_curso>', methods=['DELETE'])
def delete_course(id_curso):
    """
    Borrado lógico del curso.
    """
    return jsonify({"message": "Curso eliminado correctamente (borrado lógico)"}), 200

# ==== COMUNIDAD ====

@community_bp.route('/explore', methods=['GET'])
def explore_community():
    """
    Lista cursos públicos.
    """
    return jsonify({
        "data": [
            {
                "id_curso": 88,
                "titulo": "Master en React Intermedio",
                "autor": {"id_usuario": 5, "nombre_usuario": "GrokMaster99"},
                "valoracion_promedio": 4.8,
                "duracion_total": "12:00:00",
                "etiquetas": ["React", "Frontend"]
            }
        ]
    }), 200

@community_bp.route('/<int:id_curso>/rate', methods=['POST'])
def rate_course(id_curso):
    """
    Valorar un curso público.
    """
    data = request.json
    return jsonify({
        "message": "Valoración registrada con éxito",
        "nueva_valoracion_promedio": 4.9
    }), 200
