from .database import DatabaseConnectionManager, DatabaseConnection
from domain.models import User
import logging
import json

class UserRepository:
    def __init__(self):
        self.db = DatabaseConnection()

    # =========================================================================
    # LECTURA DE USUARIOS
    # =========================================================================

    def get_user_by_google_id(self, google_id: str) -> User:
        query = """
            SELECT id_usuario, id_google, correo, nombre_usuario, foto_perfil_url, creado_en
            FROM usuarios WHERE id_google = %s
        """
        results = self.db.execute_query(query, (google_id,))
        if results:
            row = results[0]
            return User(id=row[0], google_id=row[1], email=row[2], name=row[3], picture=row[4], created_at=row[5])
        return None

    def get_user_by_id(self, user_id: int) -> User:
        query = """
            SELECT id_usuario, id_google, correo, nombre_usuario, foto_perfil_url, creado_en
            FROM usuarios WHERE id_usuario = %s
        """
        results = self.db.execute_query(query, (user_id,))
        if results:
            row = results[0]
            return User(id=row[0], google_id=row[1], email=row[2], name=row[3], picture=row[4], created_at=row[5])
        return None

    # =========================================================================
    # CREACIÓN Y ACTUALIZACIÓN
    # =========================================================================

    def create_user(self, google_id: str, email: str, name: str, picture: str) -> User:
        query = """
            INSERT INTO usuarios (id_google, correo, nombre_usuario, nombre_completo, foto_perfil_url)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id_usuario, id_google, correo, nombre_usuario, foto_perfil_url, creado_en
        """
        try:
            with DatabaseConnectionManager() as cursor:
                # Usamos el 'name' tanto para nombre_usuario como para nombre_completo al registrar
                cursor.execute(query, (google_id, email, name, name, picture))
                row = cursor.fetchone()
                return User(id=row[0], google_id=row[1], email=row[2], name=row[3], picture=row[4], created_at=row[5])
        except Exception as e:
            logging.error(f"Error creando usuario: {e}")
            raise

    def update_user_profile(self, user_id: int, data: dict) -> bool:
        """Actualiza campos de perfil permitidos."""
        fields = []
        values = []

        if "username" in data and data["username"]:
            fields.append("nombre_usuario = %s")
            values.append(data["username"])

        if "phone" in data:
            # Nota: Si agregas la columna telefono a la BD después, descomenta
            pass 

        if not fields:
            return False

        values.append(user_id)
        query = f"UPDATE usuarios SET {', '.join(fields)}, actualizado_en = NOW() WHERE id_usuario = %s"

        try:
            self.db.execute_query(query, tuple(values), fetch=False)
            return True
        except Exception as e:
            logging.error(f"Error actualizando perfil del usuario {user_id}: {e}")
            raise

    # =========================================================================
    # DASHBOARD / ESTADÍSTICAS
    # =========================================================================

    def get_user_stats(self, user_id: int) -> dict:
        query = """
            SELECT
                COUNT(*) FILTER (WHERE status = 'completed') AS cursos_completados,
                COUNT(*) FILTER (WHERE status = 'failed')    AS cursos_fallidos,
                COUNT(*) AS total_generaciones
            FROM course_jobs
            WHERE id_usuario = %s
        """
        try:
            results = self.db.execute_query(query, (user_id,))
            if results:
                row = results[0]
                return {
                    "cursos_completados": row[0] or 0,
                    "cursos_fallidos":    row[1] or 0,
                    "total_generaciones": row[2] or 0
                }
        except Exception as e:
            logging.error(f"Error obteniendo stats del usuario {user_id}: {e}")
        return {"cursos_completados": 0, "cursos_fallidos": 0, "total_generaciones": 0}

    def get_recent_courses(self, user_id: int, limit: int = 5) -> list:
        query = """
            SELECT job_id, prompt, status, creado_en, course_outline
            FROM course_jobs
            WHERE id_usuario = %s AND status = 'completed'
            ORDER BY creado_en DESC
            LIMIT %s
        """
        try:
            results = self.db.execute_query(query, (user_id, limit))
            courses = []
            for row in results:
                outline = row[4] or {}
                courses.append({
                    "job_id":     row[0],
                    "prompt":     row[1],
                    "status":     row[2],
                    "created_at": row[3].isoformat() if row[3] else None,
                    "title":      outline.get("title", row[1]),
                    "level":      outline.get("level", "N/A"),
                })
            return courses
        except Exception as e:
            logging.error(f"Error obteniendo cursos recientes del usuario {user_id}: {e}")
            return []

    def get_user_courses(self, user_id: int, page: int = 1, limit: int = 10,
                         status_filter: str = None, sort: str = "recent", q: str = None) -> dict:
        offset = (page - 1) * limit
        conditions = ["id_usuario = %s"]
        params = [user_id]

        if status_filter and status_filter != "all":
            conditions.append("status = %s")
            params.append(status_filter)

        if q:
            conditions.append("prompt ILIKE %s")
            params.append(f"%{q}%")

        order_by = "creado_en DESC"
        where_clause = " AND ".join(conditions)

        count_query = f"SELECT COUNT(*) FROM course_jobs WHERE {where_clause}"
        data_query = f"""
            SELECT job_id, prompt, status, creado_en, course_outline
            FROM course_jobs WHERE {where_clause}
            ORDER BY {order_by} LIMIT %s OFFSET %s
        """

        try:
            total = self.db.execute_query(count_query, tuple(params))[0][0]
            params_data = params + [limit, offset]
            rows = self.db.execute_query(data_query, tuple(params_data))

            courses = []
            for row in rows:
                outline = row[4] or {}
                courses.append({
                    "job_id":     row[0],
                    "prompt":     row[1],
                    "status":     row[2],
                    "created_at": row[3].isoformat() if row[3] else None,
                    "title":      outline.get("title", row[1]),
                    "level":      outline.get("level", "N/A"),
                })

            return {
                "data": courses,
                "meta": {
                    "total_items":  total,
                    "current_page": page,
                    "total_pages":  -(-total // limit)
                }
            }
        except Exception as e:
            logging.error(f"Error listando cursos: {e}")
            return {"data": [], "meta": {"total_items": 0, "current_page": 1, "total_pages": 0}}
