from .database import DatabaseConnectionManager, DatabaseConnection
from domain.models import User
import logging

class UserRepository:
    def __init__(self):
        self.db = DatabaseConnection()
        self._initialize_tables()

    def _initialize_tables(self):
        query = """
        CREATE TABLE IF NOT EXISTS usuarios (
            id SERIAL PRIMARY KEY,
            google_id VARCHAR(255) UNIQUE NOT NULL,
            email VARCHAR(255) UNIQUE NOT NULL,
            name VARCHAR(255) NOT NULL,
            picture TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS cursos_historial (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES usuarios(id),
            course_data JSONB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        try:
            with DatabaseConnectionManager() as cursor:
                cursor.execute(query)
                logging.info("Tablas de usuarios inicializadas correctamente.")
        except Exception as e:
            logging.error(f"Error inicializando tablas: {e}")

    def get_user_by_google_id(self, google_id: str) -> User:
        query = "SELECT id, google_id, email, name, picture, created_at FROM usuarios WHERE google_id = %s"
        results = self.db.execute_query(query, (google_id,))
        if results:
            row = results[0]
            return User(id=row[0], google_id=row[1], email=row[2], name=row[3], picture=row[4], created_at=row[5])
        return None

    def create_user(self, google_id: str, email: str, name: str, picture: str) -> User:
        query = """
        INSERT INTO usuarios (google_id, email, name, picture)
        VALUES (%s, %s, %s, %s)
        RETURNING id, google_id, email, name, picture, created_at
        """
        try:
            with DatabaseConnectionManager() as cursor:
                cursor.execute(query, (google_id, email, name, picture))
                row = cursor.fetchone()
                return User(id=row[0], google_id=row[1], email=row[2], name=row[3], picture=row[4], created_at=row[5])
        except Exception as e:
            logging.error(f"Error creando usuario: {e}")
            raise

    def save_course_history(self, user_id: int, course_data: dict):
        import json
        query = "INSERT INTO cursos_historial (user_id, course_data) VALUES (%s, %s)"
        try:
            with DatabaseConnectionManager() as cursor:
                cursor.execute(query, (user_id, json.dumps(course_data)))
        except Exception as e:
            logging.error(f"Error guardando historial: {e}")
            raise

    def get_user_history(self, user_id: int):
        query = "SELECT course_data, created_at FROM cursos_historial WHERE user_id = %s ORDER BY created_at DESC"
        results = self.db.execute_query(query, (user_id,))
        return [{"course": r[0], "created_at": r[1]} for r in results]
