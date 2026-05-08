import os
import psycopg2
from psycopg2 import pool
from dotenv import load_dotenv
import logging

# Cargar variables de entorno
load_dotenv()


class DatabaseConnection:
    """
    Clase Singleton para manejar la conexión a PostgreSQL
    Implementa un pool de conexiones para mejor rendimiento
    """
    _instance = None
    _connection_pool = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseConnection, cls).__new__(cls)
            cls._instance._initialize_pool()
        return cls._instance

    def _initialize_pool(self):
        """Inicializa el pool de conexiones a la base de datos"""
        if self._connection_pool is None:
            try:
                # Obtener credenciales del archivo .env
                db_config = {
                    'database': os.getenv('DB_NAME'),
                    'user': os.getenv('DB_USER'),
                    'password': os.getenv('DB_PASSWORD'),
                    'host': os.getenv('DB_HOST'),
                    'port': os.getenv('DB_PORT')
                }

                # Validar que todas las credenciales estén presentes
                if not all(db_config.values()):
                    raise ValueError("Faltan credenciales de base de datos en el archivo .env")

                # Crear pool de conexiones (sslmode=require para Neon)
                self._connection_pool = psycopg2.pool.SimpleConnectionPool(
                    1,  # Mínimo de conexiones
                    20,  # Máximo de conexiones
                    sslmode='require',
                    **db_config
                )

                logging.info("Pool de conexiones a PostgreSQL creado exitosamente")

            except Exception as e:
                logging.error(f"Error al crear el pool de conexiones: {e}")
                raise

    def get_connection(self):
        """Obtiene una conexión viva del pool."""
        try:
            if self._connection_pool:
                for _ in range(3):  # Reintentos si la conexión está muerta
                    connection = self._connection_pool.getconn()
                    if connection:
                        if connection.closed != 0:
                            # Si la conexión está cerrada, la descartamos
                            self._connection_pool.putconn(connection, close=True)
                            continue
                        return connection
                raise Exception("No se pudo obtener una conexión válida del pool")
            else:
                raise Exception("Pool de conexiones no inicializado")
        except Exception as e:
            logging.error(f"Error al obtener conexión: {e}")
            raise

    def return_connection(self, connection):
        """Devuelve una conexión al pool."""
        try:
            if self._connection_pool and connection:
                close_conn = (connection.closed != 0)
                self._connection_pool.putconn(connection, close=close_conn)
        except Exception as e:
            logging.error(f"Error al devolver conexión al pool: {e}")

    def close_all_connections(self):
        """Cierra todas las conexiones del pool"""
        try:
            if self._connection_pool:
                self._connection_pool.closeall()
                logging.info("Todas las conexiones cerradas")
        except Exception as e:
            logging.error(f"Error al cerrar conexiones: {e}")

    def execute_query(self, query, params=None, fetch=True):
        """Ejecuta una consulta SQL"""
        connection = None
        cursor = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor()

            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            if fetch:
                results = cursor.fetchall()
                if connection.closed == 0:
                    connection.commit()
                return results
            else:
                if connection.closed == 0:
                    connection.commit()
                return cursor.rowcount

        except Exception as e:
            if connection and connection.closed == 0:
                try:
                    connection.rollback()
                except Exception:
                    pass
            logging.error(f"Error ejecutando consulta: {e}")
            raise
        finally:
            if cursor and not cursor.closed:
                try:
                    cursor.close()
                except Exception:
                    pass
            if connection:
                self.return_connection(connection)

    def execute_many(self, query, params_list):
        """Ejecuta múltiples consultas con diferentes parámetros"""
        connection = None
        cursor = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor()

            cursor.executemany(query, params_list)
            if connection.closed == 0:
                connection.commit()
            return cursor.rowcount

        except Exception as e:
            if connection and connection.closed == 0:
                try:
                    connection.rollback()
                except Exception:
                    pass
            logging.error(f"Error ejecutando consultas múltiples: {e}")
            raise
        finally:
            if cursor and not cursor.closed:
                try:
                    cursor.close()
                except Exception:
                    pass
            if connection:
                self.return_connection(connection)


# Context manager para usar la conexión de forma segura
class DatabaseConnectionManager:
    """Context manager para manejar conexiones de forma segura"""

    def __init__(self):
        self.db = DatabaseConnection()
        self.connection = None
        self.cursor = None

    def __enter__(self):
        self.connection = self.db.get_connection()
        self.cursor = self.connection.cursor()
        return self.cursor

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.connection and self.connection.closed == 0:
            try:
                if exc_type is not None:
                    self.connection.rollback()
                else:
                    self.connection.commit()
            except psycopg2.InterfaceError:
                pass

        if self.cursor and not self.cursor.closed:
            try:
                self.cursor.close()
            except Exception:
                pass
                
        if self.connection:
            self.db.return_connection(self.connection)


# Ejemplo de uso
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(level=logging.INFO)

    try:
        # Crear instancia del singleton
        db = DatabaseConnection()

        # Ejemplo 1: Usando el método execute_query
        results = db.execute_query("SELECT version();")
        print("Versión de PostgreSQL:", results[0][0])

        # Ejemplo 2: Usando el context manager
        with DatabaseConnectionManager() as cursor:
            cursor.execute("SELECT current_database();")
            db_name = cursor.fetchone()
            print("Base de datos actual:", db_name[0])

        # Ejemplo 3: Insertar datos
        # db.execute_query(
        #     "INSERT INTO usuarios (nombre, email) VALUES (%s, %s)",
        #     ("Juan Pérez", "juan@email.com"),
        #     fetch=False
        # )

    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Cerrar todas las conexiones al finalizar
        db.close_all_connections()