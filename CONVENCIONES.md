# Convenciones de Desarrollo - AprendIA Backend

Este documento establece los estándares y normativas que todo el equipo de desarrollo debe seguir al contribuir al backend de AprendIA.

---

## 1. Nomenclatura de Commits (Conventional Commits)

Para mantener un historial de Git limpio y fácil de leer, utilizaremos la convención de **Conventional Commits**. Cada mensaje de commit debe tener la siguiente estructura:

`tipo(alcance): descripción breve`

### Tipos Permitidos:
*   `feat`: Añade una nueva funcionalidad (ej. *feat(auth): integración con Google OAuth2*).
*   `fix`: Soluciona un error o bug (ej. *fix(youtube): corrección del cálculo de score nulo*).
*   `docs`: Cambios exclusivos en la documentación (ej. *docs(readme): actualización de comandos de instalación*).
*   `refactor`: Cambios en el código que no añaden funcionalidades ni corrigen errores, pero mejoran la estructura (ej. *refactor(api): extracción del cliente de youtube*).
*   `test`: Añade o corrige pruebas unitarias/integración.
*   `chore`: Tareas de mantenimiento, actualización de dependencias o configuraciones (ej. *chore(deps): actualización de requirements.txt*).

---

## 2. Arquitectura Teórica (Clean Architecture)

El backend está construido bajo los principios de **Clean Architecture**. El objetivo principal de esta arquitectura es **aislar las reglas de negocio** de los detalles técnicos (como la base de datos, las APIs externas o el framework web). 

### La Regla de Dependencia
Las dependencias en el código siempre deben apuntar **hacia adentro**, hacia las políticas de alto nivel (Dominio). La capa de Infraestructura conoce a la de Aplicación y Dominio, pero la capa de Dominio no conoce absolutamente nada de Infraestructura (no sabe qué es Flask ni qué es PostgreSQL).

### Estructura de Capas:
1.  **Capa de Dominio (`domain/`)**: El corazón del software. Contiene las Entidades (`models.py`). Aquí se definen los datos puros (ej. qué es un Curso, qué es un Usuario). No contiene librerías externas.
2.  **Capa de Aplicación (`application/`)**: Contiene los "Casos de Uso". Es la orquestadora. Toma los datos del mundo exterior, los procesa aplicando las reglas de negocio (ej. validación con LLMs, ensamblaje de cursos) y los guarda.
3.  **Capa de Infraestructura (`infrastructure/`)**: Contiene los detalles técnicos y conexiones con el mundo exterior. Aquí viven las consultas SQL a PostgreSQL (`db/`), las llamadas a la API de YouTube (`api/`) y los modelos pesados de HuggingFace/Fast-Whisper (`inference/`).
4.  **Capa de Presentación (`presentation/`)**: La puerta de entrada a la aplicación. En nuestro caso, está manejada por **Flask**. Recibe peticiones HTTP, extrae el JSON y se lo pasa a la Capa de Aplicación. **No debe contener lógica de negocio**.

---

## 3. Convenciones de Tecnologías

### A. Python
*   **Estilo:** Seguir el estándar **PEP 8**. (Se recomienda usar formateadores como `black` o `autopep8`).
*   **Nomenclatura:**
    *   Variables, funciones y métodos: `snake_case` (ej. `calcular_score_video()`).
    *   Clases: `PascalCase` (ej. `YouTubeAPIClient`).
    *   Constantes: `UPPER_SNAKE_CASE` (ej. `YOUTUBE_API_KEY`).
*   **Tipado (Type Hinting):** Siempre que sea posible, especificar los tipos de entrada y salida de las funciones para mejorar la legibilidad y el autocompletado.
    ```python
    # Correcto:
    def get_user_by_id(user_id: int) -> User:
    ```
*   **Comentarios:** Usar docstrings (`""" """`) para describir clases y funciones complejas.

### B. Flask
*   **Blueprints:** Todas las rutas deben ser registradas utilizando Blueprints (ver `presentation/routes.py`) para mantener `app.py` limpio.
*   **Controladores "Flacos":** Las funciones de ruta en Flask **nunca** deben ejecutar lógica compleja, cálculos o llamadas a bases de datos directamente. Su única responsabilidad es recibir el `request`, pasarlo a un servicio de la carpeta `application/` y devolver un `jsonify()`.

### C. Base de Datos (PostgreSQL)
*   **Nomenclatura de Tablas:** Siempre en `snake_case` y en **plural** (ej. `usuarios`, `cursos_historial`).
*   **Nomenclatura de Columnas:** Siempre en `snake_case` (ej. `google_id`, `created_at`).
*   **Llaves Primarias:** Se utilizará `id` (entero autoincremental `SERIAL`) como estándar para la llave primaria principal de cualquier tabla.
*   **Fechas:** Todas las tablas deben incluir una columna `created_at` tipo `TIMESTAMP DEFAULT CURRENT_TIMESTAMP` para auditoría básica.
*   **Conexiones:** Siempre utilizar el manejador de contexto (`with DatabaseConnectionManager() as cursor:`) para asegurar que las conexiones al pool se abran y cierren de manera segura, previniendo fugas de memoria.
