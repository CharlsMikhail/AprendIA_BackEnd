# AprendIA - Backend: Generador de Cursos con IA

Este es el backend de **AprendIA**, una plataforma basada en Inteligencia Artificial y Clean Architecture diseñada para generar cursos educativos personalizados a partir de un tema (prompt) dado por el usuario. La aplicación integra APIs de YouTube, LLMs (Gemini/OpenAI) y está preparada para ejecutar análisis de sentimiento con Transformers y transcripciones con Fast-Whisper acelerados por GPU.

---

## 📂 Estructura de Carpetas y Funcionalidad

El proyecto ha sido refactorizado siguiendo los principios de **Clean Architecture** para garantizar modularidad y escalabilidad.

### `domain/` (Capa de Dominio)
Contiene las entidades y reglas de negocio puras independientes de cualquier framework.
- **`models.py`**: Define las clases de datos (`User`, `Course`, `Section`, `VideoCandidate`) utilizando `dataclasses` para representar la información de manera estructurada en toda la app.

### `application/` (Capa de Casos de Uso)
Orquesta el flujo de los datos desde y hacia las entidades.
- **`auth_service.py`**: Maneja la lógica de autenticación mediante Google OAuth2 y la creación de sesiones de usuario.
- **`ai_services.py`**: Encargado de las interacciones con LLMs (actualmente Google Gemini) para estructurar el contenido de los cursos.
- **`course_pipeline.py`**: Es el orquestador principal del proyecto. Ensambla el curso conectando la IA, la API de YouTube, y la evaluación de transcripciones y sentimientos.

### `infrastructure/` (Capa de Infraestructura)
Implementa los detalles técnicos (bases de datos, APIs externas, modelos de inferencia).
- **`db/database.py`**: Patrón Singleton que maneja un Pool de conexiones a PostgreSQL mediante `psycopg2`.
- **`db/user_repository.py`**: Lógica de acceso a datos (queries) para crear usuarios, buscar historial y guardar cursos.
- **`api/youtube_api.py`**: Cliente de la API oficial de YouTube (v3) encargado de realizar búsquedas y calcular los puntajes base (métricas) de los videos.
- **`evaluators/transcription_evaluator.py`**: Procesa, limpia y resume las transcripciones de los videos utilizando NLP básico (TF-IDF, spaCy, nltk).
- **`inference/sentiment_analyzer.py`**: Módulo dedicado a ejecutar Transformers (ej. RoBERTa) en GPU para análisis de comentarios.
- **`inference/transcriber.py`**: Transcriptor local utilizando Fast-Whisper y extracción de audio.

### `presentation/` (Capa de Presentación)
- **`routes.py`**: Exposición de los endpoints HTTP mediante Blueprints de Flask. Separa el ruteo de la lógica de negocio.

### Raíz del Proyecto
- **`app.py`**: Punto de entrada de la aplicación. Configura Flask, CORS, carga las variables de entorno y registra las rutas.
- **`requirements.txt`**: Dependencias de Python.
- **`.env`**: Archivo de variables de entorno con credenciales secretas.
- **`docs/`**: Directorio para documentos estáticos (ej. `Criterios para el filtrado de videos - AprendIA.csv`).

---

## 📝 Actividades Pendientes (`TODO`) por Archivo

Varios módulos han sido preparados como esqueletos para recibir la nueva lógica avanzada. Estas son las tareas pendientes que el equipo debe completar:

*   **`application/ai_services.py`**:
    *   `TODO`: Implementar la lógica real para validar Términos y Condiciones (Guardrails) y refinar los prompts con un LLM. Actualmente retorna el prompt sin procesar.
    *   `TODO`: Mejorar el método de extracción de la palabra clave (topic) desde el prompt en la estructuración.
*   **`application/course_pipeline.py`**:
    *   `TODO`: Integrar el bucle de Análisis de Sentimiento (Descargar comentarios -> evaluar con `sentiment_analyzer`).
    *   `TODO`: Ejecutar la evaluación real de las transcripciones y compararlas semánticamente (RAG).
    *   `TODO`: Armar el algoritmo de ranking final para escoger el mejor video mezclando métricas + sentimiento + RAG.
*   **`infrastructure/api/youtube_api.py`**:
    *   `TODO`: Modificar el filtro de búsqueda para obtener la base de 50 videos (actualmente en 10 para evitar cuotas rápidas).
    *   `TODO`: Incorporar el umbral estricto para descartar videos con puntaje < 0.3.
    *   `TODO`: Refactorizar el algoritmo de balance para incluir la métrica específica del creador de contenido.
*   **`infrastructure/evaluators/transcription_evaluator.py`**:
    *   `TODO`: Integrar la solución fallback a Fast-Whisper cuando la API de YouTube devuelva un error (código comentado provisto).
    *   `TODO`: Reemplazar el TF-IDF por una arquitectura real de RAG (Vector Database) para validación semántica avanzada.
*   **`infrastructure/inference/sentiment_analyzer.py`**:
    *   `TODO`: Inicializar el modelo HuggingFace en la memoria de la GPU (CUDA).
    *   `TODO`: Implementar la tokenización y predicción para generar el índice de confianza (0 a 1).
*   **`infrastructure/inference/transcriber.py`**:
    *   `TODO`: Implementar `yt-dlp` para descarga local de audio.
    *   `TODO`: Inicializar y configurar el motor de `faster-whisper`.

---

## 🚀 Comandos para Correr el Programa

**1. Requisitos previos:**
*   Python 3.10+
*   PostgreSQL instalado y corriendo.
*   (Opcional) Tarjeta gráfica NVIDIA (CUDA) para los modelos de la capa de inferencia.

**2. Clonar y preparar el entorno virtual:**
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

**3. Instalar dependencias:**
```bash
pip install -r requirements.txt
python -m spacy download es_core_news_sm
```

**4. Configurar Variables de Entorno (`.env`):**
Asegúrate de crear un archivo `.env` en la raíz (basado en tus credenciales originales) con las siguientes llaves:
```env
YOUTUBE_API_KEY=tu_clave_youtube
GOOGLE_API_KEY=tu_clave_gemini
GOOGLE_CLIENT_ID=tu_client_id_oauth
GOOGLE_CLIENT_SECRET=tu_client_secret_oauth
FLASK_SECRET_KEY=clave_segura_para_sesiones
OAUTHLIB_INSECURE_TRANSPORT=1 # (1 solo en desarrollo local)

# Base de datos
DB_NAME=nombre_db
DB_USER=usuario_db
DB_PASSWORD=password_db
DB_HOST=localhost
DB_PORT=5432
```

**5. Ejecutar la aplicación:**
```bash
python app.py
```
El servidor levantará en `http://localhost:5000`.

---

## 💡 Posibles Mejoras (Arquitectura Futura)

1.  **Microservicio de Inferencia (GPU):**
    Una vez que la lógica de Fast-Whisper y los Transformers de análisis de sentimiento (`infrastructure/inference/`) esté finalizada, se recomienda extraer estos dos módulos hacia un contenedor (Docker) independiente o un microservicio en **FastAPI**. Esto evitará que la ejecución pesada en GPU bloquee los *workers* ligeros del servidor web Flask de AprendIA.
2.  **Uso de Docker y Docker Compose:**
    Crear un `Dockerfile` para la aplicación y un `docker-compose.yml` que levante de manera simultánea el backend y la base de datos PostgreSQL. Esto facilitará enormemente el despliegue del proyecto en entornos de producción sin tener que configurar las bases de datos manualmente.
3.  **Migrar a un ORM Completo (SQLAlchemy):**
    Si la base de datos de usuarios y cursos crece en complejidad (relaciones, categorías, favoritos), sería ideal reemplazar las queries crudas de `psycopg2` en el `user_repository.py` por **SQLAlchemy** (que ya se encuentra en el `requirements.txt`) para aprovechar migraciones automáticas mediante Alembic.
4.  **Integración de Base de Datos Vectorial (RAG):**
    Para la capa de validación de transcripciones, considerar desplegar ChromaDB, Pinecone o FAISS para comparar dinámicamente el contenido de los videos extraídos contra corpus educativos certificados.