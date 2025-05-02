# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import re
import random
import time
from datetime import datetime
from googleapiclient.discovery import build
# from openai import AzureOpenAI # Eliminado
import requests # Añadido para llamadas a Google API
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
# from youtube_transcript_api.formatters import TextFormatter # No se usaba
from googleapiclient.errors import HttpError

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# --- Environment variables ---
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
# AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT") # Eliminado
# AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY") # Eliminado
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # Añadido

# --- Google Generative AI API Configuration ---
# Puedes cambiar 'gemini-1.5-flash-latest' si necesitas otro modelo
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash-latest")
GOOGLE_API_URL_TEMPLATE = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent"


# Configure Azure OpenAI client # Eliminado
# azure_client = AzureOpenAI(
#     azure_endpoint=AZURE_OPENAI_ENDPOINT,
#     api_key=AZURE_OPENAI_API_KEY,
#     api_version="2024-12-01-preview" # Ajusta la versión si es necesario
# )

# --- Helper Function for Google Generative AI API Call (Minimal version) ---
def call_google_generative_api_for_text(prompt_text):
    """
    Calls the Google Generative AI API and returns the generated text.
    Minimal error handling for direct replacement.
    """
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")

    url = f"{GOOGLE_API_URL_TEMPLATE}?key={GOOGLE_API_KEY}"
    headers = {'Content-Type': 'application/json'}
    # Estructura simple de la solicitud para Gemini
    body = json.dumps({
        "contents": [{
            "parts": [{"text": prompt_text}]
        }],
         # Mantener configuraciones de generación simples o excluirlas
         # "generationConfig": {
         #    "temperature": 0.7, # Coincide con el original
         # }
    })

    try:
        response = requests.post(url, headers=headers, data=body, timeout=120)
        response.raise_for_status()
        response_data = response.json()

        # Extracción estándar del texto generado por Gemini
        generated_text = response_data['candidates'][0]['content']['parts'][0]['text']
        return generated_text

    except requests.exceptions.RequestException as req_err:
        print(f"Error en la solicitud a Google API: {req_err}")
        # Intenta mostrar la respuesta si existe
        try:
            print("Error Body:", response.text)
        except:
            pass
        raise # Relanzar el error para que sea capturado por el llamador original
    except (KeyError, IndexError) as parse_err:
        print(f"Error parseando respuesta de Google API: {parse_err}")
        print("Raw Response Data:", response_data)
        raise ValueError(f"Formato de respuesta inesperado de Google API: {parse_err}")
    except Exception as e:
        print(f"Error inesperado en llamada a Google API: {e}")
        raise


def get_course_outline(prompt):
    """Genera el esquema del curso basado en el prompt y el nivel (Usando Google API)"""
    # Definir los niveles disponibles y sus características (sin cambios)
    levels = {
        "principiante": {
            "keywords": ["principiante", "inicial", "introductorio", "desde cero", "básico"],
            "num_sections": 4,
            "description": "Enfocado en conceptos fundamentales y primeros pasos",
            "depth": "superficial",
            "focus": "comprensión de conceptos básicos"
        },
        "intermedio": {
            "keywords": ["intermedio", "medio"],
            "num_sections": 6,
            "description": "Profundización en conceptos y aplicaciones prácticas",
            "depth": "moderada",
            "focus": "aplicación de conceptos y desarrollo de habilidades"
        },
        "avanzado": {
            "keywords": ["avanzado", "experto", "profesional"],
            "num_sections": 8,
            "description": "Temas especializados y técnicas avanzadas",
            "depth": "profunda",
            "focus": "optimización y casos de uso complejos"
        },
        "maestro": {
            "keywords": ["maestro", "master", "completo", "exhaustivo"],
            "num_sections": 10,
            "description": "Cobertura exhaustiva y especialización avanzada",
            "depth": "muy profunda",
            "focus": "dominio completo y técnicas de vanguardia"
        }
    }

    # Nivel por defecto (sin cambios)
    level = "principiante"

    # Convertir el prompt a minúsculas para la comparación (sin cambios)
    prompt_lower = prompt.lower()

    # Detectar el nivel en el prompt (sin cambios)
    for level_name, level_info in levels.items():
        if any(keyword in prompt_lower for keyword in level_info["keywords"]):
            level = level_name
            break

    # Eliminar palabras clave de nivel del prompt para obtener el tema principal (sin cambios)
    topic = prompt
    for level_info in levels.values():
        for keyword in level_info["keywords"]:
            topic = topic.replace(keyword, "").strip()

    # Obtener la configuración del nivel seleccionado (sin cambios)
    level_config = levels[level]

    try:
        # Primero, generar la introducción general (Usando Google API)
        # Mantener los prompts originales, combinándolos para Google API
        intro_system_message = f"""
        Eres un experto en educación. Tu tarea es crear una introducción general y motivadora para un curso de nivel {level} sobre {topic}.

        La introducción debe:
        1. Ser breve y atractiva
        2. Explicar por qué es importante aprender {topic}
        3. Describir el enfoque del curso
        4. Motivar a los estudiantes
        5. No entrar en detalles técnicos (esos irán en las secciones)

        IMPORTANTE: No incluyas ninguna referencia a videos, URLs o contenido multimedia en la respuesta.

        Proporciona la respuesta en formato JSON:
        {{
            "introduction": "Texto de la introducción"
        }}
        """
        intro_user_message = f"Crea una introducción para un curso sobre: {topic}"
        intro_full_prompt = f"{intro_system_message}\n\nUSER QUESTION:\n{intro_user_message}" # Combinar

        # Reemplazo de llamada Azure -> Google
        intro_response_text = call_google_generative_api_for_text(intro_full_prompt)
        # Limpiar posible markdown añadido por la API (puede ser necesario con Gemini)
        intro_response_text = re.sub(r'^```json\s*', '', intro_response_text).strip()
        intro_response_text = re.sub(r'\s*```$', '', intro_response_text).strip()
        intro_data = json.loads(intro_response_text) # Parsear el JSON devuelto por la IA

        # Luego, generar el contenido detallado del curso (Usando Google API)
        # Mantener los prompts originales, combinándolos para Google API
        content_system_message = f"""
        Eres un experto en diseño de cursos educativos. Tu tarea es crear un esquema detallado para un curso de nivel {level} sobre {topic}.

        El curso debe incluir:
        1. Un título atractivo y descriptivo que refleje el nivel {level}
        2. {level_config['num_sections']} secciones principales, cada una con:
           - Título descriptivo
           - Descripción detallada del contenido
        3. Objetivos de aprendizaje específicos al nivel {level}
        4. Requisitos previos apropiados para el nivel

        Para un curso de nivel {level}, asegúrate de:
        - {level_config['description']}
        - Mantener una profundidad {level_config['depth']} en los temas
        - Enfocarse en {level_config['focus']}
        - {f"Incluir ejercicios prácticos y proyectos" if level != "principiante" else "Incluir ejemplos simples y ejercicios guiados"}
        - {f"Cubrir temas especializados y técnicas avanzadas" if level in ["avanzado", "maestro"] else "Mantener un enfoque en conceptos fundamentales"}

        IMPORTANTE: No incluyas ninguna referencia a videos, URLs o contenido multimedia en la respuesta.

        Proporciona la respuesta en formato JSON con la siguiente estructura:
        {{
            "title": "Título del curso",
            "sections": [
                {{
                    "title": "Título de la sección",
                    "description": "Descripción detallada"
                }}
                // ... más secciones
            ],
            "learningOutcomes": ["Objetivo 1", "Objetivo 2", ...],
            "requirements": ["Requisito 1", "Requisito 2", ...],
            "level": "{level}",
            "level_description": "{level_config['description']}",
            "total_sections": {level_config['num_sections']}
        }}
        """
        content_user_message = f"Crea un curso sobre: {topic}"
        content_full_prompt = f"{content_system_message}\n\nUSER QUESTION:\n{content_user_message}" # Combinar

        # Reemplazo de llamada Azure -> Google
        content_response_text = call_google_generative_api_for_text(content_full_prompt)
        # Limpiar posible markdown añadido por la API
        content_response_text = re.sub(r'^```json\s*', '', content_response_text).strip()
        content_response_text = re.sub(r'\s*```$', '', content_response_text).strip()
        content_data = json.loads(content_response_text) # Parsear el JSON devuelto por la IA

        # Combinar la introducción con el contenido del curso (sin cambios)
        course_outline = {
            **content_data,
            "introduction": intro_data["introduction"]
        }

        # Añadir queries de búsqueda por defecto (esto estaba en tu código original implícitamente o añadido después, lo mantenemos si estaba antes)
        # Si no estaba en el original que me diste, puedes quitar este bloque.
        # Asumiendo que necesitas estas queries como en el ejemplo anterior:
        course_outline["searchQueries"] = {
             "introductory": f"introducción a {topic} para {level}s",
             **{f"section{i}": f"{section['title']} tutorial {level}" for i, section in enumerate(course_outline["sections"])}
        }

        return course_outline

    # Mantener el manejo de errores original
    except Exception as e:
        print(f"Error generando esquema del curso: {e}")
        # Considera loguear el texto recibido si falla el json.loads
        if 'intro_response_text' in locals() and isinstance(e, json.JSONDecodeError):
            print("Texto recibido (intro):", intro_response_text)
        if 'content_response_text' in locals() and isinstance(e, json.JSONDecodeError):
            print("Texto recibido (content):", content_response_text)
        raise


# --- Resto de las funciones SIN CAMBIOS ---

def calculate_video_score(video_details, snippet, statistics, days_since_published, total_minutes):
    """Calcula la puntuación de un video basada en múltiples criterios (SIN CAMBIOS)"""
    # 1. Relevancia (30%)
    relevance_score = 1.0

    # 2. Calidad del video (25%)
    quality_score = 0
    # Simplificado, asume que tags pueden no estar presentes o ser fiables
    definition = video_details.get('contentDetails', {}).get('definition')
    if definition == 'hd':
        quality_score = 0.8
    elif definition == 'sd':
        quality_score = 0.4
    else: # default a bit lower if unknown
        quality_score = 0.3

    # Intento original con tags (menos fiable)
    # if "HD" in snippet.get("tags", []):
    #     quality_score += 0.3
    # if "4K" in snippet.get("tags", []):
    #     quality_score += 0.4
    # if "1080p" in snippet.get("tags", []):
    #     quality_score += 0.3


    # 3. Engagement (20%)
    views = int(statistics.get("viewCount", 0))
    likes = int(statistics.get("likeCount", 0))
    comments = int(statistics.get("commentCount", 0))

    engagement_score = 0
    if views > 100: # Evitar división por cero y ruido inicial
        # Normalizar ratios para que no se disparen con pocas views
        like_ratio = (likes / views)
        comment_ratio = (comments / views)
        # Ponderar y limitar a 1.0
        engagement_score = min(1.0, (like_ratio * 5) * 0.6 + (comment_ratio * 20) * 0.4) # Ajustar multiplicadores según necesidad

    # 4. Actualidad (15%)
    recency_score = 1.0 if days_since_published <= 365 else (0.5 if days_since_published <= 730 else 0.2) # Más granular

    # 5. Duración (10%) - Ideal entre 5 y 25 mins
    duration_score = 1.0 if 5 <= total_minutes <= 25 else (0.5 if total_minutes < 5 else 0.7) # Penalizar menos los largos > 25

    # Cálculo de la puntuación final con pesos (SIN CAMBIOS)
    final_score = (
            relevance_score * 0.30 +
            quality_score * 0.25 +
            engagement_score * 0.20 +
            recency_score * 0.15 +
            duration_score * 0.10
    )

    return final_score # Ya no redondea aquí para más precisión en el ordenamiento

def get_video_transcript(video_id, max_minutes=None):
    """Obtiene la transcripción de un video de YouTube en español (SIN CAMBIOS)"""
    try:
        # Intentar obtener la transcripción en español (original)
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['es'])

        if max_minutes is None:
            return " ".join([entry['text'] for entry in transcript])

        max_seconds = max_minutes * 60
        text_parts = []
        current_time = 0

        for entry in transcript:
            # Usar el tiempo de inicio para decidir si incluirlo
            if entry['start'] >= max_seconds:
                break

            text_parts.append(entry['text'])
            # Actualizar current_time podría ser más preciso con start + duration
            # pero el original solo usaba start > max_seconds para cortar
            # Mantenemos la lógica original de corte por entry['start']
            # current_time = entry['start'] + entry['duration'] # Comentado para mantener original

        return " ".join(text_parts)
    except Exception as e:
        # Mantener manejo de error original
        if "No transcripts were found" in str(e) or "TranscriptsDisabled" in str(e):
            print(f"No se encontró transcripción en español o están deshabilitadas para el video {video_id}")
            return None
        print(f"Error obteniendo transcripción para {video_id}: {e}")
        return None

def get_video_comments(video_id, max_results=50):
    """Obtiene los comentarios de un video (SIN CAMBIOS)"""
    try:
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=max_results,
            textFormat="plainText",
            order="relevance" # Mantener orden original
        )
        response = request.execute()

        comments = []
        for item in response['items']:
            # Acceso original a los datos del comentario
            comment_snippet = item['snippet']['topLevelComment']['snippet']
            comments.append({
                "author": comment_snippet['authorDisplayName'],
                "text": comment_snippet['textDisplay'] # Usar textDisplay como en original
                # "publishedAt": comment_snippet['publishedAt'], # No estaban en el original
                # "likeCount": comment_snippet['likeCount'] # No estaban en el original
            })
        return comments
    except HttpError as e:
        # Mantener manejo de error original
        if e.resp.status == 403 and 'commentsDisabled' in str(e.content): # Checar e.content es más robusto
            print(f"Los comentarios están deshabilitados para el video {video_id}")
            return []
        else:
            print(f"Error HTTP {e.resp.status} al obtener comentarios para {video_id}: {e}")
            return []
    except Exception as e:
        print(f"Error inesperado obteniendo comentarios para {video_id}: {e}")
        return []

def analyze_video_content(video_id, section_content, used_video_ids):
    """Analiza si el contenido del video es relevante para la sección usando Google API"""
    if video_id in used_video_ids:
        # Mantener lógica original
        return False, "Video ya usado en otra sección"

    # Obtener transcripción completa (como en el original)
    print(f"--- Analizando Video ID: {video_id} ---")
    transcript = get_video_transcript(video_id, max_minutes=None)
    if not transcript:
        # Mantener lógica original
        return False, "No se pudo obtener la transcripción"

    # Limitar tamaño de transcripción para evitar errores de API (buena práctica añadirla)
    max_chars = 20000 # Límite generoso pero seguro
    if len(transcript) > max_chars:
        transcript = transcript[:max_chars] + "..."
        print(f"Transcripción truncada a {max_chars} caracteres para análisis.")

    try:
        # Usar Google API para analizar la relevancia
        # Mantener los prompts originales, combinándolos
        system_message = """
        Eres un experto en análisis de contenido educativo. Tu tarea es determinar si el contenido de un video es relevante para una sección específica de un curso.

        Debes analizar:
        1. Si el contenido del video coincide con el tema de la sección
        2. Si el nivel de profundidad es apropiado
        3. Si la información es precisa y relevante
        4. Si el video cubre los conceptos principales mencionados en la descripción de la sección

        Responde con un JSON que contenga:
        {
            "is_relevant": true/false,
            "confidence_score": numero entre 0 y 1,
            "reason": "explicación detallada de por qué el video es o no relevante"
        }
        """

        user_message = f"""
        Analiza si el siguiente contenido de video es relevante para esta sección del curso:

        Título y descripción de la sección:
        {section_content}

        Transcripción del video:
        {transcript}
        """
        analysis_full_prompt = f"{system_message}\n\nUSER QUESTION:\n{user_message}" # Combinar

        # Reemplazo de llamada Azure -> Google
        analysis_response_text = call_google_generative_api_for_text(analysis_full_prompt)
        # Limpiar posible markdown añadido por la API
        analysis_response_text = re.sub(r'^```json\s*', '', analysis_response_text).strip()
        analysis_response_text = re.sub(r'\s*```$', '', analysis_response_text).strip()

        # Parsear la respuesta JSON (como en el original)
        analysis = json.loads(analysis_response_text)

        # Mantener lógica de decisión original
        # Si la confianza es menor a 0.7, consideramos que el video no es lo suficientemente relevante
        if analysis.get("confidence_score", 0) < 0.7: # Usar .get con default
             print(f"Análisis video {video_id}: Rechazado por baja confianza ({analysis.get('confidence_score', 0):.2f}). Razón: {analysis.get('reason', 'N/A')}")
             return False, f"Contenido no suficientemente relevante (Confianza < 0.7): {analysis.get('reason', 'Sin razón específica')}"

        # Si es relevante y la confianza es >= 0.7
        if analysis.get("is_relevant", False): # Usar .get con default
             print(f"Análisis video {video_id}: Aprobado (Confianza: {analysis.get('confidence_score', 0):.2f}). Razón: {analysis.get('reason', 'N/A')}")
             return True, f"Video aprobado por análisis de contenido. {analysis.get('reason', 'Análisis exitoso')}"
        else:
             print(f"Análisis video {video_id}: Rechazado por IA. Razón: {analysis.get('reason', 'N/A')}")
             return False, f"Contenido no relevante según análisis: {analysis.get('reason', 'IA determinó no relevante')}"

    # Mantener manejo de errores original
    except Exception as e:
        print(f"Error en el análisis de contenido para {video_id}: {e}")
        # Considera loguear el texto si falla json.loads
        if 'analysis_response_text' in locals() and isinstance(e, json.JSONDecodeError):
             print("Texto recibido (análisis):", analysis_response_text)
        # Retornar False si hay cualquier error durante el análisis
        return False, f"Error durante el análisis de contenido: {str(e)}"

def search_youtube_videos(query, max_results=4, section_content="", used_video_ids=None):
    """Search for YouTube videos based on query and return the best match"""
    if used_video_ids is None:
        used_video_ids = set()

    try:
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

        # Primera búsqueda con licencia YouTube
        request_youtube = youtube.search().list(
            part="snippet",
            q=query,
            type="video",
            videoLicense="creativeCommon",
            maxResults=max_results * 2,
            relevanceLanguage="es",
            videoDuration="medium",
            order="relevance"
        )
        response = request_youtube.execute()

        videos = []
        approved_videos = []

        if response.get("items"):
            # Obtener IDs de video para metadatos adicionales
            video_ids = [item["id"]["videoId"] for item in response["items"]]

            # Obtener estadísticas y detalles del contenido
            video_details = youtube.videos().list(
                part="statistics,contentDetails,snippet",
                id=",".join(video_ids)
            ).execute()

            # Crear mapa de ID a detalles
            details_map = {item["id"]: item for item in video_details["items"]}

            # Procesar cada video
            for item in response["items"]:
                video_id = item["id"]["videoId"]
                video_details = details_map.get(video_id, {})
                statistics = video_details.get("statistics", {})
                content_details = video_details.get("contentDetails", {})
                snippet = video_details.get("snippet", {})

                # Extraer duración
                duration = content_details.get("duration", "PT0M0S")
                minutes_match = re.search(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration)
                if minutes_match:
                    hours = int(minutes_match.group(1) or 0)
                    minutes = int(minutes_match.group(2) or 0)
                    seconds = int(minutes_match.group(3) or 0)
                    total_minutes = (hours * 60) + minutes + (seconds / 60)
                    duration_str = f"{hours}h {minutes}m" if hours > 0 else f"{minutes} min"
                else:
                    total_minutes = 0
                    duration_str = "Desconocido"

                # Calcular días desde publicación
                published_at = datetime.strptime(snippet.get("publishedAt", ""), "%Y-%m-%dT%H:%M:%SZ")
                days_since_published = (datetime.now() - published_at).days

                # Calcular puntuación
                score = calculate_video_score(video_details, snippet, statistics, days_since_published, total_minutes)

                # Verificar contenido y relevancia
                is_approved, reason = analyze_video_content(video_id, section_content, used_video_ids)

                if is_approved:
                    videos.append({
                        "title": item["snippet"]["title"],
                        "description": item["snippet"]["description"],
                        "url": f"https://www.youtube.com/watch?v={video_id}",
                        "videoUrl": f"https://www.youtube.com/embed/{video_id}",
                        "thumbnail": item["snippet"]["thumbnails"]["high"]["url"],
                        "channelTitle": item["snippet"]["channelTitle"],
                        "publishedAt": item["snippet"]["publishedAt"],
                        "views": int(statistics.get("viewCount", 0)),
                        "likes": int(statistics.get("likeCount", 0)),
                        "comments": int(statistics.get("commentCount", 0)),
                        "duration": duration_str,
                        "score": score,
                        "videoId": video_id
                    })

            # Retornar videos ordenados por puntuación
            return sorted(videos, key=lambda x: x["score"], reverse=True)[:max_results]

        return []

    except Exception as e:
        print(f"Error searching YouTube videos: {e}")
        return []

# --- Endpoint Flask y Lógica Principal SIN CAMBIOS ---
@app.route("/solicitar_cursos", methods=["POST"])
def solicitar_cursos():
    """Main endpoint to generate a course based on a prompt (Lógica original con llamadas a IA actualizadas)"""
    start_time = time.time() # Para medir tiempo total
    try:
        data = request.json
        prompt = data.get("prompt", "").strip()

        if not prompt:
            return jsonify({"error": "El prompt no puede estar vacío"}), 400

        # Verificar API keys al inicio
        if not GOOGLE_API_KEY or not YOUTUBE_API_KEY:
            return jsonify({"error": "Error de configuración: Faltan claves API en el servidor."}), 500

        print(f"\n=== Iniciando generación de curso para prompt: '{prompt}' ===")

        # Generate course outline using Google Generative AI (ya modificado arriba)
        try:
            course_outline = get_course_outline(prompt)
            print("Esquema del curso generado con éxito.")
        except Exception as e:
             print(f"Fallo crítico: No se pudo generar el esquema del curso. Error: {e}")
             return jsonify({"error": f"Error al generar la estructura base del curso: {str(e)}"}), 500


        # Generate a random course ID (lógica original)
        course_id = f"course_{int(time.time())}" # No estaba en el original explícitamente, pero útil

        # Track used video IDs to avoid duplicates (lógica original)
        used_video_ids = set()

        # Search for introductory video (lógica original)
        introduction = None
        intro_video_data = None # Para guardar datos del video de intro
        intro_query = course_outline.get("searchQueries", {}).get("introductory", f"¿Qué es {prompt.split(' ')[0]}") # Query original
        print(f"Buscando video de introducción con query: '{intro_query}'")
        intro_videos = search_youtube_videos(
            intro_query,
            max_results=1, # Solo 1 para intro
            # Pasar descripción de la intro para análisis de IA
            section_content=course_outline.get("introduction", ""),
            used_video_ids=used_video_ids
        )

        # Prepare introduction data (lógica original)
        if intro_videos:
            best_intro_video = intro_videos[0]
            used_video_ids.add(best_intro_video["videoId"])
            intro_video_data = best_intro_video # Guardar para cálculos de duración
            introduction = {
                "content": course_outline["introduction"],
                "videoUrl": best_intro_video["videoUrl"], # URL original
                "duration": best_intro_video["duration"],
                # Añadir otros campos útiles si se desea
                "title": best_intro_video["title"],
                "videoId": best_intro_video["videoId"],
            }
            print(f"Video de introducción encontrado: {best_intro_video['videoId']}")
        else:
            print("Advertencia: No se encontró video para la introducción.")
            # Mantener intro sin video si no se encuentra
            introduction = {
                 "content": course_outline.get("introduction", "Introducción no disponible"),
                 "videoUrl": None,
                 "duration": "N/A"
            }


        # Prepare course sections with videos (lógica original)
        sections = []
        total_minutes_calculation = 0 # Para sumar duraciones

        print(f"\n--- Buscando videos para {len(course_outline.get('sections',[]))} secciones ---")
        # Add the course sections (lógica original)
        for i, section in enumerate(course_outline.get("sections", [])):
            section_id = i + 1
            print(f"Procesando Sección {section_id}: {section.get('title', 'Sin Título')}")

            # Get search query for this section (lógica original)
            section_query_key = f"section{i}"
            # Query original o fallback simple
            default_query = f"{section.get('title', '')} {prompt.split(' ')[0]}"
            section_query = course_outline.get("searchQueries", {}).get(section_query_key, default_query)

            # Search for videos for this section (lógica original)
            section_videos = search_youtube_videos(
                section_query,
                max_results=1, # Solo 1 video por sección
                # Pasar descripción de la sección para análisis de IA
                section_content=section.get("description", ""),
                used_video_ids=used_video_ids
            )

            section_data = {
                "id": section_id,
                "title": section.get("title", f"Sección {section_id}"),
                "content": section.get("description", "Contenido no disponible."),
                "videoUrl": None, # Default
                "duration": "N/A", # Default
                "classes": 1, # Mantener lógica original
                "videoId": None,
                "videoTitle": None,
            }

            if section_videos:
                best_video = section_videos[0]
                used_video_ids.add(best_video["videoId"])
                section_data.update({
                    "videoUrl": best_video["videoUrl"], # URL original
                    "duration": best_video["duration"],
                    "videoId": best_video["videoId"],
                    "videoTitle": best_video["title"],
                })
                # Acumular duración en minutos si existe
                total_minutes_calculation += best_video.get("totalMinutes", 0)
                print(f"Video encontrado para sección {section_id}: {best_video['videoId']}")
            else:
                print(f"Advertencia: No se encontró video para la sección {section_id}")
                # La sección se añade sin video

            sections.append(section_data)


        # Add final evaluation section (lógica original)
        sections.append({
            "id": len(sections) + 1,
            "title": "Evaluación Final",
            "content": "Evalúa lo aprendido en el curso.", # Contenido genérico
            "videoUrl": None, # Sin video
            "duration": "N/A",
            "classes": 1, # Cuenta como una lección/sección
            "videoId": None,
            "videoTitle": None
        })

        # Calculate total classes and estimate total duration (lógica original)
        total_classes = len(sections) # Número total de secciones incluyendo evaluación

        # Añadir duración de la intro si hubo video
        if intro_video_data:
            total_minutes_calculation += intro_video_data.get("totalMinutes", 0)

        # Formatear duración total (lógica original)
        total_duration_str = "N/A"
        if total_minutes_calculation > 0:
             total_hours = int(total_minutes_calculation // 60)
             total_mins = int(total_minutes_calculation % 60)
             if total_hours > 0:
                 total_duration_str = f"{total_hours}h {total_mins}m"
             else:
                 total_duration_str = f"{total_mins}m"


        # Prepare final course data (estructura original)
        current_date = datetime.now().strftime("%m/%Y") # Formato original
        course_data = {
            # Mantener la estructura de respuesta JSON original
            "title": course_outline.get("title", f"Curso sobre {prompt}"),
            "introduction": introduction, # El objeto de introducción preparado antes
            "instructor": "IA Professor", # Valor original
            "rating": round(random.uniform(4.5, 4.9), 1), # Lógica original
            "students": random.randint(5000, 15000), # Lógica original
            "lastUpdated": current_date, # Lógica original
            "language": "Español", # Valor original
            "totalDuration": total_duration_str, # Calculado arriba
            "totalLessons": total_classes, # Calculado arriba
            "sections": sections, # Lista de secciones preparada arriba
            "learningOutcomes": course_outline.get("learningOutcomes", []), # Del outline
            "requirements": course_outline.get("requirements", []), # Del outline
            "level": course_outline.get("level", "principiante"), # Del outline
            "level_description": course_outline.get("level_description", "") # Del outline
        }

        end_time = time.time()
        print(f"=== Generación de curso completada en {end_time - start_time:.2f} segundos ===")
        return jsonify(course_data)

    # Mantener manejo de errores original del endpoint
    except Exception as e:
        end_time = time.time()
        print(f"Error generando curso después de {end_time - start_time:.2f} segundos: {e}")
        import traceback
        traceback.print_exc()
        # Devolver error genérico como en el original
        return jsonify({"error": f"Ocurrió un error inesperado: {str(e)}"}), 500


# Add a health check endpoint (SIN CAMBIOS)
@app.route("/health", methods=["GET"])
def health_check():
    # Chequeo básico original
    youtube_ok = bool(YOUTUBE_API_KEY)
    # Ahora checa Google API Key
    google_ok = bool(GOOGLE_API_KEY)
    status = {
        "status": "ok" if youtube_ok and google_ok else "error",
        "dependencies": {
            "youtube_api": "configured" if youtube_ok else "missing_key",
            "google_generative_api": "configured" if google_ok else "missing_key"
        }
     }
    status_code = 200 if status["status"] == "ok" else 503
    return jsonify(status), status_code

# --- Ejecución Principal SIN CAMBIOS ---
if __name__ == "__main__":
    # Mantener forma original de correr la app
    port = int(os.environ.get("PORT", 5000)) # Puerto 5000 como en muchos ejemplos Flask
    app.run(host="0.0.0.0", port=port, debug=True) # debug=True como en original