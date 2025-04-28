from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import re
import random
import time
from datetime import datetime
from googleapiclient.discovery import build
from openai import AzureOpenAI
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from googleapiclient.errors import HttpError

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Environment variables
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

# Configure Azure OpenAI client
azure_client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2024-12-01-preview"
)


def get_course_outline(prompt):
    """Genera el esquema del curso basado en el prompt y el nivel"""
    # Definir los niveles disponibles y sus características
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

    # Nivel por defecto
    level = "principiante"

    # Convertir el prompt a minúsculas para la comparación
    prompt_lower = prompt.lower()

    # Detectar el nivel en el prompt
    for level_name, level_info in levels.items():
        if any(keyword in prompt_lower for keyword in level_info["keywords"]):
            level = level_name
            break

    # Eliminar palabras clave de nivel del prompt para obtener el tema principal
    topic = prompt
    for level_info in levels.values():
        for keyword in level_info["keywords"]:
            topic = topic.replace(keyword, "").strip()

    # Obtener la configuración del nivel seleccionado
    level_config = levels[level]

    try:
        # Primero, generar la introducción general
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
        
        intro_response = azure_client.chat.completions.create(
            model="gpt-4-aprendia",
            messages=[
                {"role": "system", "content": intro_system_message},
                {"role": "user", "content": f"Crea una introducción para un curso sobre: {topic}"}
            ],
            temperature=0.7
        )
        
        intro_data = json.loads(intro_response.choices[0].message.content)
        
        # Luego, generar el contenido detallado del curso
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
            ],
            "learningOutcomes": ["Objetivo 1", "Objetivo 2", ...],
            "requirements": ["Requisito 1", "Requisito 2", ...],
            "level": "{level}",
            "level_description": "{level_config['description']}",
            "total_sections": {level_config['num_sections']}
        }}
        """
        
        content_response = azure_client.chat.completions.create(
            model="gpt-4-aprendia",
            messages=[
                {"role": "system", "content": content_system_message},
                {"role": "user", "content": f"Crea un curso sobre: {topic}"}
            ],
            temperature=0.7
        )
        
        content_data = json.loads(content_response.choices[0].message.content)
        
        # Combinar la introducción con el contenido del curso
        course_outline = {
            **content_data,
            "introduction": intro_data["introduction"]
        }
        
        return course_outline

    except Exception as e:
        print(f"Error generando esquema del curso: {e}")
        raise


def calculate_video_score(video_details, snippet, statistics, days_since_published, total_minutes):
    """Calcula la puntuación de un video basada en múltiples criterios"""
    # 1. Relevancia (30%)
    relevance_score = 1.0  # Base score, se puede ajustar según el título/descripción

    # 2. Calidad del video (25%)
    quality_score = 0
    if "HD" in snippet.get("tags", []):
        quality_score += 0.3
    if "4K" in snippet.get("tags", []):
        quality_score += 0.4
    if "1080p" in snippet.get("tags", []):
        quality_score += 0.3

    # 3. Engagement (20%)
    views = int(statistics.get("viewCount", 0))
    likes = int(statistics.get("likeCount", 0))
    comments = int(statistics.get("commentCount", 0))

    engagement_score = 0
    if views > 0:
        like_ratio = likes / views
        comment_ratio = comments / views
        engagement_score = (like_ratio * 0.6) + (comment_ratio * 0.4)

    # 4. Actualidad (15%)
    recency_score = 1.0 if days_since_published <= 365 else 0.5

    # 5. Duración (10%)
    duration_score = 1.0 if 5 <= total_minutes <= 20 else 0.5

    # Cálculo de la puntuación final con pesos
    final_score = (
            relevance_score * 0.30 +
            quality_score * 0.25 +
            engagement_score * 0.20 +
            recency_score * 0.15 +
            duration_score * 0.10
    )

    return final_score


def get_video_transcript(video_id, max_minutes=None):
    """Obtiene la transcripción de un video de YouTube en español"""
    try:
        # Intentar obtener la transcripción en español
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['es'])

        if max_minutes is None:
            # Si no se especifica max_minutes, devolver toda la transcripción
            return " ".join([entry['text'] for entry in transcript])

        # Calcular el tiempo máximo en segundos
        max_seconds = max_minutes * 60

        # Filtrar y unir el texto hasta el tiempo máximo
        text_parts = []
        current_time = 0

        for entry in transcript:
            if current_time >= max_seconds:
                break

            text_parts.append(entry['text'])
            current_time = entry['start'] + entry['duration']

        return " ".join(text_parts)
    except Exception as e:
        if "No transcripts were found" in str(e):
            print(f"No se encontró transcripción en español para el video {video_id}")
            return None
        print(f"Error obteniendo transcripción: {e}")
        return None


def get_video_comments(video_id, max_results=50):
    """Obtiene los comentarios de un video"""
    try:
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=max_results,
            textFormat="plainText",
            order="relevance"
        )
        response = request.execute()

        comments = []
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            author = item['snippet']['topLevelComment']['snippet']['authorDisplayName']
            comments.append({
                "author": author,
                "text": comment
            })
        return comments
    except HttpError as e:
        if e.resp.status == 403 and 'commentsDisabled' in str(e):
            print(f"Los comentarios están deshabilitados para el video {video_id}")
            return []  # Retornar lista vacía en lugar de fallar
        else:
            print(f"Error HTTP al obtener comentarios: {e}")
            return []
    except Exception as e:
        print(f"Error obteniendo comentarios: {e}")
        return []


def analyze_video_content(video_id, section_content, used_video_ids):
    """Analiza si el contenido del video es relevante para la sección usando Azure OpenAI"""
    if video_id in used_video_ids:
        return False, "Video ya usado en otra sección"

    # Obtener transcripción completa
    transcript = get_video_transcript(video_id, max_minutes=None)  # Obtener toda la transcripción
    if not transcript:
        return False, "No se pudo obtener la transcripción"

    try:
        # Usar Azure OpenAI para analizar la relevancia
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
            "confidence_score": número entre 0 y 1,
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

        response = azure_client.chat.completions.create(
            model="gpt-4-aprendia",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3
        )

        # Extraer y parsear la respuesta JSON
        analysis = json.loads(response.choices[0].message.content)

        # Si la confianza es menor a 0.7, consideramos que el video no es lo suficientemente relevante
        if analysis["confidence_score"] < 0.7:
            return False, f"Contenido no suficientemente relevante: {analysis['reason']}"

        return True, "Video aprobado por análisis de contenido"

    except Exception as e:
        print(f"Error en el análisis de contenido: {e}")
        return False, f"Error en el análisis de contenido: {str(e)}"


def search_youtube_videos(query, max_results=5, section_content="", used_video_ids=None):
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


@app.route("/solicitar_cursos", methods=["POST"])
def solicitar_cursos():
    """Main endpoint to generate a course based on a prompt"""
    try:
        data = request.json
        prompt = data.get("prompt", "")

        if not prompt:
            return jsonify({"error": "El prompt no puede estar vacío"}), 400

        # Generate course outline using Azure OpenAI
        course_outline = get_course_outline(prompt)

        # Generate a random course ID
        course_id = f"course_{int(time.time())}"

        # Track used video IDs to avoid duplicates
        used_video_ids = set()

        # Search for introductory video
        intro_query = course_outline.get("searchQueries", {}).get("introductory", f"¿Qué es {prompt}")
        print(intro_query)
        intro_videos = search_youtube_videos(
            intro_query, 
            max_results=3,
            section_content=course_outline["introduction"],
            used_video_ids=used_video_ids
        )

        # Prepare introduction data
        introduction = None
        if intro_videos:
            best_intro_video = intro_videos[0]
            used_video_ids.add(best_intro_video["videoId"])
            introduction = {
                "content": course_outline["introduction"],
                "videoUrl": best_intro_video["videoUrl"],
                "duration": best_intro_video["duration"]
            }

        # Prepare course sections with videos
        sections = []

        # Add the course sections
        for i, section in enumerate(course_outline["sections"]):
            section_id = i + 1

            # Get search query for this section
            section_query = course_outline.get("searchQueries", {}).get(f"section{i}", f"{section['title']} {prompt}")

            # Search for videos for this section
            section_videos = search_youtube_videos(
                section_query, 
                max_results=3,
                section_content=section["description"],
                used_video_ids=used_video_ids
            )

            if section_videos:
                best_video = section_videos[0]
                used_video_ids.add(best_video["videoId"])
                sections.append({
                    "id": section_id,
                    "title": section["title"],
                    "content": section["description"],
                    "videoUrl": best_video["videoUrl"],
                    "duration": best_video["duration"],
                    "classes": 1
                })

        # Add final evaluation section
        sections.append({
            "id": len(sections) + 1,
            "title": "Evaluación Final",
            "classes": 1
        })

        # Calculate total classes and estimate total duration
        total_classes = len(sections)

        # Calculate total duration in minutes
        total_minutes = 0
        for section in sections:
            if "duration" in section:
                duration_text = section["duration"]
                # Extract hours and minutes
                hours_match = re.search(r'(\d+)h', duration_text)
                minutes_match = re.search(r'(\d+) min|(\d+)m', duration_text)

                hours = int(hours_match.group(1)) if hours_match else 0
                if minutes_match:
                    minutes = int(minutes_match.group(1) or minutes_match.group(2) or 0)
                else:
                    minutes = 0

                total_minutes += (hours * 60) + minutes

        total_duration = f"{total_minutes // 60}h {total_minutes % 60}m" if total_minutes >= 60 else f"{total_minutes}m"

        # Prepare final course data
        current_date = datetime.now().strftime("%m/%Y")
        course_data = {
            "title": course_outline["title"],
            "introduction": introduction,
            "instructor": "IA Professor",
            "rating": round(random.uniform(4.5, 4.9), 1),
            "students": random.randint(5000, 15000),
            "lastUpdated": current_date,
            "language": "Español",
            "totalDuration": total_duration,
            "totalLessons": total_classes,
            "sections": sections,
            "learningOutcomes": course_outline["learningOutcomes"],
            "requirements": course_outline["requirements"],
            "level": course_outline.get("level", "principiante"),
            "level_description": course_outline.get("level_description", "")
        }

        return jsonify(course_data)

    except Exception as e:
        print(f"Error generating course: {e}")
        return jsonify({"error": str(e)}), 500


# Add a health check endpoint
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True)