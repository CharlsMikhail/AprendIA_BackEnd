from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import os
import json
import re
import random
import time
from datetime import datetime
from googleapiclient.discovery import build
import requests
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.errors import HttpError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from textblob import TextBlob
from spellchecker import SpellChecker
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
import spacy
import concurrent.futures
from pysentimiento import create_analyzer
import emoji

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# --- Environment variables ---
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
# AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT") # Eliminado    
# AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY") # Eliminado
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Añadido

# --- Google Generative AI API Configuration ---
# Puedes cambiar 'gemini-1.5-flash-latest' si necesitas otro modelo
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash-preview-05-20")
#GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash-latest")
GOOGLE_API_URL_TEMPLATE = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent"

# Descargar recursos necesarios de NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')

# Cargar modelo de spaCy para español
try:
    nlp = spacy.load("es_core_news_sm")
except:
    print("Modelo de spaCy no encontrado. Por favor, ejecuta: python -m spacy download es_core_news_sm")

# Diccionario de correcciones comunes
CORRECCIONES_COMUNES = {
    "ke": "que",
    "q": "que",
    "xq": "porque",
    "pq": "porque",
    "tb": "también",
    "tmb": "también",
    "tambn": "también",
    "x": "por",
    "d": "de",
    "k": "que",
    "q": "que",
    "w": "con",
    "c": "con",
    "m": "me",
    "t": "te",
    "s": "es",
    "xfa": "por favor",
    "pls": "por favor",
    "plz": "por favor",
    "thx": "gracias",
    "ty": "gracias",
    "np": "no problem",
    "yw": "de nada",
    "nw": "de nada",
    "np": "no hay problema",
    "nvm": "no importa",
    "idk": "no sé",
    "idc": "no me importa",
    "tbh": "para ser honesto",
    "imo": "en mi opinión",
    "imho": "en mi humilde opinión",
    "afaik": "por lo que sé",
    "afaict": "por lo que puedo ver",
    "afaics": "por lo que puedo ver",
    "afaict": "por lo que puedo ver",
    "afaics": "por lo que puedo ver",
    "afaict": "por lo que puedo ver",
    "afaics": "por lo que puedo ver",
    "afaict": "por lo que puedo ver",
    "afaics": "por lo que puedo ver",
}

# Al inicio del archivo (después de imports)
analyzer_sentimiento = create_analyzer(task="sentiment", lang="es")

def procesar_transcripcion(texto):
    """
    Procesa y limpia una transcripción de video.
    """
    try:
        # 1. Limpieza básica
        # Eliminar timestamps y marcas de subtítulos
        texto = re.sub(r'\[\d{2}:\d{2}:\d{2}\]|\[\d{2}:\d{2}\]', '', texto)
        texto = re.sub(r'\[.*?\]', '', texto)  # Eliminar [Música], [Aplausos], etc.

        # Eliminar caracteres especiales y emojis
        texto = re.sub(r'[^\w\s]', ' ', texto)

        # Convertir a minúsculas
        texto = texto.lower()

        # 2. Corrección de errores comunes
        # Aplicar diccionario de correcciones
        palabras = texto.split()
        palabras_corregidas = [CORRECCIONES_COMUNES.get(palabra, palabra) for palabra in palabras]
        texto = ' '.join(palabras_corregidas)

        # Usar SpellChecker para correcciones adicionales
        spell = SpellChecker(language='es')
        palabras = texto.split()
        palabras_corregidas = []
        for palabra in palabras:
            if len(palabra) > 2:  # Ignorar palabras muy cortas
                correccion = spell.correction(palabra)
                palabras_corregidas.append(correccion if correccion else palabra)
            else:
                palabras_corregidas.append(palabra)
        texto = ' '.join(palabras_corregidas)

        # 3. Eliminar stopwords
        stop_words = set(stopwords.words('spanish'))
        palabras = word_tokenize(texto)
        palabras_filtradas = [palabra for palabra in palabras if palabra not in stop_words]
        texto = ' '.join(palabras_filtradas)

        # 4. Eliminar redundancias
        # Usar TextBlob para detectar y eliminar repeticiones
        blob = TextBlob(texto)
        oraciones = blob.sentences
        oraciones_unicas = []
        for oracion in oraciones:
            if oracion not in oraciones_unicas:
                oraciones_unicas.append(str(oracion))
        texto = ' '.join(oraciones_unicas)

        # 5. Resumen del texto
        # Usar TextRank para extraer las oraciones más importantes
        parser = PlaintextParser.from_string(texto, Tokenizer("spanish"))
        summarizer = TextRankSummarizer()
        resumen = summarizer(parser.document, sentences_count=5)  # Obtener 5 oraciones más importantes
        texto = ' '.join([str(sentence) for sentence in resumen])

        # 6. Análisis de relevancia por segmentos
        # Dividir en segmentos de 30 segundos (aproximadamente)
        segmentos = sent_tokenize(texto)
        segmentos_relevantes = []

        # Calcular TF-IDF para cada segmento
        vectorizer = TfidfVectorizer(stop_words='spanish')
        try:
            tfidf_matrix = vectorizer.fit_transform(segmentos)
            # Calcular la densidad de palabras clave para cada segmento
            densidades = np.array(tfidf_matrix.sum(axis=1)).flatten()
            # Seleccionar segmentos con alta densidad
            umbral = np.mean(densidades)
            for i, densidad in enumerate(densidades):
                if densidad > umbral:
                    segmentos_relevantes.append(segmentos[i])
        except:
            # Si hay error en el cálculo de TF-IDF, usar todos los segmentos
            segmentos_relevantes = segmentos

        texto = ' '.join(segmentos_relevantes)

        return texto

    except Exception as e:
        print(f"Error procesando transcripción: {e}")
        return texto  # Devolver texto original si hay error


def get_video_transcript(video_id, max_minutes=None):
    """Obtiene la transcripción de un video de YouTube en español"""
    try:
        # Intentar obtener la transcripción en español
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['es'])

        if max_minutes is None:
            # Procesar la transcripción completa
            texto_completo = " ".join([entry['text'] for entry in transcript])
            return procesar_transcripcion(texto_completo)

        max_seconds = max_minutes * 60
        text_parts = []
        current_time = 0

        for entry in transcript:
            if entry['start'] >= max_seconds:
                break
            text_parts.append(entry['text'])

        # Procesar la transcripción parcial
        texto_parcial = " ".join(text_parts)
        return procesar_transcripcion(texto_parcial)

    except Exception as e:
        if "No transcripts were found" in str(e) or "TranscriptsDisabled" in str(e):
            print(f"No se encontró transcripción en español o están deshabilitadas para el video {video_id}")
            return None
        print(f"Error obteniendo transcripción para {video_id}: {e}")
        return None


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
        raise  # Relanzar el error para que sea capturado por el llamador original
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

    # Nivel por defecto
    level = "principiante"

    # Convertir el prompt a minúsculas para la comparación
    prompt_lower = prompt.lower()

    # Detectar el nivel en el prompt
    for level_name, level_info in levels.items():
        if any(keyword in prompt_lower for keyword in level_info["keywords"]):
            level = level_name
            break

    # Extraer el tema principal del prompt
    topic_extraction_prompt = f"""
    Analiza el siguiente prompt y extrae el tema principal del curso. 
    El tema principal debe ser una palabra o frase corta que identifique específicamente de qué trata el curso.
    Por ejemplo:
    - "curso de Java para principiantes porque quiero aprender programación" -> "Java"
    - "quiero aprender a tocar la guitarra desde cero" -> "Guitarra"
    - "curso de cocina italiana para principiantes" -> "Cocina italiana"
    
    Prompt: {prompt}
    
    Responde SOLO con el tema principal, sin explicaciones adicionales.
    """
    
    try:
        topic = call_google_generative_api_for_text(topic_extraction_prompt).strip()
    except Exception as e:
        print(f"Error extrayendo tema principal: {e}")
        # Fallback: eliminar palabras clave de nivel del prompt
        topic = prompt
        for level_info in levels.values():
            for keyword in level_info["keywords"]:
                topic = topic.replace(keyword, "").strip()

    # Obtener la configuración del nivel seleccionado
    level_config = levels[level]
    max_workers_outline = min(32, (os.cpu_count() or 1) * 2) # Para paralelizar llamadas a IA aquí

    try:
        # Preparar los prompts para introducción y contenido
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
        intro_full_prompt = f"{intro_system_message}\n\nUSER QUESTION:\n{intro_user_message}"  # Combinar

        content_system_message = f"""
        Eres un experto en diseño de cursos educativos. Tu tarea es crear un esquema detallado para un curso de nivel {level} sobre {topic}.
        
        El curso debe incluir:
        1. Un título atractivo y descriptivo que refleje el nivel {level}
        2. {level_config['num_sections']} secciones principales, cada una con:
           - Título descriptivo que incluya el tema principal ({topic})
           - Descripción detallada del contenido
        3. Objetivos de aprendizaje específicos al nivel {level}
        4. Requisitos previos apropiados para el nivel
        
        Para un curso de nivel {level}, asegúrate de:
        - {level_config['description']}
        - Mantener una profundidad {level_config['depth']} en los temas
        - Enfocarse en {level_config['focus']}
        - {f"Incluir ejercicios prácticos y proyectos" if level != "principiante" else "Incluir ejemplos simples y ejercicios guiados"}
        - {f"Cubrir temas especializados y técnicas avanzadas" if level in ["avanzado", "maestro"] else "Mantener un enfoque en conceptos fundamentales"}
        
        IMPORTANTE: 
        1. No incluyas ninguna referencia a videos, URLs o contenido multimedia en la respuesta.
        2. Cada título de sección DEBE incluir el tema principal ({topic}) para asegurar que los videos encontrados sean relevantes.
        
        Proporciona la respuesta en formato JSON con la siguiente estructura:
        {{
            "title": "Título del curso",
            "sections": [
                {{
                    "title": "Título de la sección (incluyendo {topic})",
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
        content_user_message = f"Crea un curso sobre: {topic}"
        content_full_prompt = f"{content_system_message}\n\nUSER QUESTION:\n{content_user_message}"  # Combinar

        intro_data = None
        content_data = None

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers_outline) as executor:
            future_intro = executor.submit(call_google_generative_api_for_text, intro_full_prompt)
            future_content = executor.submit(call_google_generative_api_for_text, content_full_prompt)

            try:
                intro_response_text = future_intro.result()
                intro_response_text = re.sub(r'^```json\s*', '', intro_response_text).strip()
                intro_response_text = re.sub(r'\s*```$', '', intro_response_text).strip()
                intro_data = json.loads(intro_response_text)
            except Exception as e_intro:
                print(f"Error generando introducción del curso en paralelo: {e_intro}")
                # Podríamos decidir si continuar sin introducción o lanzar el error
                raise  # Por ahora, relanzamos si falla la introducción

            try:
                content_response_text = future_content.result()
                content_response_text = re.sub(r'^```json\s*', '', content_response_text).strip()
                content_response_text = re.sub(r'\s*```$', '', content_response_text).strip()
                content_data = json.loads(content_response_text)
            except Exception as e_content:
                print(f"Error generando contenido del curso en paralelo: {e_content}")
                # Podríamos decidir si continuar sin contenido o lanzar el error
                raise # Por ahora, relanzamos si falla el contenido
        
        # Combinar la introducción con el contenido del curso
        course_outline = {
            **content_data, # Asegurarse que content_data no sea None
            "introduction": intro_data["introduction"], # Asegurarse que intro_data no sea None
            "extracted_topic": topic
        }
        
        # Generar queries de búsqueda mejoradas
        course_outline["searchQueries"] = {
            "introductory": f"introducción a {topic} para {level}s",
            **{f"section{i}": f"{section['title']} {topic} tutorial {level}" for i, section in enumerate(course_outline["sections"])}
        }
        
        return course_outline
                
    except Exception as e:
        print(f"Error generando esquema del curso: {e}")
        raise


# --- Resto de las funciones SIN CAMBIOS ---

def calculate_video_score(video_details, snippet, statistics, days_since_published, total_minutes, section_content=""):
    """Calcula la puntuación de un video basada en múltiples criterios"""
    try:
        # 1. Relevancia (30%) - Ahora usando similitud de texto
        relevance_score = 0.0
        if section_content:
            # Combinar título y descripción del video
            video_text = f"{snippet.get('title', '')} {snippet.get('description', '')}"

            # Crear vectorizador TF-IDF
            vectorizer = TfidfVectorizer(stop_words='english')
            try:
                # Convertir textos a vectores TF-IDF
                tfidf_matrix = vectorizer.fit_transform([video_text, section_content])
                # Calcular similitud coseno
                similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                relevance_score = float(similarity)
            except Exception as e:
                print(f"Error calculando similitud: {e}")
                relevance_score = 0.5  # Valor por defecto si hay error
        else:
            relevance_score = 0.5  # Valor por defecto si no hay contenido de sección

        # 2. Verificación de transcripción (20%)
        transcript_score = 0.0
        try:
            # Intentar obtener la transcripción sin descargarla completamente
            transcript_list = YouTubeTranscriptApi.list_transcripts(snippet.get('id', {}).get('videoId', ''))
            # Verificar si existe transcripción en español
            if transcript_list.find_transcript(['es']):
                transcript_score = 1.0
            else:
                # Si no hay en español, verificar si hay en otros idiomas
                available_transcripts = transcript_list.find_manually_created_transcript()
                if available_transcripts:
                    transcript_score = 0.5  # Medio punto si hay transcripción en otro idioma
        except Exception as e:
            print(f"Error verificando transcripción: {e}")
            transcript_score = 0.0

        # 3. Calidad del video (15%)
        quality_score = 0
        definition = video_details.get('contentDetails', {}).get('definition')
        if definition == 'hd':
            quality_score = 0.8
        elif definition == 'sd':
            quality_score = 0.4
        else:
            quality_score = 0.3

        # 4. Engagement (15%)
        views = int(statistics.get("viewCount", 0))
        likes = int(statistics.get("likeCount", 0))
        comments = int(statistics.get("commentCount", 0))

        engagement_score = 0
        if views > 100:
            like_ratio = (likes / views)
            comment_ratio = (comments / views)
            engagement_score = min(1.0, (like_ratio * 5) * 0.6 + (comment_ratio * 20) * 0.4)

        # 5. Actualidad (10%)
        recency_score = 1.0 if days_since_published <= 365 else (
            0.5 if days_since_published <= 730 else 0.2)

        # 6. Duración (10%) - Ideal entre 5 y 25 mins
        duration_score = 1.0 if 5 <= total_minutes <= 25 else (
            0.5 if total_minutes < 5 else 0.7)

        # Cálculo de la puntuación final con nuevos pesos
        final_score = (
                relevance_score * 0.30 +
                transcript_score * 0.20 +
                quality_score * 0.15 +
                engagement_score * 0.15 +
                recency_score * 0.10 +
                duration_score * 0.10
        )

        # Si no hay transcripción disponible, penalizar significativamente
        if transcript_score == 0:
            final_score *= 0.5

        return final_score

    except Exception as e:
        print(f"Error calculando score del video: {e}")
        return 0.0  # Retornar 0 si hay algún error en el cálculo


def get_video_comments(video_id, max_results=50):
    """Obtiene los comentarios de un video (SIN CAMBIOS)"""
    try:
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=max_results,
            textFormat="plainText",
            order="relevance"  # Mantener orden original
        )
        response = request.execute()

        comments = []
        for item in response['items']:
            # Acceso original a los datos del comentario
            comment_snippet = item['snippet']['topLevelComment']['snippet']
            comments.append({
                "author": comment_snippet['authorDisplayName'],
                "text": comment_snippet['textDisplay']  # Usar textDisplay como en original
                # "publishedAt": comment_snippet['publishedAt'], # No estaban en el original
                # "likeCount": comment_snippet['likeCount'] # No estaban en el original
            })
        return comments
    except HttpError as e:
        # Mantener manejo de error original
        if e.resp.status == 403 and 'commentsDisabled' in str(e.content):  # Checar e.content es más robusto
            print(f"Los comentarios están deshabilitados para el video {video_id}")
            return []
        else:
            print(f"Error HTTP {e.resp.status} al obtener comentarios para {video_id}: {e}")
            return []
    except Exception as e:
        print(f"Error inesperado obteniendo comentarios para {video_id}: {e}")
        return []


def verificar_transcripcion_disponible(video_id):
    """Verifica si un video tiene transcripción disponible sin descargarla"""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        # Verificar si existe transcripción en español
        if transcript_list.find_transcript(['es']):
            return True
        # Si no hay en español, verificar si hay en otros idiomas
        available_transcripts = transcript_list.find_manually_created_transcript()
        return bool(available_transcripts)
    except Exception as e:
        print(f"Error verificando transcripción para video {video_id}: {e}")
        return False


def search_youtube_videos(query, max_results=4, section_content="", used_video_ids=None):
    """Search for YouTube videos based on query and return the best match"""
    if used_video_ids is None:
        used_video_ids = set()

    # Determinar el número de hilos
    max_workers = min(32, (os.cpu_count() or 1) * 2)

    try:
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

        # Primera búsqueda con licencia YouTube, retrieving up to 10 videos (antes 50)
        request_youtube = youtube.search().list(
            part="snippet",
            q=query,
            type="video",
            videoLicense="creativeCommon",
            maxResults=10,  # Cambiado de 50 a 10
            relevanceLanguage="es",
            videoDuration="medium",
            order="relevance"
        )
        response = request_youtube.execute()

        videos = []
        all_videos_data = []
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

                # Verificar si tiene transcripción disponible
                if not verificar_transcripcion_disponible(video_id):
                    print(f"Video {video_id} descartado: No tiene transcripción disponible")
                    continue

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

                # Calcular puntuación inicial
                score = calculate_video_score(video_details, snippet, statistics, days_since_published, total_minutes)

                # Collect all relevant video data
                video_data = {
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
                    "videoId": video_id,
                    "totalMinutes": total_minutes,
                    "full_snippet": snippet,
                    "full_statistics": statistics,
                    "full_contentDetails": content_details
                }

                all_videos_data.append(video_data)

            # Ordenar los videos por score y tomar los 5 mejores (antes 10)
            sorted_videos = sorted(all_videos_data, key=lambda x: x["score"], reverse=True)
            top_5_videos = sorted_videos[:5] # Cambiado de top_10_videos a top_5_videos y slice a :5

            # Procesar las transcripciones de los 5 mejores videos en paralelo
            videos_with_transcripts = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_video = {executor.submit(get_video_transcript, video["videoId"]): video for video in top_5_videos} # Usar top_5_videos
                for future in concurrent.futures.as_completed(future_to_video):
                    video = future_to_video[future]
                    try:
                        transcript = future.result()
                        if transcript:
                            video["processed_transcript"] = transcript
                            videos_with_transcripts.append(video)
                        else:
                            print(f"Video {video['videoId']} descartado: No se pudo obtener transcripción en el procesamiento paralelo.")
                    except Exception as exc:
                        print(f"Error obteniendo transcripción para {video['videoId']} en paralelo: {exc}")
            
            return videos_with_transcripts # Devolver solo los videos con transcripción exitosa

        return []

    except Exception as e:
        print(f"Error searching YouTube videos: {e}")
        return []


def limpiar_comentarios(comentarios):
    """Limpia y normaliza una lista de comentarios."""
    comentarios_limpios = set()
    for comentario in comentarios:
        texto = comentario.get("text", "")
        # Eliminar emojis
        texto = emoji.replace_emoji(texto, replace="")
        # Eliminar símbolos y caracteres especiales
        texto = re.sub(r'[^\w\s]', '', texto)
        # Convertir a minúsculas y eliminar espacios extra
        texto = texto.lower().strip()
        if texto:
            comentarios_limpios.add(texto)
    return list(comentarios_limpios)


def analizar_sentimiento_comentarios(comentarios_limpios, analyzer):
    """Analiza el sentimiento de los comentarios usando pysentimiento y retorna resumen."""
    if not comentarios_limpios:
        return {
            "resumen": "Sin comentarios",
            "proporcion": {"POS": 0, "NEU": 0, "NEG": 0},
            "total": 0
        }
    resultados = {"POS": 0, "NEU": 0, "NEG": 0}
    for comentario in comentarios_limpios:
        resultado = analyzer.predict(comentario)
        resultados[resultado.output.upper()] += 1
    total = len(comentarios_limpios)
    proporcion = {k: v / total for k, v in resultados.items()}
    # Resumen general
    mayor = max(proporcion, key=proporcion.get)
    resumen = {
        "POS": "mayoría positiva",
        "NEU": "mayoría neutral",
        "NEG": "mayoría negativa"
    }[mayor]
    return {
        "resumen": resumen,
        "proporcion": proporcion,
        "total": total
    }


def analyze_video_content(video_id, section_content, used_video_ids, course_topic, video_youtube_title, processed_transcript_text=None, analyzer=analyzer_sentimiento):
    """Analiza si el contenido del video es relevante para la sección usando Google API y sentimiento de comentarios."""
    if video_id in used_video_ids:
        return False, "Video ya usado en otra sección", None

    analysis_response_json = None
    try:
        transcript = processed_transcript_text
        if not transcript:
            print(f"No se proveyó transcripción para {video_id} (Título: '{video_youtube_title}', Tema Curso: '{course_topic}'), obteniéndola ahora...")
            transcript = get_video_transcript(video_id)
            if not transcript:
                return False, "No se pudo obtener la transcripción", None

        # Obtener y procesar comentarios
        comentarios = get_video_comments(video_id, max_results=50)
        comentarios_limpios = limpiar_comentarios(comentarios)
        sentimiento = analizar_sentimiento_comentarios(comentarios_limpios, analyzer)

        # Preparar el prompt para la IA
        system_message = f"""
Eres un experto en análisis de contenido educativo. Tu tarea es determinar si el contenido de un video (considerando su título original de YouTube, su transcripción y el sentimiento de los comentarios) es relevante para una sección específica de un curso sobre '{course_topic}'.

Debes analizar:
1. Si el contenido del video (título y transcripción) coincide con el tema de la sección descrito abajo.
2. Si el nivel de profundidad es apropiado.
3. Si la información es precisa y relevante para '{course_topic}'.
4. Si el video (título y transcripción) cubre los conceptos principales mencionados en la descripción de la sección, en el contexto de '{course_topic}'.
5. Crucialmente, verifica que el video (título y transcripción) trate específicamente sobre '{course_topic}' y no sobre temas relacionados pero distintos.
6. Analiza el sentimiento general de los comentarios: si la mayoría es negativa, penaliza el video; si es positiva, súmalo al score.

IMPORTANTE: El título debe ser del tema correcto (por ejemplo, si la sección es de variables en Java, rechaza si el título es de variables en C++ o C).

Asigna un score de relevancia total considerando:
- Título: 30%
- Transcripción: 50%
- Sentimiento de comentarios: 20%

Responde con un JSON que contenga:
{{
    'is_relevant': true/false,
    'confidence_score': número entre 0 y 1,  # Score total considerando los pesos
    'reason': "explicación detallada de por qué el video es o no relevante, considerando específicamente el tema del curso '{course_topic}', el título del video, su transcripción y el sentimiento de los comentarios.",
    'topic_match': true/false, # True si el video (título y transcripción) es sobre '{course_topic}', False en caso contrario.
    'topic_match_score': número entre 0 y 1, # Confianza en que el video (título y transcripción) trata sobre '{course_topic}'
    'sentiment_summary': "{sentimiento['resumen']}",
    'sentiment_proportion': {sentimiento['proporcion']},
    'sentiment_total_comments': {sentimiento['total']}
}}
"""

        user_message = f"""
Analiza si el siguiente contenido de video es relevante para esta sección del curso:

Título y descripción de la sección:
{section_content}

Título original del video de YouTube:
{video_youtube_title}

Transcripción procesada del video:
{transcript}
"""
        analysis_full_prompt = f"{system_message}\n\nUSER QUESTION:\n{user_message}"

        # Llamar a la API de Google
        analysis_response_text = call_google_generative_api_for_text(analysis_full_prompt)
        analysis_response_text = re.sub(r'^```json\s*', '', analysis_response_text).strip()
        analysis_response_text = re.sub(r'\s*```$', '', analysis_response_text).strip()
        analysis_response_json = json.loads(analysis_response_text)

        # Verificar si el tema coincide específicamente (título estricto)
        if not analysis_response_json.get("topic_match", False):
            print(f"Análisis video {video_id}: Rechazado por tema incorrecto. Razón: {analysis_response_json.get('reason', 'N/A')}")
            return False, f"El video no trata específicamente del tema correcto: {analysis_response_json.get('reason', 'Sin razón específica')}", analysis_response_json

        # Si el tema coincide pero la confianza es baja, aún podríamos aceptarlo como último recurso
        if analysis_response_json.get("confidence_score", 0) < 0.7:
            if analysis_response_json.get("topic_match_score", 0) >= 0.8:  # Si el tema coincide bien
                print(f"Análisis video {video_id}: Aceptado con baja confianza pero tema correcto (Confianza: {analysis_response_json.get('confidence_score', 0):.2f})")
                return True, f"Video aceptado como alternativa. {analysis_response_json.get('reason', 'Análisis exitoso')}", analysis_response_json
            else:
                print(f"Análisis video {video_id}: Rechazado por baja confianza ({analysis_response_json.get('confidence_score', 0):.2f}). Razón: {analysis_response_json.get('reason', 'N/A')}")
                return False, f"Contenido no suficientemente relevante (Confianza < 0.7): {analysis_response_json.get('reason', 'Sin razón específica')}", analysis_response_json

        if analysis_response_json.get("is_relevant", False):
            print(f"Análisis video {video_id}: Aprobado (Confianza: {analysis_response_json.get('confidence_score', 0):.2f}). Razón: {analysis_response_json.get('reason', 'N/A')}")
            return True, f"Video aprobado por análisis de contenido. {analysis_response_json.get('reason', 'Análisis exitoso')}", analysis_response_json
        else:
            print(f"Análisis video {video_id}: Rechazado por IA. Razón: {analysis_response_json.get('reason', 'N/A')}")
            return False, f"Contenido no relevante según análisis: {analysis_response_json.get('reason', 'IA determinó no relevante')}", analysis_response_json

    except Exception as e:
        print(f"Error en el análisis de contenido para {video_id}: {e}")
        if 'analysis_response_text' in locals() and isinstance(e, json.JSONDecodeError):
            print("Texto recibido (análisis):", analysis_response_text)
        return False, f"Error durante el análisis de contenido: {str(e)}", analysis_response_json # Devolver analysis_response_json aunque haya error para inspección


def process_section_parallel(section_data, section_index, topic, used_video_ids, analyzer):
    """Procesa una sección del curso en paralelo, incluyendo búsqueda y análisis de videos."""
    section_id = section_index + 1
    print(f"Procesando Sección {section_id}: {section_data.get('title', 'Sin Título')}")

    section_title_full = section_data.get('title', '')
            
    # Lógica de simplificación de la consulta principal
    simplified_title_parts = section_title_full.split(':')[0].strip()
    if not simplified_title_parts or len(simplified_title_parts.split()) < 2 or simplified_title_parts == section_title_full:
        simplified_title_parts = ' '.join(section_title_full.split()[:4]).strip()
            
    # Construir la consulta principal simplificada con el topic
    if topic.lower() in simplified_title_parts.lower():
        section_query = simplified_title_parts
    else:
        section_query = f"{simplified_title_parts} {topic}"
    section_query = section_query.strip()

    print(f"Buscando videos para Sección {section_id} con query principal simplificada: '{section_query}'")
    all_section_videos_data = search_youtube_videos(
        section_query,
        section_content=section_data.get("description", ""),
        used_video_ids=used_video_ids
    )

    if not all_section_videos_data:
        # Alternativa 1: Usar el título completo de la sección + topic
        alt_query_1 = f"{section_title_full} {topic}".strip()
        print(f"Query simplificada falló. Intentando alternativa 1 (título completo) para Sección {section_id}: '{alt_query_1}'")
        all_section_videos_data = search_youtube_videos(
            alt_query_1,
            section_content=section_data.get("description", ""),
            used_video_ids=used_video_ids
        )

        if not all_section_videos_data:
            # Alternativa 2: Usar solo el topic y el nivel
            alt_query_2 = f"{topic} {section_data.get('level', 'principiante')}".strip()
            print(f"Alternativa 1 falló. Intentando alternativa 2 (solo tema y nivel) para Sección {section_id}: '{alt_query_2}'")
            all_section_videos_data = search_youtube_videos(
                alt_query_2,
                section_content=section_data.get("description", ""),
                used_video_ids=used_video_ids
            )
            
    section_result = {
        "id": section_id,
        "title": section_data.get("title", f"Sección {section_id}"),
        "content": section_data.get("description", "Contenido no disponible."),
        "videoUrl": None,
        "duration": "N/A",
        "classes": 1,
        "videoId": None,
        "videoTitle": None,
        "videos_data": all_section_videos_data
    }

    return section_result

def get_comments_parallel(video_id, max_results=50):
    """Obtiene comentarios de un video en paralelo."""
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
            comment_snippet = item['snippet']['topLevelComment']['snippet']
            comments.append({
                "author": comment_snippet['authorDisplayName'],
                "text": comment_snippet['textDisplay']
            })
        return comments
    except HttpError as e:
        if e.resp.status == 403 and 'commentsDisabled' in str(e.content):
            print(f"Los comentarios están deshabilitados para el video {video_id}")
            return []
        else:
            print(f"Error HTTP {e.resp.status} al obtener comentarios para {video_id}: {e}")
            return []
    except Exception as e:
        print(f"Error inesperado obteniendo comentarios para {video_id}: {e}")
        return []

@app.route("/solicitar_cursos", methods=["POST"])
@cross_origin()
def solicitar_cursos():
    try:
        data = request.json
        prompt = data.get("prompt", "").strip()

        if not prompt:
            return jsonify({"error": "El prompt no puede estar vacío"}), 400
        
        max_workers = min(32, (os.cpu_count() or 1) * 2)

        if not GOOGLE_API_KEY or not YOUTUBE_API_KEY:
            return jsonify({"error": "Error de configuración: Faltan claves API en el servidor."}), 500

        print(f"\n=== Iniciando generación de curso para prompt: '{prompt}' ===")

        try:
            course_outline = get_course_outline(prompt)
            print("Esquema del curso generado con éxito.")
        except Exception as e:
            print(f"Fallo crítico: No se pudo generar el esquema del curso. Error: {e}")
            return jsonify({"error": f"Error al generar la estructura base del curso: {str(e)}"}), 500

        topic = course_outline.get("extracted_topic", prompt.split(' ')[0])
        print(f"Tema principal extraído para la búsqueda de videos: '{topic}'")

        course_id = f"course_{int(time.time())}"
        used_video_ids = set()

        # Procesar secciones en paralelo
        print("\n--- Procesando secciones en paralelo ---")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_section = {
                executor.submit(
                    process_section_parallel,
                    section,
                    i,
                    topic,
                    used_video_ids,
                    analyzer_sentimiento
                ): (i, section)
                for i, section in enumerate(course_outline.get("sections", []))
            }

            sections = []
            for future in concurrent.futures.as_completed(future_to_section):
                try:
                    section_result = future.result()
                    sections.append(section_result)
                except Exception as exc:
                    print(f"Error procesando sección en paralelo: {exc}")

        # Ordenar secciones por ID
        sections.sort(key=lambda x: x["id"])

        # Procesar videos de cada sección en paralelo
        print("\n--- Analizando videos de secciones en paralelo ---")
        for section in sections:
            if not section.get("videos_data"):
                continue

            approved_section_videos = []
            all_analyzed_videos_with_ai_results = []

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_video_section = {
                    executor.submit(
                        analyze_video_content, 
                        video_data["videoId"], 
                        section.get("content", ""), 
                        used_video_ids,
                        topic, 
                        video_data.get("title"),
                        video_data.get("processed_transcript"),
                        analyzer=analyzer_sentimiento
                    ): video_data
                    for video_data in section["videos_data"]
                }

                for future in concurrent.futures.as_completed(future_to_video_section):
                    video_data = future_to_video_section[future]
                    try:
                        is_approved, reason, analysis_details = future.result()
                        if is_approved:
                            approved_section_videos.append(video_data)
                        if analysis_details:
                            all_analyzed_videos_with_ai_results.append((video_data, analysis_details))
                    except Exception as exc:
                        print(f"Error analizando video de sección {video_data['videoId']} en paralelo: {exc}")

            # Seleccionar el mejor video para la sección
            if approved_section_videos:
                best_video = sorted(approved_section_videos, key=lambda x: x["score"], reverse=True)[0]
                used_video_ids.add(best_video["videoId"])
                section.update({
                    "duration": best_video["duration"],
                    "videoId": best_video["videoId"],
                    "videoTitle": best_video["title"],
                    "videoUrl": f"https://www.youtube.com/embed/{best_video['videoId']}"
                })
            elif all_analyzed_videos_with_ai_results:
                available_for_fallback = [
                    (vid_data, analysis)
                    for vid_data, analysis in all_analyzed_videos_with_ai_results
                    if vid_data["videoId"] not in used_video_ids
                ]

                if available_for_fallback:
                    available_for_fallback.sort(key=lambda x: x[1].get('confidence_score', 0), reverse=True)
                    best_fallback_video_data, best_fallback_analysis = available_for_fallback[0]
                    
                    used_video_ids.add(best_fallback_video_data["videoId"])
                    section.update({
                        "duration": best_fallback_video_data["duration"],
                        "videoId": best_fallback_video_data["videoId"],
                        "videoTitle": best_fallback_video_data["title"],
                        "videoUrl": f"https://www.youtube.com/embed/{best_fallback_video_data['videoId']}"
                    })

            # Eliminar datos temporales
            section.pop("videos_data", None)

        # Añadir sección de evaluación final
        sections.append({
            "id": len(sections) + 1,
            "title": "Evaluación Final",
            "content": "Evalúa lo aprendido en el curso.",
            "videoUrl": None,
            "duration": "N/A",
            "classes": 1,
            "videoId": None,
            "videoTitle": None
        })

        # Calcular duración total y clases
        total_classes = len(sections)
        total_minutes_calculation = sum(
            float(section.get("duration", "0").split()[0]) 
            for section in sections 
            if section.get("duration") != "N/A"
        )

        total_duration_str = "N/A"
        if total_minutes_calculation > 0:
            total_hours = int(total_minutes_calculation // 60)
            total_mins = int(total_minutes_calculation % 60)
            if total_hours > 0:
                total_duration_str = f"{total_hours}h {total_mins}m"
            else:
                total_duration_str = f"{total_mins}m"

        current_date = datetime.now().strftime("%m/%Y")
        course_data = {
            "title": course_outline.get("title", f"Curso sobre {prompt}"),
            "introduction": course_outline.get("introduction", "Introducción no disponible"),
            "instructor": "IA Professor",
            "rating": round(random.uniform(4.5, 4.9), 1),
            "students": random.randint(5000, 15000),
            "lastUpdated": current_date,
            "language": "Español",
            "totalDuration": total_duration_str,
            "totalLessons": total_classes,
            "sections": sections,
            "learningOutcomes": course_outline.get("learningOutcomes", []),
            "requirements": course_outline.get("requirements", []),
            "level": course_outline.get("level", "principiante"),
            "level_description": course_outline.get("level_description", "")
        }

        end_time = time.time()
        print(f"=== Generación de curso completada en {end_time :.2f} segundos ===")
        return jsonify(course_data)

    except Exception as e:
        print(f"Error generating course: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Ocurrió un error inesperado: {str(e)}"}), 500


# Add a health check endpoint (SIN CAMBIOS)
@app.route("/health", methods=["GET"])
@cross_origin()
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

from flask import request, jsonify, redirect, url_for, session
from authlib.integrations.flask_client import OAuth
from datetime import datetime
import secrets
import json
from db import DatabaseConnection  # Importa tu clase singleton


# Cargar variables de entorno
load_dotenv()

# Configuración de OAuth con Google
oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id=os.environ.get("GOOGLE_CLIENT_ID"),
    client_secret=os.environ.get("GOOGLE_CLIENT_SECRET"),
    access_token_url='https://oauth2.googleapis.com/token',
    access_token_params=None,
    authorize_url='https://accounts.google.com/o/oauth2/v2/auth',
    authorize_params=None,
    api_base_url='https://www.googleapis.com/oauth2/v3/',
    userinfo_endpoint='https://openidconnect.googleapis.com/v1/userinfo',
    jwks_uri='https://www.googleapis.com/oauth2/v3/certs',
    client_kwargs={'scope': 'openid email profile'},
)

# Instancia del singleton de base de datos
db = DatabaseConnection()


def generate_token():
    """Genera un token único para el usuario"""
    return secrets.token_urlsafe(32)


def create_or_update_user(user_info):
    """Crea o actualiza un usuario basado en la información de Google"""
    try:
        # Verificar si el usuario ya existe
        existing_user = db.execute_query(
            "SELECT id_usuario, token FROM usuarios WHERE correo = %s",
            (user_info['email'],)
        )

        if existing_user:
            # Usuario existe, actualizar token y datos
            user_id = existing_user[0][0]
            new_token = generate_token()

            db.execute_query(
                """UPDATE usuarios 
                   SET token = %s, nombre_completo = %s, actualizado_en = CURRENT_TIMESTAMP 
                   WHERE id_usuario = %s""",
                (new_token, user_info['name'], user_id),
                fetch=False
            )

            return {'id_usuario': user_id, 'token': new_token, 'is_new': False}
        else:
            # Usuario nuevo, crear registro
            new_token = generate_token()
            username = user_info['email'].split('@')[0]  # Usar parte del email como username

            # Verificar que el username sea único
            counter = 1
            original_username = username
            while True:
                existing_username = db.execute_query(
                    "SELECT id_usuario FROM usuarios WHERE nombre_usuario = %s",
                    (username,)
                )
                if not existing_username:
                    break
                username = f"{original_username}_{counter}"
                counter += 1

            # Insertar nuevo usuario
            result = db.execute_query(
                """INSERT INTO usuarios (nombre_usuario, correo, contrasena_hash, nombre_completo, token) 
                   VALUES (%s, %s, %s, %s, %s) RETURNING id_usuario""",
                (username, user_info['email'], 'google_auth', user_info['name'], new_token),
                fetch=True
            )

            return {'id_usuario': result[0][0], 'token': new_token, 'is_new': True}

    except Exception as e:
        print(f"Error en create_or_update_user: {e}")
        raise


def verify_token(token):
    """Verifica si el token es válido y retorna el usuario"""
    try:
        result = db.execute_query(
            "SELECT id_usuario, nombre_usuario, correo, nombre_completo FROM usuarios WHERE token = %s AND esta_activo = true",
            (token,)
        )
        return result[0] if result else None
    except Exception as e:
        print(f"Error en verify_token: {e}")
        return None


def require_auth():
    """Decorator para requerir autenticación"""

    def decorator(f):
        def wrapper(*args, **kwargs):
            token = request.headers.get('Authorization')
            if not token or not token.startswith('Bearer '):
                return jsonify({'error': 'Token requerido'}), 401

            token = token.replace('Bearer ', '')
            user = verify_token(token)
            if not user:
                return jsonify({'error': 'Token inválido'}), 401

            # Agregar usuario a la request
            request.current_user = {
                'id_usuario': user[0],
                'nombre_usuario': user[1],
                'correo': user[2],
                'nombre_completo': user[3]
            }
            return f(*args, **kwargs)

        wrapper.__name__ = f.__name__
        return wrapper

    return decorator


# ENDPOINTS DE AUTENTICACIÓN

@app.route('/login')
@cross_origin()
def login():
    """Inicia el proceso de autenticación con Google"""
    redirect_uri = url_for('auth_callback', _external=True)
    return google.authorize_redirect(redirect_uri)


@app.route('/login/credentials', methods=['POST'])
@cross_origin()
def login_credentials():
    """Autenticación con email y contraseña"""
    try:
        # Obtener datos del request
        data = request.get_json()
        if not data or 'email' not in data or 'password' not in data:
            return jsonify({'error': 'Email y contraseña son requeridos'}), 400

        email = data['email']
        password = data['password']

        # Buscar usuario por email
        user_result = db.execute_query(
            """SELECT id_usuario, nombre_usuario, correo, contrasena_hash, 
                      nombre_completo, token, creado_en, actualizado_en 
               FROM usuarios WHERE correo = %s""",
            (email,)
        )

        if not user_result:
            return jsonify({'error': 'Credenciales inválidas'}), 401

        user_data = user_result[0]
        stored_password_hash = user_data[3]

        # Verificar contraseña (comparación directa)
        if stored_password_hash != password:
            return jsonify({'error': 'Credenciales inválidas'}), 401

        # Generar nuevo token
        new_token = generate_token()

        # Actualizar token en la base de datos
        db.execute_query(
            "UPDATE usuarios SET token = %s, actualizado_en = CURRENT_TIMESTAMP WHERE id_usuario = %s",
            (new_token, user_data[0]),
            fetch=False
        )

        # Retornar datos del usuario
        return jsonify({
            'success': True,
            'user': {
                'id_usuario': user_data[0],
                'nombre_usuario': user_data[1],
                'correo': user_data[2],
                'nombre_completo': user_data[4],
                'token': new_token,
                'creado_en': user_data[6].isoformat() if user_data[6] else None,
                'actualizado_en': user_data[7].isoformat() if user_data[7] else None
            }
        }), 200

    except Exception as e:
        print(f"Error en login_credentials: {e}")
        return jsonify({'error': 'Error interno del servidor'}), 500

@app.route('/auth/callback')
@cross_origin()
def auth_callback():
    """Callback de Google OAuth"""
    try:
        token = google.authorize_access_token()
        user_info = token.get('userinfo')

        if user_info:
            # Crear o actualizar usuario
            user_data = create_or_update_user(user_info)

            # Guardar en sesión
            session['user_token'] = user_data['token']
            session['user_id'] = user_data['id_usuario']

            return redirect('/dashboard')
        else:
            return jsonify({'error': 'No se pudo obtener información del usuario'}), 400

    except Exception as e:
        print(f"Error en auth_callback: {e}")
        return jsonify({'error': 'Error en el proceso de autenticación'}), 500


@app.route('/dashboard')
@cross_origin()
def dashboard():
    """Dashboard principal (requiere autenticación)"""
    if 'user_token' not in session:
        return redirect('/login')

    try:
        user = verify_token(session['user_token'])
        if not user:
            session.clear()
            return redirect('/login')

        return jsonify({
            'message': 'Bienvenido al dashboard',
            'user': {
                'id': user[0],
                'username': user[1],
                'email': user[2],
                'name': user[3]
            }
        })
    except Exception as e:
        print(f"Error en dashboard: {e}")
        return jsonify({'error': 'Error interno del servidor'}), 500


@app.route('/logout')
@cross_origin()
def logout():
    """Cerrar sesión"""
    session.clear()
    return jsonify({'message': 'Sesión cerrada exitosamente'})


# ENDPOINTS DE CURSOS

@app.route('/api/cursos', methods=['GET'])
@require_auth()
@cross_origin()
def get_cursos():
    """Obtiene la lista de cursos del usuario autenticado"""
    try:
        user_id = request.current_user['id_usuario']

        cursos = db.execute_query(
            """SELECT c.id_curso, c.titulo, c.descripcion, c.nivel_profundidad, 
                      c.duracion_total, c.creado_en, c.estado, c.es_publico, c.idioma,
                      COUNT(sc.id_seccion) as total_secciones
               FROM cursos c
               LEFT JOIN secciones_curso sc ON c.id_curso = sc.id_curso
               WHERE c.id_usuario = %s
               GROUP BY c.id_curso, c.titulo, c.descripcion, c.nivel_profundidad, 
                        c.duracion_total, c.creado_en, c.estado, c.es_publico, c.idioma
               ORDER BY c.creado_en DESC""",
            (user_id,)
        )

        cursos_list = []
        for curso in cursos:
            cursos_list.append({
                'id_curso': curso[0],
                'titulo': curso[1],
                'descripcion': curso[2],
                'nivel_profundidad': curso[3],
                'duracion_total': str(curso[4]) if curso[4] else None,
                'creado_en': curso[5].isoformat() if curso[5] else None,
                'estado': curso[6],
                'es_publico': curso[7],
                'idioma': curso[8],
                'total_secciones': curso[9]
            })

        return jsonify({
            'cursos': cursos_list,
            'total': len(cursos_list)
        })

    except Exception as e:
        print(f"Error en get_cursos: {e}")
        return jsonify({'error': 'Error al obtener cursos'}), 500


@app.route('/api/cursos/<int:curso_id>', methods=['GET'])
@require_auth()
@cross_origin()
def get_curso_detalle(curso_id):
    """Obtiene el detalle completo de un curso específico"""
    try:
        user_id = request.current_user['id_usuario']

        # Verificar que el curso pertenece al usuario
        curso = db.execute_query(
            """SELECT c.id_curso, c.titulo, c.descripcion, c.nivel_profundidad, 
                      c.duracion_total, c.creado_en, c.actualizado_en, c.estado, 
                      c.es_publico, c.idioma
               FROM cursos c
               WHERE c.id_curso = %s AND c.id_usuario = %s""",
            (curso_id, user_id)
        )

        if not curso:
            return jsonify({'error': 'Curso no encontrado'}), 404

        curso_data = curso[0]

        # Obtener secciones del curso
        secciones = db.execute_query(
            """SELECT sc.id_seccion, sc.titulo, sc.descripcion, sc.indice_orden,
                      sc.creado_en, sc.id_seccion_padre
               FROM secciones_curso sc
               WHERE sc.id_curso = %s
               ORDER BY sc.indice_orden""",
            (curso_id,)
        )

        # Obtener videos por sección
        secciones_list = []
        for seccion in secciones:
            seccion_id = seccion[0]

            videos = db.execute_query(
                """SELECT v.id_video, v.id_video_youtube, v.titulo, v.nombre_canal,
                          v.url, v.duracion, v.publicado_en, v.conteo_vistas,
                          v.licencia, vsc.indice_orden
                   FROM videos v
                   JOIN videos_secciones_curso vsc ON v.id_video = vsc.id_video
                   WHERE vsc.id_seccion = %s
                   ORDER BY vsc.indice_orden""",
                (seccion_id,)
            )

            videos_list = []
            for video in videos:
                videos_list.append({
                    'id_video': video[0],
                    'id_video_youtube': video[1],
                    'titulo': video[2],
                    'nombre_canal': video[3],
                    'url': video[4],
                    'duracion': str(video[5]) if video[5] else None,
                    'publicado_en': video[6].isoformat() if video[6] else None,
                    'conteo_vistas': video[7],
                    'licencia': video[8],
                    'orden': video[9]
                })

            secciones_list.append({
                'id_seccion': seccion[0],
                'titulo': seccion[1],
                'descripcion': seccion[2],
                'indice_orden': seccion[3],
                'creado_en': seccion[4].isoformat() if seccion[4] else None,
                'id_seccion_padre': seccion[5],
                'videos': videos_list
            })

        # Obtener etiquetas del curso
        etiquetas = db.execute_query(
            """SELECT e.id_etiqueta, e.nombre
               FROM etiquetas e
               JOIN etiquetas_curso ec ON e.id_etiqueta = ec.id_etiqueta
               WHERE ec.id_curso = %s""",
            (curso_id,)
        )

        etiquetas_list = [{'id_etiqueta': et[0], 'nombre': et[1]} for et in etiquetas]

        curso_completo = {
            'id_curso': curso_data[0],
            'titulo': curso_data[1],
            'descripcion': curso_data[2],
            'nivel_profundidad': curso_data[3],
            'duracion_total': str(curso_data[4]) if curso_data[4] else None,
            'creado_en': curso_data[5].isoformat() if curso_data[5] else None,
            'actualizado_en': curso_data[6].isoformat() if curso_data[6] else None,
            'estado': curso_data[7],
            'es_publico': curso_data[8],
            'idioma': curso_data[9],
            'secciones': secciones_list,
            'etiquetas': etiquetas_list
        }

        return jsonify(curso_completo)

    except Exception as e:
        print(f"Error en get_curso_detalle: {e}")
        return jsonify({'error': 'Error al obtener detalle del curso'}), 500


@app.route('/api/cursos', methods=['POST'])
@require_auth()
@cross_origin()
def crear_curso():
    """Crea un nuevo curso con todas sus dependencias"""
    try:
        user_id = request.current_user['id_usuario']
        data = request.get_json()

        if not data:
            return jsonify({'error': 'Datos requeridos'}), 400

        # Extraer datos del fullData
        full_data = data.get('fullData', {})
        if not full_data:
            return jsonify({'error': 'fullData requerido'}), 400

        # Determinar nivel numérico basado en el nivel de texto
        nivel_map = {
            'principiante': 1,
            'intermedio': 2,
            'avanzado': 3
        }
        nivel_profundidad = nivel_map.get(full_data.get('level', 'principiante'), 1)

        # Calcular duración total si hay secciones
        duracion_total = full_data.get('totalDuration', '0m')

        # Crear el curso principal
        curso_result = db.execute_query(
            """INSERT INTO cursos (
                id_usuario, titulo, descripcion, nivel_profundidad, 
                duracion_total, estado, es_publico, idioma
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id_curso""",
            (
                user_id,
                full_data.get('title', ''),
                full_data.get('introduction', ''),
                nivel_profundidad,
                f"{duracion_total} minutes" if duracion_total != 'N/A' else None,
                'borrador',  # estado
                False,  # es_publico
                full_data.get('language', 'es')
            ),
            fetch=True
        )

        curso_id = curso_result[0][0]

        # Procesar secciones si existen
        if 'sections' in full_data and full_data['sections']:
            for seccion_data in full_data['sections']:
                # Crear sección
                seccion_result = db.execute_query(
                    """INSERT INTO secciones_curso (id_curso, titulo, descripcion, indice_orden)
                       VALUES (%s, %s, %s, %s) RETURNING id_seccion""",
                    (
                        curso_id,
                        seccion_data.get('title', ''),
                        seccion_data.get('content', ''),
                        seccion_data.get('id', seccion_data.get('indice_orden', 1))
                    ),
                    fetch=True
                )

                seccion_id = seccion_result[0][0]

                # Si hay video asociado, crearlo
                if seccion_data.get('videoId') and seccion_data.get('videoTitle'):
                    # Primero verificar si el video ya existe
                    video_existente = db.execute_query(
                        "SELECT id_video FROM videos WHERE id_video_youtube = %s",
                        (seccion_data['videoId'],)
                    )

                    if video_existente:
                        video_id = video_existente[0][0]
                    else:
                        # Crear nuevo video
                        duracion_video = seccion_data.get('duration', 'N/A')
                        if duracion_video == 'N/A':
                            duracion_interval = None
                        else:
                            # Convertir formato "X min" a interval
                            try:
                                minutes = int(duracion_video.replace(' min', '').replace('m', ''))
                                duracion_interval = f"{minutes} minutes"
                            except:
                                duracion_interval = None

                        video_result = db.execute_query(
                            """INSERT INTO videos (id_video_youtube, titulo, nombre_canal, url, 
                                                   duracion, licencia, conteo_vistas)
                               VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id_video""",
                            (
                                seccion_data['videoId'],
                                seccion_data['videoTitle'],
                                'Canal YouTube',  # Valor por defecto
                                seccion_data.get('videoUrl',
                                                 f"https://www.youtube.com/watch?v={seccion_data['videoId']}"),
                                duracion_interval,
                                'Standard YouTube License',  # Valor por defecto
                                0  # Conteo inicial
                            ),
                            fetch=True
                        )
                        video_id = video_result[0][0]

                    # Asociar video con sección
                    db.execute_query(
                        """INSERT INTO videos_secciones_curso (id_seccion, id_video, indice_orden)
                           VALUES (%s, %s, %s)""",
                        (seccion_id, video_id, 1),
                        fetch=False
                    )

        # Crear etiquetas por defecto basadas en el título y nivel
        etiquetas_default = [
            full_data.get('level', 'principiante'),
            full_data.get('language', 'español')
        ]

        for etiqueta_nombre in etiquetas_default:
            # Verificar si la etiqueta existe
            etiqueta_existente = db.execute_query(
                "SELECT id_etiqueta FROM etiquetas WHERE nombre = %s",
                (etiqueta_nombre,)
            )

            if etiqueta_existente:
                etiqueta_id = etiqueta_existente[0][0]
            else:
                # Crear nueva etiqueta
                etiqueta_result = db.execute_query(
                    "INSERT INTO etiquetas (nombre) VALUES (%s) RETURNING id_etiqueta",
                    (etiqueta_nombre,),
                    fetch=True
                )
                etiqueta_id = etiqueta_result[0][0]

            # Asociar etiqueta con curso
            try:
                db.execute_query(
                    "INSERT INTO etiquetas_curso (id_curso, id_etiqueta) VALUES (%s, %s)",
                    (curso_id, etiqueta_id),
                    fetch=False
                )
            except:
                # Si ya existe la asociación, ignorar el error
                pass

        return jsonify({
            'message': 'Curso creado exitosamente',
            'id_curso': curso_id,
            'titulo': full_data.get('title', '')
        }), 201

    except Exception as e:
        print(f"Error en crear_curso: {e}")
        return jsonify({'error': 'Error al crear el curso'}), 500


# Endpoint adicional para obtener información del usuario autenticado
@app.route('/api/user/me', methods=['GET'])
@require_auth()
@cross_origin()
def get_current_user():
    """Obtiene información del usuario autenticado"""
    return jsonify({
        'user': request.current_user
    })

# --- Ejecución Principal SIN CAMBIOS ---
if __name__ == "__main__":
    # Mantener forma original de correr la app
    port = int(os.environ.get("PORT", 5000))  # Puerto 5000 como en muchos ejemplos Flask
    app.run(host="0.0.0.0", port=port, debug=True)  # debug=True como en original