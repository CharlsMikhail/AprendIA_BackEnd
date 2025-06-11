from flask import Flask, request, jsonify
from flask_cors import CORS
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
import torch
import torch.nn as nn
import torch.cuda
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import cupy as cp  # Para operaciones en GPU
from numba import cuda, jit

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuración de GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# Configuración de CUDA
if torch.cuda.is_available():
    torch.cuda.set_device(0)  # Usar primera GPU
    print(f"GPU disponible: {torch.cuda.get_device_name(0)}")
    print(f"Memoria GPU total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Variables de entorno
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash-preview-05-20")
GOOGLE_API_URL_TEMPLATE = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent"

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

# Descargar recursos NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')

# Cargar modelo spaCy
try:
    nlp = spacy.load("es_core_news_sm")
except:
    print("Modelo de spaCy no encontrado. Por favor, ejecuta: python -m spacy download es_core_news_sm")

# Inicializar analizador de sentimiento
analyzer_sentimiento = create_analyzer(task="sentiment", lang="es")

# Clase para procesamiento de texto en GPU
class TextProcessorGPU:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased')
        self.model = AutoModel.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased').to(device)

    @cuda.jit
    def process_text_gpu(self, text):
        # Implementación de procesamiento de texto en GPU
        pass

    def get_embeddings(self, texts):
        # Obtener embeddings usando BERT en GPU
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).cpu().numpy())
        return np.vstack(embeddings)

# Clase para análisis de similitud en GPU
class SimilarityAnalyzerGPU:
    def __init__(self):
        self.text_processor = TextProcessorGPU()

    def calculate_similarity(self, text1, text2):
        # Calcular similitud usando GPU
        emb1 = self.text_processor.get_embeddings([text1])
        emb2 = self.text_processor.get_embeddings([text2])

        # Convertir a CuPy para operaciones en GPU
        emb1_gpu = cp.array(emb1)
        emb2_gpu = cp.array(emb2)

        # Calcular similitud coseno en GPU
        similarity = cp.dot(emb1_gpu, emb2_gpu.T) / (cp.linalg.norm(emb1_gpu) * cp.linalg.norm(emb2_gpu))
        return float(similarity[0][0])

# Clase para procesamiento de videos en GPU
class VideoProcessorGPU:
    def __init__(self):
        self.similarity_analyzer = SimilarityAnalyzerGPU()

    def process_video_batch(self, videos_data, section_content):
        # Procesar batch de videos en GPU
        results = []
        for video_data in videos_data:
            similarity = self.similarity_analyzer.calculate_similarity(
                video_data.get("title", "") + " " + video_data.get("description", ""),
                section_content
            )
            video_data["similarity_score"] = similarity
            results.append(video_data)
        return results

# Inicializar procesadores GPU
text_processor = TextProcessorGPU()
video_processor = VideoProcessorGPU()

def process_section_gpu(section_data, section_index, topic, used_video_ids, analyzer):
    """Procesa una sección del curso usando GPU."""
    section_id = section_index + 1
    print(f"Procesando Sección {section_id}: {section_data.get('title', 'Sin Título')}")

    section_title_full = section_data.get('title', '')

    # Simplificar título
    simplified_title_parts = section_title_full.split(':')[0].strip()
    if not simplified_title_parts or len(simplified_title_parts.split()) < 2 or simplified_title_parts == section_title_full:
        simplified_title_parts = ' '.join(section_title_full.split()[:4]).strip()

    # Construir query
    if topic.lower() in simplified_title_parts.lower():
        section_query = simplified_title_parts
    else:
        section_query = f"{simplified_title_parts} {topic}"
    section_query = section_query.strip()

    print(f"Buscando videos para Sección {section_id} con query: '{section_query}'")
    all_section_videos_data = search_youtube_videos(
        section_query,
        section_content=section_data.get("description", ""),
        used_video_ids=used_video_ids
    )

    if not all_section_videos_data:
        # Intentar queries alternativas
        alt_query_1 = f"{section_title_full} {topic}".strip()
        print(f"Query simplificada falló. Intentando alternativa 1: '{alt_query_1}'")
        all_section_videos_data = search_youtube_videos(
            alt_query_1,
            section_content=section_data.get("description", ""),
            used_video_ids=used_video_ids
        )

        if not all_section_videos_data:
            alt_query_2 = f"{topic} {section_data.get('level', 'principiante')}".strip()
            print(f"Alternativa 1 falló. Intentando alternativa 2: '{alt_query_2}'")
            all_section_videos_data = search_youtube_videos(
                alt_query_2,
                section_content=section_data.get("description", ""),
                used_video_ids=used_video_ids
            )

    # Procesar videos usando GPU
    if all_section_videos_data:
        all_section_videos_data = video_processor.process_video_batch(
            all_section_videos_data,
            section_data.get("description", "")
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
            top_5_videos = sorted_videos[:5]  # Cambiado de top_10_videos a top_5_videos y slice a :5

            # Procesar las transcripciones de los 5 mejores videos en paralelo
            videos_with_transcripts = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_video = {executor.submit(get_video_transcript, video["videoId"]): video for video in
                                   top_5_videos}  # Usar top_5_videos
                for future in concurrent.futures.as_completed(future_to_video):
                    video = future_to_video[future]
                    try:
                        transcript = future.result()
                        if transcript:
                            video["processed_transcript"] = transcript
                            videos_with_transcripts.append(video)
                        else:
                            print(
                                f"Video {video['videoId']} descartado: No se pudo obtener transcripción en el procesamiento paralelo.")
                    except Exception as exc:
                        print(f"Error obteniendo transcripción para {video['videoId']} en paralelo: {exc}")

            return videos_with_transcripts  # Devolver solo los videos con transcripción exitosa

        return []

    except Exception as e:
        print(f"Error searching YouTube videos: {e}")
        return []
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

@app.route("/solicitar_cursos", methods=["POST"])
def solicitar_cursos():
    try:
        data = request.json
        prompt = data.get("prompt", "").strip()

        if not prompt:
            return jsonify({"error": "El prompt no puede estar vacío"}), 400

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

        # Procesar secciones usando GPU
        print("\n--- Procesando secciones con GPU ---")
        sections = []
        for i, section in enumerate(course_outline.get("sections", [])):
            try:
                section_result = process_section_gpu(
                    section,
                    i,
                    topic,
                    used_video_ids,
                    analyzer_sentimiento
                )
                sections.append(section_result)
            except Exception as exc:
                print(f"Error procesando sección en GPU: {exc}")

        # Ordenar secciones por ID
        sections.sort(key=lambda x: x["id"])

        # Procesar videos de cada sección
        print("\n--- Analizando videos de secciones ---")
        for section in sections:
            if not section.get("videos_data"):
                continue

            # Ordenar videos por similitud
            section["videos_data"].sort(key=lambda x: x.get("similarity_score", 0), reverse=True)

            if section["videos_data"]:
                best_video = section["videos_data"][0]
                used_video_ids.add(best_video["videoId"])
                section.update({
                    "duration": best_video["duration"],
                    "videoId": best_video["videoId"],
                    "videoTitle": best_video["title"],
                    "videoUrl": f"https://www.youtube.com/embed/{best_video['videoId']}"
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

@app.route("/health", methods=["GET"])
def health_check():
    youtube_ok = bool(YOUTUBE_API_KEY)
    google_ok = bool(GOOGLE_API_KEY)
    gpu_ok = torch.cuda.is_available()

    status = {
        "status": "ok" if youtube_ok and google_ok and gpu_ok else "error",
        "dependencies": {
            "youtube_api": "configured" if youtube_ok else "missing_key",
            "google_generative_api": "configured" if google_ok else "missing_key",
            "gpu": "available" if gpu_ok else "not_available"
        }
    }
    status_code = 200 if status["status"] == "ok" else 503
    return jsonify(status), status_code


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
    max_workers_outline = min(32, (os.cpu_count() or 1) * 2)  # Para paralelizar llamadas a IA aquí

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
                raise  # Por ahora, relanzamos si falla el contenido

        # Combinar la introducción con el contenido del curso
        course_outline = {
            **content_data,  # Asegurarse que content_data no sea None
            "introduction": intro_data["introduction"],  # Asegurarse que intro_data no sea None
            "extracted_topic": topic
        }

        # Generar queries de búsqueda mejoradas
        course_outline["searchQueries"] = {
            "introductory": f"introducción a {topic} para {level}s",
            **{f"section{i}": f"{section['title']} {topic} tutorial {level}" for i, section in
               enumerate(course_outline["sections"])}
        }

        return course_outline

    except Exception as e:
        print(f"Error generando esquema del curso: {e}")
        raise
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


# --- Resto de las funciones SIN CAMBIOS ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)