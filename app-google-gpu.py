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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True) 