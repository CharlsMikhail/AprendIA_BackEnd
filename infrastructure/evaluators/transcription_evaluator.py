import re
from youtube_transcript_api import YouTubeTranscriptApi
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from textblob import TextBlob
from spellchecker import SpellChecker
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

CORRECCIONES_COMUNES = {
    "ke": "que", "q": "que", "xq": "porque", "pq": "porque", "tb": "también",
    "xfa": "por favor", "thx": "gracias", "np": "no hay problema"
}

class TranscriptionEvaluator:
    """
    Evaluador de transcripciones y encargado de validación semántica (RAG base).
    """

    def __init__(self):
        pass

    def get_video_transcript(self, video_id, max_minutes=None):
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['es'])

            if max_minutes is None:
                texto_completo = " ".join([entry['text'] for entry in transcript])
                return self.procesar_transcripcion(texto_completo)

            max_seconds = max_minutes * 60
            text_parts = []
            for entry in transcript:
                if entry['start'] >= max_seconds:
                    break
                text_parts.append(entry['text'])

            texto_parcial = " ".join(text_parts)
            return self.procesar_transcripcion(texto_parcial)

        except Exception as e:
            # TODO: Integrar aquí la alternativa de Fast-Whisper si falla la API oficial (Comentado según directriz)
            """
            from infrastructure.inference.transcriber import FastWhisperTranscriber
            transcriber = FastWhisperTranscriber()
            return transcriber.download_and_transcribe(video_id)
            """
            print(f"Error obteniendo transcripción oficial para {video_id}: {e}")
            return None

    def procesar_transcripcion(self, texto):
        """Limpia una transcripción de video (Migrado)."""
        try:
            texto = re.sub(r'\[\d{2}:\d{2}:\d{2}\]|\[\d{2}:\d{2}\]', '', texto)
            texto = re.sub(r'\[.*?\]', '', texto)
            texto = re.sub(r'[^\w\s]', ' ', texto)
            texto = texto.lower()

            palabras = texto.split()
            palabras_corregidas = [CORRECCIONES_COMUNES.get(palabra, palabra) for palabra in palabras]
            texto = ' '.join(palabras_corregidas)

            # TODO: Agregar lógica de RAG (Retrieval-Augmented Generation) para validación semántica avanzada.
            # Por ahora se mantiene TF-IDF para simplificación de resumen.
            
            return texto
        except Exception as e:
            print(f"Error procesando transcripción: {e}")
            return texto
