import logging
import random
import string

class TranscriptionClient:
    """
    Cliente MOCK para la transcripción de videos (Fase 2).
    En Fase 3 será reemplazado por el orquestador de Whisper con workers en Colab.
    Por ahora simula recibir una URL y devolver texto placeholder.
    """
    def __init__(self):
        pass

    def transcribe(self, video_url: str) -> dict:
        """
        MOCK: Simula la transcripción de un video.
        Returns: dict con video_url, transcript (texto), success
        """
        logging.info(f"MOCK TRANSCRIPTION: Simulando transcripción para {video_url}")

        mock_transcript = (
            "Bienvenidos a esta clase donde vamos a aprender los conceptos fundamentales. "
            "Primero revisaremos la teoría básica y luego haremos ejercicios prácticos. "
            "Es importante que tengan su entorno de desarrollo configurado antes de comenzar. "
            "En este módulo cubriremos variables, tipos de datos, estructuras de control y funciones. "
            "Al final del video haremos un pequeño proyecto para integrar todo lo aprendido. "
            "Recuerden que la práctica constante es la clave para dominar cualquier lenguaje de programación. "
            "Si tienen dudas, pueden dejar sus preguntas en los comentarios."
        )

        return {
            "video_url": video_url,
            "transcript": mock_transcript,
            "success": True,
            "duration_seconds": random.randint(300, 1800)
        }
