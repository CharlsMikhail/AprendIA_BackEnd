import os
import requests
import logging
import threading

class ColabClient:
    """
    Cliente para disparar tareas asíncronas en los notebooks de Google Colab.
    (Versión PRODUCCIÓN - Envía peticiones reales a las URLs de Ngrok)
    """
    def __init__(self):
        self.secret_token = os.getenv("COLAB_SECRET_TOKEN", "AprendiaSecret2026")
        
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.secret_token}"
        }
        
        # Leemos las URLs desde el .env. 
        # Si tienes nombres específicos de endpoints (ej: /generar, /analizar), 
        # puedes concatenarlos aquí.
        self.outline_url = os.getenv("OUTLINE_GENERATOR_URL", "http://localhost:8001/generar_esquema")
        self.sentiment_url = os.getenv("SENTIMENT_ANALYZER_URL", "http://localhost:8002/analizar_ranking")

    def trigger_outline_generation(self, job_id: str, prompt: str) -> bool:
        """
        Envía el prompt al Colab #1 (Generación de Esquema).
        """
        logging.info(f"COLAB REAL #1: Iniciando generación para job {job_id} en {self.outline_url}")

        payload = {
            "job_id": job_id,
            "prompt": prompt
        }

        def post_async():
            try:
                # Timeout de 10s solo para la conexión inicial, no esperamos la respuesta completa
                # ya que Colab puede tardar y responderá mediante nuestro Webhook
                response = requests.post(self.outline_url, json=payload, headers=self.headers, timeout=10)
                logging.info(f"COLAB REAL #1 respondió con HTTP {response.status_code}")
            except Exception as e:
                logging.error(f"Error conectando al Colab #1 ({self.outline_url}): {e}")

        # Ejecutamos la petición HTTP en background para no bloquear el API principal
        threading.Thread(target=post_async).start()
        return True

    def trigger_ranking_analysis(self, job_id: str, candidates: list) -> bool:
        """
        Envía videos al Colab #2 (Análisis de sentimiento, RAG y Ranking final).
        """
        logging.info(f"COLAB REAL #2: Analizando candidatos para job {job_id} en {self.sentiment_url}")
        
        payload = {
            "job_id": job_id,
            "candidates": candidates
        }

        def post_async():
            try:
                # Hacemos POST a la URL de Ngrok
                response = requests.post(self.sentiment_url, json=payload, headers=self.headers, timeout=10)
                logging.info(f"COLAB REAL #2 respondió con HTTP {response.status_code}")
            except Exception as e:
                logging.error(f"Error conectando al Colab #2 ({self.sentiment_url}): {e}")

        threading.Thread(target=post_async).start()
        return True
