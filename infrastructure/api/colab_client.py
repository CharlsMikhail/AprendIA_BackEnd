import os
import requests
import logging

class ColabClient:
    """
    Cliente para disparar tareas asíncronas en los notebooks de Google Colab.
    """
    def __init__(self):
        # URLs de los webhooks expuestos por ngrok/localtunnel desde los Colabs
        self.colab_outline_url = os.getenv("COLAB_OUTLINE_URL", "")
        self.colab_ranking_url = os.getenv("COLAB_RANKING_URL", "")
        self.secret_token = os.getenv("COLAB_SECRET_TOKEN", "AprendiaSecret2026")
        
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.secret_token}"
        }

    def trigger_outline_generation(self, job_id: str, prompt: str) -> bool:
        """
        Dispara el Colab #1 para que valide el prompt y genere el outline del curso.
        """
        if not self.colab_outline_url:
            logging.warning("COLAB_OUTLINE_URL no configurada. Simulando disparo a Colab #1.")
            return False

        payload = {
            "job_id": job_id,
            "prompt": prompt
        }
        
        try:
            # Hacemos la petición (esperamos que sea asíncrona o que devuelva 202 Accepted rápido)
            response = requests.post(
                f"{self.colab_outline_url}/generate_outline", 
                json=payload, 
                headers=self.headers,
                timeout=5
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logging.error(f"Error al disparar Colab #1: {e}")
            return False

    def trigger_ranking_analysis(self, job_id: str, candidates: list) -> bool:
        """
        Dispara el Colab #2 para analizar sentimiento y transcripciones.
        """
        if not self.colab_ranking_url:
            logging.warning("COLAB_RANKING_URL no configurada. Simulando disparo a Colab #2.")
            return False

        payload = {
            "job_id": job_id,
            "candidates": candidates
        }
        
        try:
            response = requests.post(
                f"{self.colab_ranking_url}/analyze_ranking", 
                json=payload, 
                headers=self.headers,
                timeout=5
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logging.error(f"Error al disparar Colab #2: {e}")
            return False
