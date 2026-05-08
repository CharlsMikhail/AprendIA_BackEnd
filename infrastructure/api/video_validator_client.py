import os
import requests
import logging

class VideoValidatorClient:
    """
    Cliente para el validador de video con RAG desplegado en cramsoft.dev.
    Se usa DESPUÉS de la transcripción para verificar que el contenido 
    del video es relevante al tema del curso.
    
    POST http://aprendia.cramsoft.dev/validar-video
    Body: {"transcripcion": "<transcripción del video>"}
    """
    def __init__(self):
        self.base_url = os.getenv("VIDEO_VALIDATOR_URL", "http://aprendia.cramsoft.dev/validar-video")

    def validate(self, transcript_text: str) -> dict:
        """
        Envía la transcripción del video al validador RAG.
        Returns: dict con la respuesta del validador
        """
        if not transcript_text or not transcript_text.strip():
            logging.warning("Transcripción vacía, no se puede validar.")
            return {
                "score_final": 0.0, 
                "total_conceptos": 0, 
                "evaluaciones": [], 
                "error": "Transcripción vacía"
            }

        payload = {
            "transcripcion": transcript_text
        }

        try:
            response = requests.post(
                self.base_url,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            return data

        except requests.exceptions.HTTPError as e:
            try:
                # Si la API en cramsoft devuelve un JSON de error estructurado, intentamos leerlo
                error_data = e.response.json()
                error_msg = f"{e} - API Error: {error_data}"
            except ValueError:
                # Si devuelve HTML (ej. 502 Bad Gateway de Cloudflare)
                error_msg = f"{e} - Body HTML/Raw: {e.response.text}"
                
            logging.error(f"Error HTTP validando video con RAG: {error_msg}")
            return {
                "score_final": 0.0,
                "total_conceptos": 0,
                "evaluaciones": [],
                "error": error_msg
            }
        except Exception as e:
            logging.error(f"Error inesperado validando video con RAG: {e}")
            return {
                "score_final": 0.0,
                "total_conceptos": 0,
                "evaluaciones": [],
                "error": str(e)
            }
