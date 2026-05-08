import os
import requests
import logging
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class SentimentAnalyzerClient:
    """
    Cliente para el analizador de sentimiento desplegado en Colab vía ngrok.
    POST https://<ngrok-url>/analyze
    Body: {"url": "<youtube_url>", "max_comments": 100, "muestra": 5}
    Filtro: porcentaje_utiles >= 60%
    """
    def __init__(self):
        self.base_url = os.getenv("SENTIMENT_ANALYZER_URL", "").rstrip('/')
        self.min_useful_percentage = 60  # Umbral mínimo

    def analyze(self, youtube_url: str, max_comments: int = 100) -> dict:
        """
        Envía una URL de YouTube al analizador de sentimiento.
        Returns: dict con video_id, porcentaje_utiles, distribucion, etc.
        """
        if not self.base_url:
            logging.warning("SENTIMENT_ANALYZER_URL no configurada. Retornando mock.")
            return {
                "video_id": "mock",
                "porcentaje_utiles": 85,
                "distribucion": {"positive": 70, "negative": 15, "neutral": 15},
                "passed": True
            }

        payload = {
            "url": youtube_url,
            "max_comments": max_comments,
            "muestra": 5
        }

        try:
            response = requests.post(
                f"{self.base_url}/analyze",
                json=payload,
                timeout=60,  # Puede tardar en analizar comentarios
                verify=False # Evita el error SSL de ngrok
            )
            response.raise_for_status()
            data = response.json()

            # Solo para pruebas: mostrar credibilidad en consola
            porcentaje = data.get("porcentaje_utiles", 0)
            logging.info(f"Video evaluado [{youtube_url}]: {porcentaje}% de utilidad/credibilidad")

            # Añadir campo de filtro
            data["passed"] = porcentaje >= self.min_useful_percentage
            return data

        except requests.exceptions.HTTPError as e:
            try:
                data = e.response.json()
                
                # Si no tiene comentarios, se le da un 40% base para que no quede en 0
                # y pueda ser elegido en el fallback por encima de videos con mal sentimiento
                if data.get("total_comentarios") == 0 or e.response.status_code == 404:
                    data["porcentaje_utiles"] = 40.0
                    
                data["passed"] = data.get("porcentaje_utiles", 0) >= self.min_useful_percentage
                if "mensaje" in data:
                    logging.info(f"Análisis omitido para {youtube_url}: {data['mensaje']} (Asignado 40% por defecto)")
                return data
            except ValueError:
                error_msg = f"{e} - Body: {e.response.text}"
                logging.error(f"Error HTTP en análisis de sentimiento para {youtube_url}: {error_msg}")
                return {
                    "video_id": "error",
                    "porcentaje_utiles": 0,
                    "distribucion": {},
                    "passed": False,
                    "error": error_msg
                }
        except Exception as e:
            logging.error(f"Error en análisis de sentimiento para {youtube_url}: {e}")
            return {
                "video_id": "error",
                "porcentaje_utiles": 0,
                "distribucion": {},
                "passed": False,
                "error": str(e)
            }
