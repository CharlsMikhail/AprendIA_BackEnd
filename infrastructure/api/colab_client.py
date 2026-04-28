import os
import requests
import logging
import threading
import time

class ColabClient:
    """
    Cliente para disparar tareas asíncronas en los notebooks de Google Colab.
    (Versión Fase 1: MOCK LOCAL).
    En lugar de hacer peticiones a Colab, levanta un hilo interno que tras unos segundos
    hace un POST a localhost simulando que Colab está respondiendo.
    """
    def __init__(self):
        self.secret_token = os.getenv("COLAB_SECRET_TOKEN", "AprendiaSecret2026")
        
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.secret_token}"
        }

    def trigger_outline_generation(self, job_id: str, prompt: str) -> bool:
        """
        Simula enviar el prompt al Colab #1.
        """
        logging.info(f"MOCK COLAB #1: Iniciando generación para job {job_id} con prompt: {prompt}")

        def mock_colab_1_work():
            time.sleep(3) # Simulamos 3 segundos de procesamiento LLM
            mock_outline = {
                "title": f"Curso sobre {prompt}",
                "introduction": "Introducción simulada por Mock Colab 1",
                "sections": [
                    {"title": "Módulo 1", "description": "Intro básica", "level": "principiante"},
                    {"title": "Módulo 2", "description": "Práctica", "level": "principiante"}
                ],
                "learningOutcomes": ["Aprenderás a mockear"],
                "requirements": ["Ganas de aprender"],
                "level": "principiante",
                "level_description": "Conceptos fundamentales"
            }
            
            payload = {
                "job_id": job_id,
                "course_outline": mock_outline
            }
            try:
                requests.post("http://localhost:5000/colab/entregar_esquema", json=payload, headers=self.headers)
            except Exception as e:
                logging.error(f"Error en mock_colab_1 webhook: {e}")

        # Arrancamos el hilo simulando la asincronía de la respuesta
        threading.Thread(target=mock_colab_1_work).start()
        return True

    def trigger_ranking_analysis(self, job_id: str, candidates: list) -> bool:
        """
        Simula enviar videos al Colab #2 para análisis de sentimiento.
        """
        logging.info(f"MOCK COLAB #2: Analizando {len(candidates)} candidatos para job {job_id}")

        def mock_colab_2_work():
            time.sleep(4) # Simulamos 4 segundos de descargas y procesamiento
            
            ranked_sections = []
            for candidate_group in candidates:
                best_video = None
                if candidate_group.get("candidates"):
                    best_video = candidate_group["candidates"][0] # Simple mock: agarrar el primero
                    best_video["score"] = 98.5
                    
                ranked_sections.append({
                    "is_intro": candidate_group.get("is_intro"),
                    "section_id": candidate_group.get("section_id"),
                    "title": candidate_group.get("title"),
                    "description": candidate_group.get("description"),
                    "best_video": best_video
                })
                
            payload = {
                "job_id": job_id,
                "ranked_sections": ranked_sections
            }
            try:
                requests.post("http://localhost:5000/colab/entregar_ranking", json=payload, headers=self.headers)
            except Exception as e:
                logging.error(f"Error en mock_colab_2 webhook: {e}")

        threading.Thread(target=mock_colab_2_work).start()
        return True
