import os
import json
import requests

GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash-preview-05-20")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_API_URL_TEMPLATE = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent"

class AIServices:
    """
    Servicios para manejar llamadas a LLMs para estructurar cursos y validar prompts.
    """
    
    def __init__(self):
        pass

    # TODO: Implementar Capa de Validación y Refinamiento (Guardrails y Refinador de prompt)
    def validate_and_refine_prompt(self, prompt: str) -> str:
        """
        [ESQUELETO] Debe utilizar un LLM para validar términos y condiciones y refinar jergas.
        Por ahora, retorna el mismo prompt.
        """
        return prompt

    def call_google_generative_api_for_json(self, prompt: str):
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY no configurada.")
            
        url = f"{GOOGLE_API_URL_TEMPLATE}?key={GOOGLE_API_KEY}"
        headers = {'Content-Type': 'application/json'}
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json"}
        }
        
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        response_json = response.json()
        try:
            content_text = response_json['candidates'][0]['content']['parts'][0]['text']
            return json.loads(content_text)
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            raise Exception(f"Error procesando la respuesta de Gemini: {e}\nRespuesta cruda: {response.text}")

    def get_course_outline(self, prompt: str) -> dict:
        """
        Genera el esquema del curso basado en el prompt (Migrado desde app-google-gpu.py).
        """
        levels = {
            "principiante": {"keywords": ["principiante", "inicial", "básico"], "num_sections": 4, "description": "Conceptos fundamentales", "depth": "superficial"},
            "intermedio": {"keywords": ["intermedio", "medio"], "num_sections": 6, "description": "Profundización práctica", "depth": "moderada"},
            "avanzado": {"keywords": ["avanzado", "experto"], "num_sections": 8, "description": "Técnicas avanzadas", "depth": "profunda"},
            "maestro": {"keywords": ["maestro", "completo"], "num_sections": 10, "description": "Cobertura exhaustiva", "depth": "muy profunda"}
        }
        
        level = "principiante"
        prompt_lower = prompt.lower()
        for level_name, level_info in levels.items():
            if any(keyword in prompt_lower for keyword in level_info["keywords"]):
                level = level_name
                break

        # TODO: Refactorizar para extraer el topic de mejor manera como estaba en app-google-gpu.py (ahora uso una versión rápida).
        topic = prompt.split(' ')[0]

        level_config = levels[level]
        
        content_system_message = f"""
        Eres un experto en diseño de cursos. Crea un esquema para un curso de nivel {level} sobre {topic}.
        El curso debe tener {level_config['num_sections']} secciones.
        Asegúrate de no incluir referencias a videos o URLs.
        Proporciona la respuesta en formato JSON con la siguiente estructura:
        {{
            "title": "Título del curso",
            "introduction": "Introducción motivadora",
            "sections": [
                {{
                    "title": "Título de la sección",
                    "description": "Descripción detallada",
                    "level": "{level}"
                }}
            ],
            "learningOutcomes": ["Objetivo 1"],
            "requirements": ["Requisito 1"],
            "level": "{level}",
            "level_description": "{level_config['description']}"
        }}
        """
        
        return self.call_google_generative_api_for_json(content_system_message)
