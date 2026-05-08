import os
import json
import logging
import requests
import time
from requests.exceptions import HTTPError

GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_API_URL_TEMPLATE = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent"

class AIServices:
    """
    Servicios para manejar llamadas a LLMs para estructurar cursos y validar prompts.
    """
    
    class PromptRejectedError(Exception):
        """Excepción lanzada cuando el prompt del usuario viola las políticas educativas."""
        pass

    def __init__(self):
        pass

    def validate_and_refine_prompt(self, prompt: str) -> dict:
        """
        Utiliza Gemini para validar que el prompt sea educativo y refinarlo.
        Retorna el prompt refinado o el original si la API falla.
        """
        validation_prompt = f"""
        Actúa como un Arquitecto Educativo experto en diseño curricular y validación de contenido.

        Analiza la siguiente solicitud de un usuario que desea generar un curso:

        Solicitud del usuario:
        \"\"\"{prompt}\"\"\"

        ---

        ### OBJETIVOS

        Debes realizar TRES tareas:

        1. VALIDACIÓN
        Determina si la solicitud es:
        - Educativa y constructiva
        - Legal y segura
        - No ofensiva ni dañina

        Rechaza si incluye:
        - Actividades ilegales
        - Daño físico o psicológico
        - Contenido explícito inapropiado
        - Ambigüedad extrema sin posibilidad de interpretación educativa

        ---

        2. REFINAMIENTO
        Convierte la solicitud en un título de curso:
        - Claro, formal y académico
        - Bien delimitado (ni muy amplio ni demasiado específico)
        - Corrigiendo errores ortográficos o jerga
        - Manteniendo la intención original del usuario

        Ejemplos:
        - "aprender a aser paginas" → "Fundamentos de Desarrollo Web Frontend"
        - "quiero vender más" → "Estrategias de Marketing y Ventas para Incrementar Ingresos"

        ---

        3. DETECCIÓN DE NIVEL

        Clasifica el nivel según complejidad implícita:

        - principiante: sin prerequisitos
        - intermedio: requiere bases previas
        - avanzado: requiere experiencia sólida
        - maestro: enfoque experto, especializado o profundo

        Si hay ambigüedad, elige el nivel MÁS CONSERVADOR.

        ---

        ### REGLAS DE SALIDA (CRÍTICO)

        - Responde SOLO en JSON válido
        - NO agregues explicaciones fuera del JSON
        - NO incluyas texto adicional

        ---

        ### FORMATO DE RESPUESTA

        Si NO es válido:
        {{
          "valid": false,
          "reason": "Explicación breve, clara y objetiva"
        }}

        Si es válido:
        {{
          "valid": true,
          "refined_prompt": "Título del curso claro, formal y delimitado",
          "detected_level": "principiante|intermedio|avanzado|maestro"
        }}
        """
        try:
            result = self.call_google_generative_api_for_json(validation_prompt)
            
            if not result.get("valid", True):
                reason = result.get('reason', 'El tema no cumple con las políticas educativas de la plataforma.')
                raise self.PromptRejectedError(reason)
            
            return result
            
        except self.PromptRejectedError:
            # Re-lanzar rechazos de política para que el pipeline lo marque como fallido
            raise
        except Exception as e:
            logging.error(f"Fallo técnico al validar la seguridad del prompt: {e}")
            raise Exception("No pudimos validar tu solicitud debido a un problema técnico con nuestros servidores de IA. Por favor, intenta de nuevo más tarde.")

    def call_google_generative_api_for_json(self, prompt: str, max_retries=3):
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY no configurada.")
            
        url = f"{GOOGLE_API_URL_TEMPLATE}?key={GOOGLE_API_KEY}"
        headers = {'Content-Type': 'application/json'}
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json"}
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, json=data, timeout=30)
                response.raise_for_status()
                
                response_json = response.json()
                content_text = response_json['candidates'][0]['content']['parts'][0]['text']

                # Limpieza proactiva: A veces Gemini envuelve el JSON en ```json ... ```
                content_text = content_text.strip()
                if content_text.startswith("```json"):
                    content_text = content_text[7:]
                elif content_text.startswith("```"):
                    content_text = content_text[3:]
                if content_text.endswith("```"):
                    content_text = content_text[:-3]
                content_text = content_text.strip()
                
                return json.loads(content_text)
                
            except HTTPError as e:
                status_code = e.response.status_code
                if status_code in [503, 429]:
                    wait_time = 2 ** attempt
                    logging.warning(f"Error {status_code} de Google. Reintentando en {wait_time}s... (Intento {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    raise e
            except json.JSONDecodeError as e:
                logging.warning(f"Gemini devolvió JSON malformado. Reintentando... (Intento {attempt + 1}/{max_retries})")
                if attempt == max_retries - 1:
                    raise Exception(f"Gemini devolvió JSON inválido repetidamente: {e}\nCrudo: {content_text}")
            except (KeyError, IndexError) as e:
                raise Exception(f"Estructura inesperada de la API de Google: {e}\nCrudo: {response.text}")
                
        raise Exception(f"Fallo al conectar con Gemini API después de {max_retries} intentos.")

    def get_course_outline(self, prompt: str, detected_level: str = "principiante") -> dict:
        """
        Genera el esquema del curso basado en el prompt (Migrado desde app-google-gpu.py).
        """
        levels = {
            "principiante": {"num_sections": 4, "description": "Conceptos fundamentales", "depth": "superficial"},
            "intermedio": {"num_sections": 6, "description": "Profundización práctica", "depth": "moderada"},
            "avanzado": {"num_sections": 8, "description": "Técnicas avanzadas", "depth": "profunda"},
            "maestro": {"num_sections": 10, "description": "Cobertura exhaustiva", "depth": "muy profunda"}
        }
        
        level = detected_level if detected_level in levels else "principiante"
        level_config = levels[level]
        
        content_system_message = f"""
        Actúa como un Arquitecto Educativo experto en diseño curricular.
        
        Diseña el esquema detallado para un curso con las siguientes características:
        - TEMA: "{prompt}"
        - NIVEL: {level} ({level_config['description']})
        - PROFUNDIDAD: {level_config['depth']}
        - CANTIDAD DE SECCIONES: EXACTAMENTE {level_config['num_sections']} secciones.

        ---

        ### REGLAS DE DISEÑO
        1. Las secciones deben tener una progresión lógica (de menor a mayor complejidad).
        2. NO incluyas referencias a videos, URLs, canales de YouTube o material externo.
        3. Los títulos de las secciones deben ser cortos y académicos.
        4. Las descripciones de las secciones deben detallar EXACTAMENTE qué conceptos teóricos se enseñarán.
        5. Formula "learningOutcomes" (Resultados de Aprendizaje) claros y medibles.

        ---

        ### REGLAS DE SALIDA (CRÍTICO)
        - Responde SOLO en JSON válido.
        - NO agregues explicaciones fuera del JSON.
        - NO incluyas texto adicional ni marcas de markdown innecesarias.

        ---

        ### FORMATO DE RESPUESTA ESPERADO
        {{
            "title": "Título oficial del curso",
            "introduction": "Un párrafo motivador que explique qué logrará el estudiante",
            "sections": [
                {{
                    "title": "Nombre de la sección",
                    "description": "Lista de conceptos clave a enseñar aquí",
                    "level": "{level}"
                }}
            ],
            "learningOutcomes": ["Habilidad concreta 1", "Habilidad concreta 2"],
            "requirements": ["Conocimiento previo 1 o N/A", "Conocimiento previo 2"],
            "level": "{level}",
            "level_description": "{level_config['description']}"
        }}
        """
        
        return self.call_google_generative_api_for_json(content_system_message)
