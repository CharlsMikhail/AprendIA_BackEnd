import time
import uuid
import random
from datetime import datetime
from infrastructure.api.youtube_api import YouTubeAPIClient
from infrastructure.api.colab_client import ColabClient
from infrastructure.db.course_job_repository import CourseJobRepository
from domain.models import Course, Section, CourseJob
from app import socketio  # Asumiendo que podemos importarlo sin circular, si falla lo ajustaremos.

class CoursePipeline:
    def __init__(self):
        self.yt_client = YouTubeAPIClient()
        self.colab_client = ColabClient()
        self.job_repo = CourseJobRepository()

    def _emit_progress(self, job_id: str, percentage: int, message: str):
        """Emite actualizaciones por WebSocket al cliente conectado a la sala del job_id."""
        socketio.emit('progress_update', {
            'job_id': job_id,
            'percentage': percentage,
            'message': message
        }, to=f"job_{job_id}")

    def iniciar_generacion(self, prompt: str, user_id: int = None) -> str:
        """
        Paso 1: Se llama desde el endpoint REST inicial.
        Guarda el job y delega a Colab #1.
        """
        job_id = str(uuid.uuid4())
        job = CourseJob(
            job_id=job_id,
            prompt=prompt,
            status="pending",
            user_id=user_id
        )
        self.job_repo.save_job(job)
        self._emit_progress(job_id, 10, "Iniciando validación y estructura del curso en IA...")

        # Dispara la llamada a Colab #1 (fire and forget asíncrono o sincrono muy rápido)
        # El colab procesará y luego llamará al webhook /colab/entregar_esquema
        success = self.colab_client.trigger_outline_generation(job_id, prompt)
        if not success:
            job.status = "error"
            job.error_message = "No se pudo contactar al Colab generador."
            self.job_repo.save_job(job)
            self._emit_progress(job_id, 0, "Error contactando al servidor de IA.")

        return job_id

    def procesar_esquema(self, job_id: str, course_outline: dict):
        """
        Paso 2: Se llama cuando Colab #1 entrega el JSON del esquema.
        Realiza la búsqueda de videos localmente y envía a Colab #2.
        """
        job = self.job_repo.get_job(job_id)
        if not job:
            return

        job.course_outline = course_outline
        job.status = "outline_ready"
        self.job_repo.save_job(job)
        
        self._emit_progress(job_id, 30, "Esquema generado. Buscando los mejores videos en YouTube...")

        topic = job.prompt.split(' ')[0] # TODO: Mejorar extracción si viene del outline
        
        sections_with_candidates = []
        used_video_ids = set()

        # Buscar videos para la introducción
        intro_desc = course_outline.get("introduction", "")
        if intro_desc:
            intro_cands = self.yt_client.search_videos(
                query=f"introduccion a {topic}",
                section_content=intro_desc,
                used_video_ids=used_video_ids,
                max_results=5
            )
            if intro_cands:
                sections_with_candidates.append({
                    "is_intro": True,
                    "section_id": 0,
                    "title": "Introducción",
                    "candidates": intro_cands
                })

        # Buscar videos para el resto de las secciones
        for i, section_data in enumerate(course_outline.get("sections", [])):
            section_id = i + 1
            section_query = f"{section_data.get('title', '')} {topic}".strip()

            candidates = self.yt_client.search_videos(
                query=section_query,
                section_content=section_data.get("description", ""),
                used_video_ids=used_video_ids,
                max_results=5
            )
            
            sections_with_candidates.append({
                "is_intro": False,
                "section_id": section_id,
                "title": section_data.get('title', ''),
                "description": section_data.get('description', ''),
                "candidates": candidates
            })

        job.sections_with_candidates = sections_with_candidates
        job.status = "ranking_pending"
        self.job_repo.save_job(job)

        self._emit_progress(job_id, 60, "Videos encontrados. Evaluando transcripciones y sentimiento de la comunidad...")

        # Dispara Colab #2 para análisis de sentimiento, transcripción y ranking final
        success = self.colab_client.trigger_ranking_analysis(job_id, sections_with_candidates)
        if not success:
            job.status = "error"
            job.error_message = "No se pudo contactar al Colab de análisis de calidad."
            self.job_repo.save_job(job)
            self._emit_progress(job_id, 0, "Error en el análisis de calidad de videos.")

    def procesar_ranking(self, job_id: str, ranked_sections: list) -> Course:
        """
        Paso 3: Se llama cuando Colab #2 entrega el ranking final.
        Ensambla el objeto Course final y lo guarda.
        """
        job = self.job_repo.get_job(job_id)
        if not job:
            return None

        course_outline = job.course_outline
        
        self._emit_progress(job_id, 90, "Análisis finalizado. Ensamblando tu curso...")

        sections = []
        intro_data = None

        # Procesar los resultados devueltos por el colab
        # asumimos que ranked_sections contiene el best_video elegido para cada seccion
        for section_result in ranked_sections:
            best_video = section_result.get("best_video")
            if section_result.get("is_intro"):
                intro_data = best_video
                continue

            section = Section(
                id=section_result.get("section_id"),
                title=section_result.get("title"),
                content=section_result.get("description"),
                video_url=f"https://www.youtube.com/embed/{best_video['videoId']}" if best_video else None,
                duration=best_video.get("duration", "N/A") if best_video else "N/A",
                classes=1,
                video_id=best_video.get("videoId") if best_video else None,
                video_title=best_video.get("title") if best_video else None,
            )
            sections.append(section)

        # Evaluación Final
        sections.append(Section(
            id=len(sections) + 1,
            title="Evaluación Final",
            content="Evalúa lo aprendido en el curso.",
            classes=1
        ))

        total_minutes = 0
        for s in sections:
            if s.duration != "N/A":
                try:
                    mins = int(s.duration.split(' ')[0].replace('m','').replace('h','')) 
                    total_minutes += mins
                except:
                    pass

        total_duration_str = f"{total_minutes}m"
        current_date = datetime.now().strftime("%m/%Y")

        course = Course(
            id=f"course_{int(time.time())}",
            title=course_outline.get("title", f"Curso sobre {job.prompt}"),
            introduction=course_outline.get("introduction", ""),
            instructor="IA Professor",
            rating=round(random.uniform(4.5, 4.9), 1),
            students=random.randint(5000, 15000),
            last_updated=current_date,
            language="Español",
            total_duration=total_duration_str,
            total_lessons=len(sections),
            sections=sections,
            learning_outcomes=course_outline.get("learningOutcomes", []),
            requirements=course_outline.get("requirements", []),
            level=course_outline.get("level", "principiante"),
            level_description=course_outline.get("level_description", ""),
            user_id=job.user_id
        )

        job.final_course = course
        job.status = "completed"
        self.job_repo.save_job(job)

        self._emit_progress(job_id, 100, "¡Tu curso está listo!")
        
        # En una versión completa aquí llamaríamos a CourseRepository.save(course)
        
        return course
