import time
import uuid
import random
import logging
import threading
from datetime import datetime
from dataclasses import asdict

from infrastructure.api.youtube_api import YouTubeAPIClient
from infrastructure.api.sentiment_analyzer_client import SentimentAnalyzerClient
from infrastructure.evaluators.orquestador import WhisperOrchestratorClient
from infrastructure.api.video_validator_client import VideoValidatorClient
from infrastructure.db.course_job_repository import CourseJobRepository
from application.ai_services import AIServices
from domain.models import Course, Section, CourseJob
from presentation.extensions import socketio


class CoursePipeline:
    def __init__(self):
        self.yt_client = YouTubeAPIClient()
        self.ai_services = AIServices()
        self.sentiment_client = SentimentAnalyzerClient()
        self.whisper_orchestrator = WhisperOrchestratorClient()
        self.video_validator_client = VideoValidatorClient()
        self.job_repo = CourseJobRepository()

    def _emit_progress(self, job_id: str, percentage: int, message: str):
        """Emite actualizaciones por WebSocket al cliente conectado a la sala del job_id."""
        logging.info(f"[{job_id}] {percentage}% - {message}")
        try:
            socketio.emit('progress_update', {
                'job_id': job_id,
                'percentage': percentage,
                'message': message
            }, to=f"job_{job_id}")
        except AttributeError:
            pass # Para cuando se ejecuta desde un script fuera de Flask

    def _set_error(self, job: CourseJob, message: str):
        """Marca un job como fallido."""
        job.status = "failed"
        job.error_message = message
        self.job_repo.save_job(job)
        self._emit_progress(job.job_id, 0, f"Error: {message}")

    # ========================================================================
    # PUNTO DE ENTRADA: Se llama desde el endpoint REST
    # ========================================================================
    def iniciar_generacion(self, prompt: str, user_id: int = None) -> str:
        """
        Crea el job y lanza el pipeline completo en un hilo separado
        para no bloquear la respuesta HTTP al frontend.
        """
        job_id = str(uuid.uuid4())
        job = CourseJob(
            job_id=job_id,
            prompt=prompt,
            status="pending",
            user_id=user_id
        )
        self.job_repo.save_job(job)
        self._emit_progress(job_id, 5, "Solicitud recibida. Iniciando generación...")

        # Lanzar el pipeline completo en background
        thread = threading.Thread(target=self._run_full_pipeline, args=(job_id,))
        thread.start()

        return job_id

    # ========================================================================
    # PIPELINE COMPLETO (8 ETAPAS) — Corre en un hilo separado
    # ========================================================================
    def _run_full_pipeline(self, job_id: str):
        try:
            job = self.job_repo.get_job(job_id)
            if not job:
                return

            job.status = "processing"
            self.job_repo.save_job(job)

            # --- ETAPA 1: Validar y refinar prompt ---
            self._emit_progress(job_id, 10, "Validando y refinando tu solicitud con IA...")
            validation_result = self.ai_services.validate_and_refine_prompt(job.prompt)
            
            # Extraer las variables del dict o hacer fallback al original en caso de error técnico
            if isinstance(validation_result, dict):
                refined_prompt = validation_result.get("refined_prompt", job.prompt)
                detected_level = validation_result.get("detected_level", "principiante")
            else:
                refined_prompt = validation_result
                detected_level = "principiante"
                
            logging.info(f"[{job_id}] Prompt refinado: {refined_prompt} (Nivel: {detected_level}) Prompt original: {job.prompt}")

            # --- ETAPA 2: Generar estructura del curso ---
            self._emit_progress(job_id, 20, "Construyendo la estructura del curso con IA...")
            course_outline = self.ai_services.get_course_outline(refined_prompt, detected_level)
            
            # --- DEBUG INFO PARA FASE DE PRUEBAS ---
            course_outline["_debug_prompt_original"] = job.prompt
            course_outline["_debug_prompt_refinado"] = refined_prompt
            
            job.course_outline = course_outline
            self.job_repo.save_job(job)
            logging.info(f"[{job_id}] Outline generado: {course_outline.get('title', 'Sin título')}")

            # --- ETAPA 3: Buscar videos en YouTube ---
            self._emit_progress(job_id, 30, "Buscando los mejores videos en YouTube...")
            job.sections_with_candidates = self._buscar_videos(job, course_outline)
            self.job_repo.save_job(job)

            # --- ETAPA 4: Filtrar por métricas (calculate_video_score, ya integrado en YouTube) ---
            self._emit_progress(job_id, 45, "Filtrando videos por métricas de calidad...")
            job.sections_with_candidates = self._filtrar_por_metricas(job.sections_with_candidates)
            self.job_repo.save_job(job)

            # --- ETAPA 5: Análisis de sentimiento ---
            self._emit_progress(job_id, 55, "Analizando sentimiento de la comunidad...")
            job.sections_with_candidates = self._analizar_sentimiento(job.sections_with_candidates)
            self.job_repo.save_job(job)

            # --- ETAPA 6: Transcripción (Fase 3 Whisper) ---
            self._emit_progress(job_id, 70, "Transcribiendo videos seleccionados...")
            job.sections_with_candidates = self._transcribir_videos(job.sections_with_candidates)
            self.job_repo.save_job(job)

            # --- ETAPA 7: Validación RAG (cramsoft.dev) ---
            self._emit_progress(job_id, 80, "Verificando relevancia del contenido con IA...")
            job.sections_with_candidates = self._validar_contenido_rag(job.sections_with_candidates)
            self.job_repo.save_job(job)

            # --- ETAPA 8: Ensamblar curso ---
            self._emit_progress(job_id, 90, "Ensamblando tu curso personalizado...")
            course = self._ensamblar_curso(job, course_outline, job.sections_with_candidates)

            job.final_course = course
            job.status = "completed"
            self.job_repo.save_job(job)

            self._emit_progress(job_id, 100, "¡Tu curso está listo!")

        except Exception as e:
            logging.error(f"[{job_id}] Error en pipeline: {e}", exc_info=True)
            job = self.job_repo.get_job(job_id)
            if job:
                self._set_error(job, str(e))

    # ========================================================================
    # ETAPA 3: Búsqueda de videos
    # ========================================================================
    def _buscar_videos(self, job: CourseJob, course_outline: dict) -> list:
        topic = job.prompt
        sections_with_candidates = []
        used_video_ids = set()

        # Videos para la introducción
        intro_desc = course_outline.get("introduction", "")
        if intro_desc:
            intro_cands = self.yt_client.search_videos(
                query=f"introduccion a {topic}",
                section_content=intro_desc,
                used_video_ids=used_video_ids,
                max_results=5
            )
            if intro_cands:
                for v in intro_cands:
                    used_video_ids.add(v.get("videoId"))
                sections_with_candidates.append({
                    "is_intro": True,
                    "section_id": 0,
                    "title": "Introducción",
                    "candidates": intro_cands
                })

        # Videos para cada sección
        for i, section_data in enumerate(course_outline.get("sections", [])):
            section_id = i + 1
            section_query = f"{section_data.get('title', '')} {topic}".strip()

            candidates = self.yt_client.search_videos(
                query=section_query,
                section_content=section_data.get("description", ""),
                used_video_ids=used_video_ids,
                max_results=5
            )

            for v in candidates:
                used_video_ids.add(v.get("videoId"))

            sections_with_candidates.append({
                "is_intro": False,
                "section_id": section_id,
                "title": section_data.get('title', ''),
                "description": section_data.get('description', ''),
                "candidates": candidates
            })

        return sections_with_candidates

    # ========================================================================
    # ETAPA 4: Filtrar por métricas
    # ========================================================================
    def _filtrar_por_metricas(self, sections_with_candidates: list) -> list:
        """
        Filtra los candidatos por score de métricas (ya calculado en YouTube API).
        Solo pasan los videos con score > 0.3
        """
        min_score = 0.3
        filtered = []

        for section in sections_with_candidates:
            good_candidates = [
                v for v in section.get("candidates", [])
                if v.get("score", 0) > min_score
            ]

            if not good_candidates and section.get("candidates"):
                # Si ninguno pasa el filtro, al menos quedarse con el mejor
                good_candidates = [max(section["candidates"], key=lambda x: x.get("score", 0))]

            filtered.append({
                **section,
                "candidates": good_candidates
            })

        return filtered

    # ========================================================================
    # ETAPA 5: Análisis de sentimiento
    # ========================================================================
    def _analizar_sentimiento(self, sections: list) -> list:
        """
        Envía cada video candidato al analizador de sentimiento.
        Filtra por porcentaje_utiles >= 60%.
        """
        result = []

        for section in sections:
            analyzed_candidates = []

            for video in section.get("candidates", []):
                video_url = video.get("url", "")
                if not video_url:
                    continue

                sentiment = self.sentiment_client.analyze(video_url)
                video["sentiment"] = sentiment

                if sentiment.get("passed", False):
                    analyzed_candidates.append(video)
                    logging.info(f"Video {video.get('videoId')} PASÓ sentimiento: {sentiment.get('porcentaje_utiles')}%")
                else:
                    logging.info(f"Video {video.get('videoId')} NO PASÓ sentimiento: {sentiment.get('porcentaje_utiles')}%")

            if not analyzed_candidates and section.get("candidates"):
                # Fallback: si ninguno pasa, quedarse con el de mayor porcentaje
                best = max(section["candidates"], key=lambda x: x.get("sentiment", {}).get("porcentaje_utiles", 0))
                analyzed_candidates = [best]

            result.append({
                **section,
                "candidates": analyzed_candidates
            })

        return result

    # ========================================================================
    # ETAPA 6: Transcripción (MOCK Fase 2)
    # ========================================================================
    def _transcribir_videos(self, sections: list) -> list:
        """
        Transcribe cada video candidato usando WhisperOrchestrator (Fase 3).
        Distribuye la carga entre GPUs de Colab vía ngrok.
        """
        # 1. Recolectar todos los IDs de los videos a transcribir
        video_ids = []
        for section in sections:
            for video in section.get("candidates", []):
                url = video.get("url", "")
                # Extraer el ID de youtube. Ejemplo: https://www.youtube.com/watch?v=pmOgMzBZw2w -> pmOgMzBZw2w
                if "v=" in url:
                    vid_id = url.split("v=")[1].split("&")[0]
                    video["youtube_id"] = vid_id
                    video_ids.append(vid_id)

        if not video_ids:
            return sections

        # 2. Orquestar la transcripción de forma concurrente
        transcription_results = self.whisper_orchestrator.transcribe_videos_sync(video_ids)

        # 3. Inyectar las transcripciones de vuelta en las secciones
        for section in sections:
            for video in section.get("candidates", []):
                vid_id = video.get("youtube_id")
                if vid_id and vid_id in transcription_results:
                    data = transcription_results[vid_id]
                    if "error" not in data:
                        video["transcript"] = data.get("full_text", "")
                    else:
                        logging.warning(f"Transcripción fallida para {vid_id}: {data['error']}")
                        video["transcript"] = ""

        return sections

    # ========================================================================
    # ETAPA 7: Validación RAG (cramsoft.dev)
    # ========================================================================
    def _validar_contenido_rag(self, sections: list) -> list:
        """
        Envía la transcripción de cada video al validador RAG de cramsoft.dev.
        Verifica que el contenido sea relevante al tema del curso.
        """
        result = []

        for section in sections:
            validated_candidates = []

            for video in section.get("candidates", []):
                transcript = video.get("transcript", "")
                if not transcript:
                    continue

                validation = self.video_validator_client.validate(transcript)
                video["rag_validation"] = validation
                validated_candidates.append(video)
                logging.info(f"Video {video.get('videoId')} validación RAG: {validation}")

            if not validated_candidates and section.get("candidates"):
                validated_candidates = section["candidates"][:1]

            result.append({
                **section,
                "candidates": validated_candidates
            })

        return result

    # ========================================================================
    # ETAPA 8: Ensamblar curso final
    # ========================================================================
    def _ensamblar_curso(self, job: CourseJob, course_outline: dict, validated_sections: list) -> Course:
        sections = []

        for section_result in validated_sections:
            # Tomar el mejor candidato (el primero que sobrevivió todos los filtros)
            best_video = section_result["candidates"][0] if section_result.get("candidates") else None

            if section_result.get("is_intro"):
                continue

            section = Section(
                id=section_result.get("section_id"),
                title=section_result.get("title"),
                content=section_result.get("description", ""),
                video_url=f"https://www.youtube.com/embed/{best_video['videoId']}" if best_video else None,
                duration=best_video.get("duration", "N/A") if best_video else "N/A",
                classes=1,
                video_id=best_video.get("videoId") if best_video else None,
                video_title=best_video.get("title") if best_video else None,
            )
            sections.append(section)

        # Sección de Evaluación Final
        sections.append(Section(
            id=len(sections) + 1,
            title="Evaluación Final",
            content="Evalúa lo aprendido en el curso.",
            classes=1
        ))

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
            total_duration="0m",
            total_lessons=len(sections),
            sections=sections,
            learning_outcomes=course_outline.get("learningOutcomes", []),
            requirements=course_outline.get("requirements", []),
            level=course_outline.get("level", "principiante"),
            level_description=course_outline.get("level_description", ""),
            user_id=job.user_id
        )

        return course
