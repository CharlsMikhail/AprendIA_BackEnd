import time
import random
from datetime import datetime
from application.ai_services import AIServices
from infrastructure.api.youtube_api import YouTubeAPIClient
from infrastructure.evaluators.transcription_evaluator import TranscriptionEvaluator
from infrastructure.inference.sentiment_analyzer import TrustIndexAnalyzer
from domain.models import Course, Section, VideoCandidate

class CoursePipeline:
    def __init__(self):
        self.ai_service = AIServices()
        self.yt_client = YouTubeAPIClient()
        self.transcription_evaluator = TranscriptionEvaluator()
        self.sentiment_analyzer = TrustIndexAnalyzer()

    def generate_course(self, prompt: str, user_id: int = None) -> Course:
        # 1. Validar y refinar prompt
        refined_prompt = self.ai_service.validate_and_refine_prompt(prompt)

        # 2. Estructurar curso
        course_outline = self.ai_service.get_course_outline(refined_prompt)
        topic = refined_prompt.split(' ')[0] # TODO: Mejor extracción
        
        sections = []
        used_video_ids = set()

        for i, section_data in enumerate(course_outline.get("sections", [])):
            section_id = i + 1
            section_query = f"{section_data.get('title', '')} {topic}".strip()

            # 3. Buscar y Filtrar Videos (YouTube + Balance Metrics)
            top_videos_data = self.yt_client.search_videos(
                query=section_query,
                section_content=section_data.get("description", ""),
                used_video_ids=used_video_ids
            )

            # TODO: 4. Análisis de Sentimiento (Trust Index)
            # Aquí iteraríamos sobre top_videos_data, descargaríamos comentarios y usaríamos self.sentiment_analyzer

            # TODO: 5. Transcripción y RAG
            # Evaluar transcriptions con self.transcription_evaluator y comparar con RAG

            # TODO: 6. Ranking Final
            # Combinar scores y escoger el mejor video

            best_video_data = top_videos_data[0] if top_videos_data else None
            
            section = Section(
                id=section_id,
                title=section_data.get("title", f"Sección {section_id}"),
                content=section_data.get("description", ""),
                video_url=f"https://www.youtube.com/embed/{best_video_data['videoId']}" if best_video_data else None,
                duration=best_video_data["duration"] if best_video_data else "N/A",
                classes=1,
                video_id=best_video_data["videoId"] if best_video_data else None,
                video_title=best_video_data["title"] if best_video_data else None,
            )
            
            if best_video_data:
                used_video_ids.add(best_video_data["videoId"])

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
                    mins = int(s.duration.split(' ')[0].replace('m','').replace('h','')) # simplificación
                    total_minutes += mins
                except:
                    pass

        total_duration_str = f"{total_minutes}m"
        current_date = datetime.now().strftime("%m/%Y")

        course = Course(
            id=f"course_{int(time.time())}",
            title=course_outline.get("title", f"Curso sobre {prompt}"),
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
            user_id=user_id
        )

        return course
