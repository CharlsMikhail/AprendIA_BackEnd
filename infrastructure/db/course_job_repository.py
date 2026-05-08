import json
import logging
from domain.models import CourseJob
from infrastructure.db.database import DatabaseConnectionManager

class CourseJobRepository:
    """Repositorio para persistir el estado de los jobs en PostgreSQL (Neon)."""

    def save_job(self, job: CourseJob):
        with DatabaseConnectionManager() as cursor:
            cursor.execute("SELECT job_id FROM course_jobs WHERE job_id = %s", (job.job_id,))
            exists = cursor.fetchone()

            outline_json = json.dumps(job.course_outline) if job.course_outline else None
            candidates_json = json.dumps(job.sections_with_candidates) if job.sections_with_candidates else None
            
            # Serializar final_course
            if job.final_course and hasattr(job.final_course, '__dict__'):
                from dataclasses import asdict
                final_course_json = json.dumps(asdict(job.final_course))
            elif job.final_course:
                final_course_json = json.dumps(job.final_course)
            else:
                final_course_json = None

            if exists:
                cursor.execute("""
                    UPDATE course_jobs 
                    SET status = %s, course_outline = %s, sections_with_candidates = %s, 
                        final_course = %s, error_message = %s, actualizado_en = NOW()
                    WHERE job_id = %s
                """, (job.status, outline_json, candidates_json, final_course_json, job.error_message, job.job_id))
            else:
                cursor.execute("""
                    INSERT INTO course_jobs (job_id, id_usuario, prompt, status, course_outline, 
                                            sections_with_candidates, final_course, error_message)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (job.job_id, job.user_id, job.prompt, job.status, outline_json, 
                      candidates_json, final_course_json, job.error_message))

    def get_job(self, job_id: str) -> CourseJob:
        with DatabaseConnectionManager() as cursor:
            cursor.execute("""
                SELECT job_id, id_usuario, prompt, status, course_outline, 
                       sections_with_candidates, final_course, error_message
                FROM course_jobs WHERE job_id = %s
            """, (job_id,))
            row = cursor.fetchone()
            if not row:
                return None

            return CourseJob(
                job_id=row[0],
                user_id=row[1],
                prompt=row[2],
                status=row[3],
                course_outline=row[4] if isinstance(row[4], dict) else (json.loads(row[4]) if row[4] else None),
                sections_with_candidates=row[5] if isinstance(row[5], list) else (json.loads(row[5]) if row[5] else None),
                final_course=row[6] if isinstance(row[6], dict) else (json.loads(row[6]) if row[6] else None),
                error_message=row[7]
            )

    def update_job_status(self, job_id: str, status: str, error_message: str = None):
        with DatabaseConnectionManager() as cursor:
            if error_message:
                cursor.execute("""
                    UPDATE course_jobs SET status = %s, error_message = %s, actualizado_en = NOW() 
                    WHERE job_id = %s
                """, (status, error_message, job_id))
            else:
                cursor.execute("""
                    UPDATE course_jobs SET status = %s, actualizado_en = NOW() 
                    WHERE job_id = %s
                """, (status, job_id))
