from typing import Dict, Optional
from domain.models import CourseJob

class JobStore:
    """
    Almacén en memoria para los trabajos (CourseJobs) en progreso.
    Si se reinicia el servidor, los trabajos se perderán. En un futuro
    se puede persistir en PostgreSQL.
    """
    def __init__(self):
        self._jobs: Dict[str, CourseJob] = {}

    def save_job(self, job: CourseJob):
        self._jobs[job.job_id] = job

    def get_job(self, job_id: str) -> Optional[CourseJob]:
        return self._jobs.get(job_id)

    def update_job_status(self, job_id: str, status: str, error_message: str = None):
        job = self.get_job(job_id)
        if job:
            job.status = status
            if error_message:
                job.error_message = error_message
            self.save_job(job)

# Instancia global (Singleton para el entorno Flask en memoria)
job_store = JobStore()
