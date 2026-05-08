from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime

@dataclass
class User:
    id_usuario: int
    id_google: str
    correo: str
    nombre_usuario: str
    nombre_completo: Optional[str] = None
    foto_perfil_url: Optional[str] = None
    creado_en: Optional[datetime] = None
    actualizado_en: Optional[datetime] = None
    esta_activo: bool = True

@dataclass
class Section:
    id: int
    title: str
    content: str = ""
    video_url: str = None
    duration: str = "N/A"
    classes: int = 1
    video_id: str = None
    video_title: str = None

@dataclass
class Course:
    id: str
    title: str
    introduction: str
    instructor: str
    rating: float
    students: int
    last_updated: str
    language: str
    total_duration: str
    total_lessons: int
    sections: List[Section]
    learning_outcomes: List[str]
    requirements: List[str]
    level: str
    level_description: str
    user_id: Optional[int] = None

@dataclass
class CourseJob:
    """
    Representa el estado de un trabajo de generación de curso.
    Status permitidos por DB: 'pending', 'processing', 'completed', 'failed'
    """
    job_id: str
    prompt: str
    status: str = "pending"
    user_id: Optional[int] = None
    course_outline: Optional[Dict[str, Any]] = None
    sections_with_candidates: Optional[List[Dict[str, Any]]] = None
    final_course: Optional[Any] = None
    error_message: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
