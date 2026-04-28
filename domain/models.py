from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime

@dataclass
class User:
    id: int
    google_id: str
    email: str
    name: str
    picture: Optional[str] = None
    created_at: Optional[datetime] = None

@dataclass
class VideoCandidate:
    video_id: str
    title: str
    description: str
    url: str
    video_url: str
    thumbnail: str
    channel_title: str
    published_at: str
    views: int
    likes: int
    comments: int
    duration: str
    total_minutes: float
    score: float
    processed_transcript: Optional[str] = None
    similarity_score: Optional[float] = None

@dataclass
class Section:
    id: int
    title: str
    content: str
    video_url: Optional[str] = None
    duration: str = "N/A"
    classes: int = 1
    video_id: Optional[str] = None
    video_title: Optional[str] = None
    videos_data: Optional[List[VideoCandidate]] = None

@dataclass
class Course:
    id: Optional[str]
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
    Actúa como máquina de estados entre el pipeline local y los Colabs externos.
    """
    job_id: str
    prompt: str
    status: str = "pending"
    user_id: Optional[int] = None
    course_outline: Optional[Dict[str, Any]] = None
    sections_with_candidates: Optional[List[Dict[str, Any]]] = None
    final_course: Optional["Course"] = None
    error_message: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
