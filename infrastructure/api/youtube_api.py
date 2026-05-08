import os
import re
import logging
from datetime import datetime
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SPANISH_STOP_WORDS = ['el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'y', 'o', 'pero', 'si', 'de', 'en', 'para', 'por', 'con', 'sin', 'sobre', 'a', 'al', 'del', 'es', 'son', 'fue', 'ser', 'este', 'esta', 'ese', 'esa', 'que', 'como', 'cuando', 'donde', 'quien']

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

class YouTubeAPIClient:
    def __init__(self):
        self.youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

    def search_videos(self, query: str, section_content: str = "", used_video_ids: set = None, max_results=10) -> list:
        """Search for YouTube videos based on query and return the best match"""
        if used_video_ids is None:
            used_video_ids = set()

        try:
            request_youtube = self.youtube.search().list(
                part="snippet",
                q=query,
                type="video",
                maxResults=5, 
                relevanceLanguage="es",
                videoDuration="medium",
                order="relevance"
            )
            response = request_youtube.execute()

            all_videos_data = []
            if response.get("items"):
                video_ids = [item["id"]["videoId"] for item in response["items"]]

                video_details = self.youtube.videos().list(
                    part="statistics,contentDetails,snippet",
                    id=",".join(video_ids)
                ).execute()

                details_map = {item["id"]: item for item in video_details["items"]}

                for item in response["items"]:
                    video_id = item["id"]["videoId"]

                    if video_id in used_video_ids:
                        continue

                    video_details_data = details_map.get(video_id, {})
                    statistics = video_details_data.get("statistics", {})
                    content_details = video_details_data.get("contentDetails", {})
                    snippet = video_details_data.get("snippet", {})

                    duration = content_details.get("duration", "PT0M0S")
                    minutes_match = re.search(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration)
                    if minutes_match:
                        hours = int(minutes_match.group(1) or 0)
                        minutes = int(minutes_match.group(2) or 0)
                        seconds = int(minutes_match.group(3) or 0)
                        total_minutes = (hours * 60) + minutes + (seconds / 60)
                        duration_str = f"{hours}h {minutes}m" if hours > 0 else f"{minutes} min"
                    else:
                        total_minutes = 0
                        duration_str = "Desconocido"

                    published_at = datetime.strptime(snippet.get("publishedAt", ""), "%Y-%m-%dT%H:%M:%SZ")
                    days_since_published = (datetime.now() - published_at).days

                    # Calcular puntuación inicial
                    score = self.calculate_video_score(video_details_data, snippet, statistics, days_since_published, total_minutes, section_content)

                    video_data = {
                        "title": item["snippet"]["title"],
                        "description": item["snippet"]["description"],
                        "url": f"https://www.youtube.com/watch?v={video_id}",
                        "videoUrl": f"https://www.youtube.com/embed/{video_id}",
                        "thumbnail": item["snippet"]["thumbnails"]["high"]["url"],
                        "channelTitle": item["snippet"]["channelTitle"],
                        "publishedAt": item["snippet"]["publishedAt"],
                        "views": int(statistics.get("viewCount", 0)),
                        "likes": int(statistics.get("likeCount", 0)),
                        "comments": int(statistics.get("commentCount", 0)),
                        "duration": duration_str,
                        "score": score,
                        "videoId": video_id,
                        "totalMinutes": total_minutes
                    }

                    all_videos_data.append(video_data)

                # Ordenar los videos por score y tomar la cantidad pedida
                sorted_videos = sorted(all_videos_data, key=lambda x: x["score"], reverse=True)
                top_videos = sorted_videos[:max_results]

                return top_videos

            return []

        except HttpError as e:
            if e.resp.status == 403:
                logging.error(f"Error de Cuota o Permisos en API de YouTube (403): {e}")
            else:
                logging.error(f"Error HTTP de YouTube API: {e}")
            return []
        except Exception as e:
            logging.error(f"Error inesperado buscando videos en YouTube: {e}", exc_info=True)
            return []

    def calculate_video_score(self, video_details, snippet, statistics, days_since_published, total_minutes, section_content=""):
        """Calcula la puntuación de un video basada en múltiples criterios"""
        try:
            # 1. Relevancia (30%) - TF-IDF
            relevance_score = 0.0
            if section_content:
                video_text = f"{snippet.get('title', '')} {snippet.get('description', '')}"
                vectorizer = TfidfVectorizer(stop_words=SPANISH_STOP_WORDS)
                try:
                    tfidf_matrix = vectorizer.fit_transform([video_text, section_content])
                    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                    relevance_score = float(similarity)
                except Exception as e:
                    relevance_score = 0.5 
            else:
                relevance_score = 0.5 

            # 2. Verificación de transcripción (20%)
            # TODO: Integrar aquí la verificación oficial de transcripciones a través del evaluator.
            transcript_score = 1.0 # Placeholder

            # 3. Calidad del video (15%)
            quality_score = 0
            definition = video_details.get('contentDetails', {}).get('definition')
            if definition == 'hd':
                quality_score = 1.0
            elif definition == 'sd':
                quality_score = 0.5
            else:
                quality_score = 0.3

            # 4. Engagement (15%)
            views = int(statistics.get("viewCount", 0))
            likes = int(statistics.get("likeCount", 0))
            comments = int(statistics.get("commentCount", 0))

            engagement_score = 0
            if views > 100:
                like_ratio = (likes / views)
                comment_ratio = (comments / views)
                engagement_score = min(1.0, (like_ratio * 20) * 0.6 + (comment_ratio * 100) * 0.4)

            # 5. Actualidad (10%)
            if days_since_published <= 365:
                recency_score = 1.0
            elif days_since_published <= 365 * 3:
                recency_score = 0.8
            elif days_since_published <= 365 * 5:
                recency_score = 0.5
            else:
                recency_score = 0.3

            # 6. Duración (10%)
            duration_score = 1.0 if 5 <= total_minutes <= 25 else (0.5 if total_minutes < 5 else 0.7)

            # TODO: Refactorizar pesos y algoritmo de balance para incluir "métrica de creador" y nueva lógica.
            final_score = (
                    relevance_score * 0.30 +
                    transcript_score * 0.20 +
                    quality_score * 0.15 +
                    engagement_score * 0.15 +
                    recency_score * 0.10 +
                    duration_score * 0.10
            )

            return final_score

        except Exception as e:
            logging.error(f"Error calculando score del video: {e}")
            return 0.0
