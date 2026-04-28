import os
import re
from datetime import datetime
from googleapiclient.discovery import build
import concurrent.futures
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

class YouTubeAPIClient:
    def __init__(self):
        self.youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

    def search_videos(self, query: str, section_content: str = "", used_video_ids: set = None, max_results=10) -> list:
        # --- INICIO MOCK FASE 1 ---
        import logging
        logging.info(f"MOCK YOUTUBE: Buscando videos para '{query}'")
        import random
        mock_candidates = []
        for i in range(3):
            vid = f"mock_vid_{random.randint(1000, 9999)}"
            mock_candidates.append({
                "videoId": vid,
                "title": f"Video mockeado para {query} Parte {i+1}",
                "description": f"Descripción simulada del video {i+1}",
                "url": f"https://www.youtube.com/watch?v={vid}",
                "channelTitle": "Mock Channel",
                "publishedAt": "2023-01-01T00:00:00Z",
                "duration": f"{random.randint(5, 20)}m",
                "total_minutes": random.randint(5, 20),
                "views": random.randint(100, 10000),
                "likes": random.randint(10, 1000),
                "commentCount": random.randint(5, 100)
            })
        return mock_candidates
        # --- FIN MOCK FASE 1 ---

        """Search for YouTube videos based on query and return the best match"""
        if used_video_ids is None:
            used_video_ids = set()

        # Determinar el número de hilos
        max_workers = min(32, (os.cpu_count() or 1) * 2)

        try:
            # TODO: Implementar búsqueda base de 50 videos como se pidió en la nueva arquitectura
            # Actualmente se buscan 10 videos (según app-google-gpu.py).
            request_youtube = self.youtube.search().list(
                part="snippet",
                q=query,
                type="video",
                videoLicense="creativeCommon",
                maxResults=10, 
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

                    # TODO: La lógica de transcripción se movió. Integrar aquí el llamado al nuevo pipeline.
                    
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

                    # TODO: Filtrar videos con score > 0.3 según requerimiento
                    
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

                # Ordenar los videos por score y tomar los 5 mejores
                sorted_videos = sorted(all_videos_data, key=lambda x: x["score"], reverse=True)
                top_5_videos = sorted_videos[:5]

                return top_5_videos

            return []

        except Exception as e:
            print(f"Error searching YouTube videos: {e}")
            return []

    def calculate_video_score(self, video_details, snippet, statistics, days_since_published, total_minutes, section_content=""):
        """Calcula la puntuación de un video basada en múltiples criterios"""
        try:
            # 1. Relevancia (30%) - TF-IDF
            relevance_score = 0.0
            if section_content:
                video_text = f"{snippet.get('title', '')} {snippet.get('description', '')}"
                vectorizer = TfidfVectorizer(stop_words='english')
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
                quality_score = 0.8
            elif definition == 'sd':
                quality_score = 0.4
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
                engagement_score = min(1.0, (like_ratio * 5) * 0.6 + (comment_ratio * 20) * 0.4)

            # 5. Actualidad (10%)
            recency_score = 1.0 if days_since_published <= 365 else (0.5 if days_since_published <= 730 else 0.2)

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
            print(f"Error calculando score del video: {e}")
            return 0.0
