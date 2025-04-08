from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import re
import random
import time
from datetime import datetime
from googleapiclient.discovery import build
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Environment variables
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

# Configure Azure OpenAI client
azure_client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2024-12-01-preview"
)


def get_course_outline(prompt):
    """Use Azure OpenAI to generate a course outline based on the prompt"""
    try:
        system_message = """
        Crea un plan de curso estructurado sobre el tema proporcionado. 
        Incluye:
        1. Un título para el curso
        2. Una introducción breve que explique el propósito del curso
        3. Entre 4-6 secciones (subtemas) principales, cada una con:
           - Título descriptivo
           - Descripción breve del contenido (2-3 oraciones)
        4. Una lista de 4 objetivos de aprendizaje
        5. Una lista de 3 requisitos previos

        Proporciona la respuesta en formato JSON con la siguiente estructura:
        {
            "title": "Título del curso",
            "introduction": "Introducción del curso",
            "sections": [
                {
                    "id": 0,
                    "title": "Título de la primera sección",
                    "description": "Descripción de la primera sección"
                },
                ...
            ],
            "learningOutcomes": ["Objetivo 1", "Objetivo 2", "Objetivo 3", "Objetivo 4"],
            "requirements": ["Requisito 1", "Requisito 2", "Requisito 3"],
            "searchQueries": {
                "introductory": "Términos de búsqueda para video introductorio",
                "section0": "Términos de búsqueda específicos para la sección 0",
                "section1": "Términos de búsqueda específicos para la sección 1",
                ...
            }
        }

        Para cada sección y el video introductorio, incluye términos de búsqueda optimizados para YouTube que permitan encontrar videos educativos específicos sobre ese subtema.
        """

        response = azure_client.chat.completions.create(
            model="gpt-4-aprendia",  # Ajusta el nombre del modelo según tu despliegue
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Crea un curso sobre: {prompt}"}
            ],
            temperature=0.7
        )

        course_outline = response.choices[0].message.content

        # Extract JSON if it's surrounded by backticks or other markup
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```|```([\s\S]*?)```|(\{[\s\S]*\})', course_outline)
        if json_match:
            # Use the first matching group that contains content
            for group in json_match.groups():
                if group and '{' in group:
                    course_outline = group
                    break

        # Parse the JSON response
        try:
            outline_data = json.loads(course_outline)
            return outline_data
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            print(f"Raw content: {course_outline}")
            # Attempt to fix common JSON syntax issues
            course_outline = course_outline.replace("'", "\"")
            try:
                outline_data = json.loads(course_outline)
                return outline_data
            except:
                raise Exception("Failed to parse course outline from Azure OpenAI response")

    except Exception as e:
        print(f"Error getting course outline: {e}")
        raise


def search_youtube_videos(query, max_results=5):
    """Search for YouTube videos based on query and return the best match"""
    try:
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

        # First search for videos with Creative Commons license
        request_youtube = youtube.search().list(
            part="snippet",
            q=query,
            type="video",
            videoLicense="creativeCommon",
            maxResults=max_results,
            relevanceLanguage="es",  # Prefer Spanish results
            videoDuration="medium"  # Medium length videos (4-20 minutes)
        )
        response = request_youtube.execute()

        if not response.get("items"):
            # If no Creative Commons videos found, search for any videos
            request_youtube = youtube.search().list(
                part="snippet",
                q=query,
                type="video",
                maxResults=max_results,
                relevanceLanguage="es",
                videoDuration="medium"
            )
            response = request_youtube.execute()

        videos = []

        if response.get("items"):
            # Get video IDs for additional metadata
            video_ids = [item["id"]["videoId"] for item in response["items"]]

            # Get video statistics and content details
            video_details = youtube.videos().list(
                part="statistics,contentDetails,snippet",
                id=",".join(video_ids)
            ).execute()

            # Create a mapping of video ID to its details
            details_map = {item["id"]: item for item in video_details["items"]}

            # Process each video
            for item in response["items"]:
                video_id = item["id"]["videoId"]
                video_details = details_map.get(video_id, {})
                statistics = video_details.get("statistics", {})
                content_details = video_details.get("contentDetails", {})

                # Extract duration in a user-friendly format
                duration = content_details.get("duration", "PT0M0S")
                # Convert ISO 8601 duration to minutes
                minutes_match = re.search(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration)
                if minutes_match:
                    hours = int(minutes_match.group(1) or 0)
                    minutes = int(minutes_match.group(2) or 0)
                    seconds = int(minutes_match.group(3) or 0)
                    duration_str = f"{hours}h {minutes}m" if hours > 0 else f"{minutes} min"
                else:
                    duration_str = "Desconocido"

                videos.append({
                    "title": item["snippet"]["title"],
                    "description": item["snippet"]["description"],
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "videoUrl": f"https://www.youtube.com/embed/{video_id}",
                    "thumbnail": item["snippet"]["thumbnails"]["high"]["url"],
                    "channelTitle": item["snippet"]["channelTitle"],
                    "publishedAt": item["snippet"]["publishedAt"],
                    "views": int(statistics.get("viewCount", 0)),
                    "likes": int(statistics.get("likeCount", 0)),
                    "duration": duration_str
                })

            # Sort videos by a combination of views, likes, and relevance
            for video in videos:
                # Calculate a score based on views and likes
                views_score = min(video["views"] / 1000, 100)  # Cap at 100k views
                likes_score = min(video["likes"] / 100, 100)  # Cap at 10k likes

                # Total score is a weighted combination
                video["score"] = (views_score * 0.6) + (likes_score * 0.4)

            # Return videos sorted by score
            return sorted(videos, key=lambda x: x["score"], reverse=True)

        return []

    except Exception as e:
        print(f"Error searching YouTube videos: {e}")
        return []


@app.route("/solicitar_cursos", methods=["POST"])
def solicitar_cursos():
    """Main endpoint to generate a course based on a prompt"""
    try:
        data = request.json
        prompt = data.get("prompt", "")

        if not prompt:
            return jsonify({"error": "El prompt no puede estar vacío"}), 400

        # Generate course outline using Azure OpenAI
        course_outline = get_course_outline(prompt)

        # Generate a random course ID
        course_id = f"course_{int(time.time())}"

        # Search for introductory video
        intro_query = course_outline.get("searchQueries", {}).get("introductory", f"introducción a {prompt}")
        intro_videos = search_youtube_videos(intro_query, max_results=3)

        preview_video_url = None
        if intro_videos:
            # Use the best video for the course preview
            preview_video_url = intro_videos[0]["videoUrl"]

        # Prepare course sections with videos
        sections = []

        # Add introduction section
        if intro_videos:
            best_intro_video = intro_videos[0]
            sections.append({
                "id": 0,
                "title": "Introducción",
                "content": course_outline["introduction"],
                "videoUrl": best_intro_video["videoUrl"],
                "duration": best_intro_video["duration"],
                "classes": 1
            })

        # Add the rest of the sections
        for i, section in enumerate(course_outline["sections"]):
            section_id = i + 1  # Start from 1 since 0 is intro

            # Get search query for this section
            section_query = course_outline.get("searchQueries", {}).get(f"section{i}", f"{section['title']} {prompt}")

            # Search for videos for this section
            section_videos = search_youtube_videos(section_query, max_results=3)

            if section_videos:
                best_video = section_videos[0]
                sections.append({
                    "id": section_id,
                    "title": section["title"],
                    "content": section["description"],
                    "videoUrl": best_video["videoUrl"],
                    "duration": best_video["duration"],
                    "classes": 1
                })

        # Add final evaluation section
        sections.append({
            "id": len(sections),
            "title": "Evaluación Final",
            "classes": 1
        })

        # Calculate total classes and estimate total duration
        total_classes = len(sections)

        # Calculate total duration in minutes
        total_minutes = 0
        for section in sections:
            if "duration" in section:
                duration_text = section["duration"]
                # Extract hours and minutes
                hours_match = re.search(r'(\d+)h', duration_text)
                minutes_match = re.search(r'(\d+) min|(\d+)m', duration_text)

                hours = int(hours_match.group(1)) if hours_match else 0
                if minutes_match:
                    minutes = int(minutes_match.group(1) or minutes_match.group(2) or 0)
                else:
                    minutes = 0

                total_minutes += (hours * 60) + minutes

        total_duration = f"{total_minutes // 60}h {total_minutes % 60}m" if total_minutes >= 60 else f"{total_minutes}m"

        # Prepare final course data
        current_date = datetime.now().strftime("%m/%Y")
        course_data = {
            "title": course_outline["title"],
            "introduction": course_outline["introduction"],
            "instructor": "IA Professor",
            "rating": round(random.uniform(4.5, 4.9), 1),
            "students": random.randint(5000, 15000),
            "lastUpdated": current_date,
            "language": "Español",
            "totalDuration": total_duration,
            "totalLessons": total_classes,
            "sections": sections,
            "learningOutcomes": course_outline["learningOutcomes"],
            "requirements": course_outline["requirements"],
            "previewVideoUrl": preview_video_url
        }

        return jsonify(course_data)

    except Exception as e:
        print(f"Error generating course: {e}")
        return jsonify({"error": str(e)}), 500


# Add a health check endpoint
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True)