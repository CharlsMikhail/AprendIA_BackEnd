from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from googleapiclient.discovery import build

app = Flask(__name__)
CORS(app)  # Permite acceso desde el frontend

# Cargar clave de API de YouTube desde .env
from dotenv import load_dotenv
load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# Ruta para buscar videos
@app.route("/buscar_videos", methods=["GET"])
def buscar_videos():
    query = request.args.get("query")
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

    request_youtube = youtube.search().list(
        part="snippet", q=query, type="video", videoLicense="creativeCommon", maxResults=5
    )
    response = request_youtube.execute()

    videos = [
        {
            "titulo": item["snippet"]["title"],
            "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}",
            "canal": item["snippet"]["channelTitle"],
        }
        for item in response["items"]
    ]

    return jsonify(videos)

if __name__ == "__main__":
    app.run(debug=True)
