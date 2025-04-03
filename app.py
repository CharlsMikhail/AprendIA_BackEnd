from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from googleapiclient.discovery import build
import deepseek
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)  # Permite acceso desde el frontend

# Cargar claves de API desde .env
load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Inicializar DeepSeek
deepseek.api_key = DEEPSEEK_API_KEY

# Ruta para interpretar el chat y generar temas de aprendizaje
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    mensaje_usuario = data.get("mensaje")

    prompt = f"El usuario quiere aprender sobre: {mensaje_usuario}. Sugiere los temas clave que debe aprender en un curso estructurado."
    respuesta = deepseek.Chat.completion(model="deepseek-chat", messages=[{"role": "user", "content": prompt}])

    temas = respuesta["choices"][0]["message"]["content"].split("\n")

    return jsonify({"temas": temas})


# Ruta para buscar videos en YouTube
@app.route("/buscar_videos", methods=["GET"])
def buscar_videos():
    query = request.args.get("query")
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

    request_youtube = youtube.search().list(
        part="snippet",
        q=query,
        type="video",
        videoLicense="creativeCommon",  # Solo videos con licencia libre
        maxResults=5
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
