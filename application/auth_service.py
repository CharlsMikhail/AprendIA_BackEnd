import os
from flask import request, jsonify, redirect, session, url_for
from requests_oauthlib import OAuth2Session
from infrastructure.db.user_repository import UserRepository

# Configuración de Google OAuth
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"
OAUTHLIB_INSECURE_TRANSPORT = os.getenv("OAUTHLIB_INSECURE_TRANSPORT", "1")
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = OAUTHLIB_INSECURE_TRANSPORT

class AuthService:
    def __init__(self):
        self.user_repo = UserRepository()

    def get_google_provider_cfg(self):
        import requests
        return requests.get(GOOGLE_DISCOVERY_URL).json()

    def login(self):
        google_provider_cfg = self.get_google_provider_cfg()
        authorization_endpoint = google_provider_cfg["authorization_endpoint"]
        
        # Opciones de scope para obtener email y perfil
        oauth2_session = OAuth2Session(GOOGLE_CLIENT_ID, scope=["openid", "email", "profile"], redirect_uri=request.base_url + "/callback")
        authorization_url, state = oauth2_session.authorization_url(authorization_endpoint)
        
        # Guardar el estado para verificar el callback
        session["state"] = state
        return redirect(authorization_url)

    def callback(self):
        # Verificar estado
        if request.args.get("state") != session.get("state"):
            return jsonify({"error": "Invalid state parameter."}), 400

        google_provider_cfg = self.get_google_provider_cfg()
        token_endpoint = google_provider_cfg["token_endpoint"]

        oauth2_session = OAuth2Session(GOOGLE_CLIENT_ID, state=session["state"], redirect_uri=request.base_url)
        
        # Obtener tokens
        oauth2_session.fetch_token(
            token_endpoint,
            client_secret=GOOGLE_CLIENT_SECRET,
            authorization_response=request.url
        )

        userinfo_endpoint = google_provider_cfg["userinfo_endpoint"]
        userinfo_response = oauth2_session.get(userinfo_endpoint)

        if userinfo_response.json().get("email_verified"):
            unique_id = userinfo_response.json()["sub"]
            users_email = userinfo_response.json()["email"]
            picture = userinfo_response.json()["picture"]
            users_name = userinfo_response.json()["given_name"]

            # Buscar o crear usuario
            user = self.user_repo.get_user_by_google_id(unique_id)
            if not user:
                user = self.user_repo.create_user(unique_id, users_email, users_name, picture)

            # Iniciar sesión local (Session basada en cookies por defecto de Flask)
            session["user_id"] = user.id
            session["user_email"] = user.email
            
            return redirect(url_for("profile")) # Ajustar ruta en `routes.py`
        else:
            return "User email not available or not verified by Google.", 400

    def get_current_user(self):
        user_id = session.get("user_id")
        if not user_id:
            return None
        return user_id

    def logout(self):
        session.clear()
        return redirect(url_for("index"))
