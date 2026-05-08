import os
import jwt
import logging
from datetime import datetime, timedelta, timezone
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from infrastructure.db.user_repository import UserRepository

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "aprendia_super_secret_key_change_in_production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", "72"))


class AuthService:
    def __init__(self):
        self.user_repo = UserRepository()

    def generate_jwt(self, user_id: int) -> str:
        """Genera un JWT firmado con el user_id del usuario."""
        payload = {
            "user_id": user_id,
            "iat": datetime.now(timezone.utc),
            "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRATION_HOURS)
        }
        return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

    def login_with_google_token(self, google_id_token: str) -> dict:
        """
        Recibe un id_token de Google (desde el frontend con Google Sign-In),
        lo valida, hace upsert del usuario en la DB, y devuelve un JWT de AprendIA.
        """
        if not GOOGLE_CLIENT_ID:
            raise ValueError("GOOGLE_CLIENT_ID no está configurado en el .env")

        # Validar el id_token con los servidores de Google
        try:
            idinfo = id_token.verify_oauth2_token(
                google_id_token,
                google_requests.Request(),
                GOOGLE_CLIENT_ID
            )
        except ValueError as e:
            logging.warning(f"Google id_token inválido: {e}")
            raise ValueError("Token de Google inválido o expirado")

        google_id = idinfo["sub"]
        email = idinfo.get("email", "")
        name = idinfo.get("given_name", email.split("@")[0])
        picture = idinfo.get("picture", "")
        needs_onboarding = False

        # Buscar usuario existente o crear uno nuevo
        user = self.user_repo.get_user_by_google_id(google_id)
        if not user:
            user = self.user_repo.create_user(google_id, email, name, picture)
            needs_onboarding = True

        access_token = self.generate_jwt(user.id_usuario if hasattr(user, 'id_usuario') else user.id)

        return {
            "access_token": access_token,
            "token_type": "Bearer",
            "expires_in": JWT_EXPIRATION_HOURS * 3600,
            "needs_onboarding": needs_onboarding,
            "user": {
                "id_usuario": user.id_usuario if hasattr(user, 'id_usuario') else user.id,
                "nombre_usuario": user.nombre_usuario if hasattr(user, 'nombre_usuario') else user.name,
                "correo": user.correo if hasattr(user, 'correo') else user.email,
                "foto_perfil_url": user.foto_perfil_url if hasattr(user, 'foto_perfil_url') else user.picture
            }
        }
