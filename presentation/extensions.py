from flask_socketio import SocketIO

# Inicializamos SocketIO sin pasarle la app aún. 
# Esto permite que otros módulos lo importen sin causar importaciones circulares.
socketio = SocketIO(cors_allowed_origins="*")
