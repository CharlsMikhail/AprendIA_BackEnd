class TranscriptionEvaluator:
    """
    [OBSOLETO - MOVIDO A COLAB]
    Esta clase originalmente gestionaba la descarga y evaluación semántica (RAG) 
    de las transcripciones de YouTube. 
    Actualmente, todo este procesamiento intensivo ha sido migrado al Notebook de Google Colab (Colab #2)
    para liberar recursos de GPU y CPU en este backend.

    Mantenemos el archivo como referencia arquitectónica de dónde residía la lógica.
    """

    def __init__(self):
        pass

    def get_video_transcript(self, video_id, max_minutes=None):
        raise NotImplementedError("Este método ha sido delegado a Colab.")

    def procesar_transcripcion(self, texto):
        raise NotImplementedError("Este método ha sido delegado a Colab.")
