class TrustIndexAnalyzer:
    """
    [OBSOLETO - MOVIDO A COLAB]
    Esta clase originalmente gestionaba el análisis de sentimientos de comentarios de YouTube
    utilizando Transformers (HuggingFace) para generar el Índice de Confianza.
    
    Actualmente, este procesamiento en GPU ha sido delegado al Notebook de Google Colab (Colab #2).
    
    Mantenemos el archivo como referencia arquitectónica de dónde residía la lógica.
    """
    def __init__(self):
        pass

    def analyze_comments(self, comments: list) -> float:
        raise NotImplementedError("Este método ha sido delegado a Colab.")
