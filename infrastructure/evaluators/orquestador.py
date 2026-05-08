import os
import time
import httpx
import asyncio
import logging
from collections import defaultdict

class WhisperOrchestratorClient:
    """
    Cliente orquestador para distribuir transcripciones de YouTube 
    a través de múltiples GPUs (Workers) mediante URLs de ngrok.
    """
    def __init__(self):
        # Leer URLs desde el archivo .env, separadas por coma
        urls_raw = os.getenv("WHISPER_WORKERS", "")
        self.worker_urls = [url.strip().rstrip('/') for url in urls_raw.split(",") if url.strip()]
        self.timeout_per_video = int(os.getenv("WHISPER_TIMEOUT", "600"))

    async def _check_healthy_workers(self) -> list[str]:
        """Verifica qué workers están activos respondiendo al endpoint /health."""
        healthy = []
        if not self.worker_urls:
            logging.error("No hay URLs de Whisper configuradas en la variable WHISPER_WORKERS.")
            return healthy

        logging.info(f"Verificando {len(self.worker_urls)} workers configurados...")
        async with httpx.AsyncClient() as client:
            for url in self.worker_urls:
                try:
                    r = await client.get(f"{url}/health", timeout=5.0)
                    r.raise_for_status()
                    data = r.json()
                    logging.info(f"  ✅ Worker activo: {url} (Modelo: {data.get('model', '?')})")
                    healthy.append(url)
                except Exception as e:
                    logging.warning(f"  ❌ Worker inactivo: {url} (Error: {e})")
        return healthy

    async def _transcribe_one(self, client: httpx.AsyncClient, url: str, video_id: str, idx: int, total: int) -> dict:
        """Envía un video a un worker específico y retorna el resultado."""
        logging.info(f"  [{idx:02d}/{total}] ⏳ Transcribiendo {video_id} en worker...")
        t0 = time.time()
        try:
            r = await client.post(
                f"{url}/transcribe",
                json={"video_id": video_id},
                timeout=self.timeout_per_video,
            )
            r.raise_for_status()
            result = r.json()
            elapsed = round(time.time() - t0, 1)
            logging.info(f"  [{idx:02d}/{total}] ✅ {video_id} completado ({elapsed}s)")
            return result
        except Exception as e:
            elapsed = round(time.time() - t0, 1)
            logging.error(f"  [{idx:02d}/{total}] ❌ Error en {video_id}: {e} ({elapsed}s)")
            return {"video_id": video_id, "error": str(e)}

    async def transcribe_videos_async(self, video_ids: list[str]) -> dict:
        """
        Método principal asíncrono.
        Distribuye los videos equitativamente entre los workers saludables.
        """
        if not video_ids:
            return {}

        healthy_workers = await self._check_healthy_workers()
        if not healthy_workers:
            raise RuntimeError("No hay workers de Whisper disponibles (Revisa tu ngrok y tu archivo .env).")

        # Un semáforo por worker — garantiza 1 job activo por GPU simultáneamente
        semaphores = {url: asyncio.Semaphore(1) for url in healthy_workers}
        load = defaultdict(int)
        results = {}
        total = len(video_ids)

        async def process(client, video_id, idx):
            # Elegir el worker con menos carga actual
            url = min(healthy_workers, key=lambda u: load[u])
            load[url] += 1
            async with semaphores[url]:
                res = await self._transcribe_one(client, url, video_id, idx, total)
            load[url] -= 1
            results[video_id] = res

        async with httpx.AsyncClient() as client:
            tasks = [process(client, vid, i+1) for i, vid in enumerate(video_ids)]
            await asyncio.gather(*tasks)

        # Generar métricas
        exitosos = [v for v in results.values() if "error" not in v]
        logging.info(f"Transcripción finalizada: {len(exitosos)}/{total} exitosos.")
        
        return results

    def transcribe_videos_sync(self, video_ids: list[str]) -> dict:
        """
        Wrapper síncrono para ser llamado desde el hilo del pipeline (CoursePipeline).
        Maneja su propio event loop de asyncio en RAM, sin guardar archivos físicos.
        """
        # Como el pipeline corre en un `threading.Thread` separado sin loop de eventos propio,
        # asyncio.run() creará un loop, ejecutará todo, y se cerrará limpiamente.
        try:
            return asyncio.run(self.transcribe_videos_async(video_ids))
        except Exception as e:
            logging.error(f"Error crítico en el orquestador Whisper: {e}")
            raise