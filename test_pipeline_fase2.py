import sys
sys.stdout.reconfigure(encoding='utf-8')
import time
import json
from dotenv import load_dotenv
load_dotenv()

from application.course_pipeline import CoursePipeline
from infrastructure.db.course_job_repository import CourseJobRepository

def run_test():
    print("Iniciando prueba del pipeline real (sin mocks)...")
    
    # 1. Instanciar dependencias
    pipeline = CoursePipeline()
    repo = CourseJobRepository()
    
    # 2. Iniciar el pipeline
    prompt = "Quiero un curso para aprender a tocar guitarra desde cero"
    print(f"Enviando prompt: '{prompt}'")
    
    job_id = pipeline.iniciar_generacion(prompt=prompt, user_id=None)
    print(f"Job ID generado y guardado en DB: {job_id}")
    
    # 3. Esperar y monitorear la DB
    print("\nMonitoreando persistencia en PostgreSQL...")
    max_retries = 450 # Esperar hasta 15 minutos (15 * 60 / 2) para dar tiempo a Whisper y RAG
    
    for i in range(max_retries):
        job = repo.get_job(job_id)
        
        if not job:
            print("ERROR: El job no se encuentra en la base de datos.")
            return
            
        print(f"[{i*2}s] Status actual en DB: {job.status}")
        
        # --- Volcar información en tiempo real a JSON ---
        import copy
        dump_data = {
            "status": job.status,
            "error_message": job.error_message,
            "prompt_original": None,
            "prompt_refinado": None,
            "course_outline": None,
            "sections": []
        }
        
        if job.course_outline:
            # Extraer los campos de debug si existen
            outline_copy = copy.deepcopy(job.course_outline)
            dump_data["prompt_original"] = outline_copy.pop("_debug_prompt_original", job.prompt)
            dump_data["prompt_refinado"] = outline_copy.pop("_debug_prompt_refinado", None)
            dump_data["course_outline"] = outline_copy
        
        if job.sections_with_candidates:
            clean_sections = copy.deepcopy(job.sections_with_candidates)
            for s in clean_sections:
                for c in s.get("candidates", []):
                    # Eliminar la transcripción gigante, solo dejar un flag
                    if "transcript" in c:
                        c["has_transcript"] = bool(c["transcript"])
                        del c["transcript"]
            dump_data["sections"] = clean_sections
            
        with open("debug_pipeline_fase3.json", "w", encoding="utf-8") as f:
            json.dump(dump_data, f, indent=2, ensure_ascii=False)

        if job.status in ["completed", "failed"]:
            print("\n" + "="*50)
            print(f"EXITO: Pipeline finalizado con status: {job.status}")
            
            if job.status == "completed":
                print(f"\nEsquema generado:\n{json.dumps(job.course_outline, indent=2, ensure_ascii=False)[:300]}...\n")
                
                print(f"Candidatos finales guardados:\n")
                for section in job.sections_with_candidates:
                    print(f"- Sección: {section['title']}")
                    if section['candidates']:
                        print(f"  Mejor video: {section['candidates'][0]['title']} (Score: {section['candidates'][0].get('score', 0):.2f})")
                        print(f"  Sentimiento (% útil): {section['candidates'][0].get('sentiment', {}).get('porcentaje_utiles', 'N/A')}")
                    else:
                        print("  Sin videos")
                        
                print("\nComprobación de persistencia: ÉXITO. Todos los datos están en Neon.")
            else:
                print(f"Mensaje de error: {job.error_message}")
                
            break
            
        time.sleep(2)
        
    else:
        print("\n⏳ Tiempo de espera agotado.")

if __name__ == "__main__":
    run_test()
