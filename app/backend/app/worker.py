import asyncio
import os
import sys
import shutil
import argparse
import multiprocessing
import httpx
import json
import psycopg2
from psycopg2.extras import Json
from pathlib import Path
from dotenv import load_dotenv

# Try to load .env from project root (../../../../.env) if it exists, or from the current directory
PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
root_env = PROJECT_ROOT_DIR / ".env"

if root_env.exists():
    load_dotenv(dotenv_path=root_env)
else:
    load_dotenv()

from concurrent.futures import ProcessPoolExecutor
from sqlalchemy import text
from app.db import AsyncSessionLocal
from contextlib import redirect_stdout, redirect_stderr

# API Keys from .env
GEMINI_API_KEY_1 = os.getenv("GOOGLE_API_KEY_1")
GEMINI_API_KEY_2 = os.getenv("GOOGLE_API_KEY_2")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

DATABASE_URL = os.getenv("DATABASE_URL")
# Create sync connection URL (e.g. postgresql+asyncpg://... -> postgresql://...)
DATABASE_URL_SYNC = DATABASE_URL.replace("+asyncpg", "") if DATABASE_URL else ""

# Cross-platform path handling for local_storage
# In Docker, it will use /pipeline/local_storage if mounted, otherwise project_root/local_storage
PROJECT_ROOT = Path("/pipeline") if Path("/pipeline").exists() else PROJECT_ROOT_DIR
LOCAL_STORAGE_DIR = os.getenv("LOCAL_STORAGE_DIR", str(PROJECT_ROOT / "local_storage"))

def update_job_stage_sync(job_id: str, current_stages: list, current_stage_text: str):
    """Synchronously updates the database to avoid asyncpg thread conflicts."""
    if not DATABASE_URL_SYNC:
        return
    try:
        conn = psycopg2.connect(DATABASE_URL_SYNC)
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE jobs SET pipeline_stages = %s, current_stage = %s WHERE id = %s",
                    (Json(current_stages), current_stage_text, job_id)
                )
        conn.close()
    except Exception as e:
        print(f"--- [Worker Sync DB Error] Failed to update stage: {e} ---", flush=True)

def pipeline_process(job_id, input_path):
    # 파이프라인 루트(graphLec)를 절대 경로로 계산
    # Docker: /app/app/worker.py → PIPELINE_ROOT=/app (env로 명시)
    # Local:  graphLec/app/backend/app/worker.py → 4단계 상위
    current_file = Path(__file__).resolve()
    pipeline_path = os.getenv("PIPELINE_ROOT", str(current_file.parent.parent.parent.parent))

    print(f"--- [Child Process {job_id}] Setting sys.path to: {pipeline_path} ---", flush=True)
    
    if pipeline_path not in sys.path:
        sys.path.insert(0, pipeline_path)
    
    os.chdir(pipeline_path)
        
    # Injected from environment variables
    os.environ["GOOGLE_API_KEY_1"] = GEMINI_API_KEY_1 or ""
    os.environ["GOOGLE_API_KEY_2"] = GEMINI_API_KEY_2 or ""
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY or ""

    try:
        # Resolve absolute path for video_path if it's relative
        video_path = Path(input_path)
        if not video_path.is_absolute():
            video_path = Path(pipeline_path) / input_path
            
        print(f"--- [Child Process {job_id}] Target video: {video_path} ---", flush=True)

        output_dir = Path(LOCAL_STORAGE_DIR) / "results" / job_id
        slides_dir = output_dir / "slides"
        log_file_path = output_dir / "pipeline.log"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        slides_dir.mkdir(parents=True, exist_ok=True)
        
        import main as pipeline
        args = pipeline.get_parser().parse_args([
            "--input", str(video_path),
            "--output", str(output_dir),
            "--slides", str(slides_dir),
            "--metadata-dir", str(output_dir / "metadata"),
            "--lance-root", str(output_dir / "lancedb")
        ])
        
        stages_state = {
            "stt": "wait", "voice": "wait", "scene": "wait", 
            "integrate": "wait", "summarize": "wait"
        }
        
        def on_progress(stage_key, status):
            if stage_key in stages_state:
                stages_state[stage_key] = status
            
            stages_array = [{"stage": k, "status": v} for k, v in stages_state.items()]
            stage_text = f"Processing {stage_key}..." if status == "run" else f"Finished {stage_key}"
            
            update_job_stage_sync(job_id, stages_array, stage_text)
            print(f"[{job_id}] Progress updated: {stage_key} -> {status}", flush=True)

        on_progress("init", "wait")

        print(f"--- [Child Process {job_id}] Importing pipeline main... ---", flush=True)
        with open(log_file_path, "w", encoding="utf-8") as log_file:
            with redirect_stdout(log_file), redirect_stderr(log_file):
                try:
                    print(f"[{job_id}] Starting pipeline...")
                    pipeline.run_pipeline(args, progress_callback=on_progress)
                except ImportError as ie:
                    print(f"[{job_id}] ImportError inside child: {ie}. sys.path: {sys.path}")
                    raise
        
        return True, str(output_dir), None
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"--- [Child Process {job_id}] FAILED: {e} ---", flush=True)
        log_file_path = Path(LOCAL_STORAGE_DIR) / "results" / job_id / "pipeline.log"
        if log_file_path.parent.exists():
            with open(log_file_path, "a", encoding="utf-8") as log_file:
                log_file.write(f"\n[{job_id}] Pipeline failed: {error_details}\n")
        return False, None, str(e)

async def worker_loop():
    mp_context = multiprocessing.get_context('spawn')
    executor = ProcessPoolExecutor(mp_context=mp_context)
    
    # [Recovery] Mark orphaned 'running' jobs as 'error' on startup
    async with AsyncSessionLocal() as db:
        result = await db.execute(text("UPDATE jobs SET status = 'error', error_message = 'Worker shutdown during execution' WHERE status = 'running'"))
        count = result.rowcount
        if count > 0:
            print(f"--- [Worker Recovery] Marked {count} orphaned jobs as error. ---", flush=True)
        await db.commit()

    print(f"--- [Worker] Worker loop started. Storage: {LOCAL_STORAGE_DIR} ---", flush=True)
    while True:
        try:
            job_id_val = None
            async with AsyncSessionLocal() as db:
                result = await db.execute(text("""
                    SELECT id, status, input_path FROM jobs WHERE status = 'pending'
                    FOR UPDATE SKIP LOCKED LIMIT 1
                """))
                job = result.mappings().first()
                
                if not job:
                    total_count = await db.execute(text("SELECT count(*) FROM jobs"))
                    count = total_count.scalar()
                    print(f"--- [Worker Heartbeat] Total jobs in DB: {count}. No pending jobs found. ---", flush=True)
                    await asyncio.sleep(5)
                    continue
                
                job_id_val = job['id']
                job_id_str = str(job_id_val)
                job_input_path = job['input_path']

                print(f"--- [Worker] FOUND PENDING JOB: {job_id_str} ---", flush=True)
                await db.execute(text("""
                    UPDATE jobs 
                    SET status = 'running', current_stage = 'Starting pipeline', updated_at = now() 
                    WHERE id = :id
                """), {"id": job_id_val})
                await db.commit()

            # DB 세션을 닫은 뒤 executor 실행 — 장시간 파이프라인 중 세션 타임아웃 방지
            print(f"--- [Worker] Starting pipeline for job: {job_id_str} ---", flush=True)
            loop = asyncio.get_running_loop()
            success, output_dir, error = await loop.run_in_executor(
                executor, pipeline_process, job_id_str, job_input_path
            )
            
            # 완료 후 새 세션으로 최종 상태 업데이트
            print(f"--- [Worker] Pipeline finished for job: {job_id_str}. Success: {success} ---", flush=True)
            async with AsyncSessionLocal() as db:
                if success:
                    await db.execute(text("""
                        UPDATE jobs 
                        SET status = 'done', output_dir = :out_dir, current_stage = 'Finished', updated_at = now() 
                        WHERE id = :id
                    """), {"id": job_id_val, "out_dir": output_dir})
                else:
                    print(f"--- [Worker ERROR] Job {job_id_str} Failed: {error} ---", flush=True)
                    await db.execute(text("""
                        UPDATE jobs 
                        SET status = 'error', error_message = :err, current_stage = 'Failed', updated_at = now()
                        WHERE id = :id
                    """), {"id": job_id_val, "err": error})
                await db.commit()

        except asyncio.CancelledError:
            print("--- [Worker] Loop cancelled by shutdown. ---", flush=True)
            break
        except Exception as e:
            print(f"--- [Worker FATAL ERROR in loop]: {e} ---", flush=True)
            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(worker_loop())