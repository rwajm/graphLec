import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from dotenv import load_dotenv

# Try to load .env from project root (../../.env) if it exists, or from the current directory
root_env = Path(__file__).resolve().parent.parent.parent / ".env"
if root_env.exists():
    load_dotenv(dotenv_path=root_env)
else:
    load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.db import init_db
from app.worker import worker_loop
from app.routers import jobs, results

# Cross-platform path handling for local_storage
# main.py is in app/backend/app/ -> 4 levels deep from root (including filename)
PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
PROJECT_ROOT = Path("/pipeline") if Path("/pipeline").exists() else PROJECT_ROOT_DIR
LOCAL_STORAGE_DIR = os.getenv("LOCAL_STORAGE_DIR", str(PROJECT_ROOT / "local_storage"))

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- [FastAPI] Starting lifespan events... ---", flush=True)
    
    # Ensure local storage directory exists
    os.makedirs(LOCAL_STORAGE_DIR, exist_ok=True)
    print(f"--- [FastAPI] Local storage directory ensured: {LOCAL_STORAGE_DIR} ---", flush=True)
    
    try:
        print("--- [FastAPI] Initializing DB... ---", flush=True)
        await init_db()
        print("--- [FastAPI] DB initialized successfully. ---", flush=True)
    except Exception as e:
        print(f"--- [FastAPI] ERROR initializing DB: {e} ---", flush=True)
    
    print("--- [FastAPI] Starting worker loops... ---", flush=True)
    tasks = [asyncio.create_task(worker_loop()) for _ in range(3)]
    print(f"--- [FastAPI] {len(tasks)} worker tasks created. ---", flush=True)
    
    yield
    
    print("--- [FastAPI] Shutting down... Cancelling workers. ---", flush=True)
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    print("--- [FastAPI] Shutdown complete. ---", flush=True)

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite 기본 포트
        "http://127.0.0.1:5173",
        "http://localhost:8000",  # 백엔드 자체
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount local storage for serving result files
app.mount("/files", StaticFiles(directory=LOCAL_STORAGE_DIR), name="files")

app.include_router(jobs.router)
app.include_router(results.router)

@app.get("/health")
def health_check():
    return {"status": "ok"}