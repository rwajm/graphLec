from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from app.db import get_db
import uuid
import os
import shutil
import logging

from pathlib import Path

import json
import asyncio
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from app.db import get_db
import uuid
import os
import shutil
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs")

# Cross-platform path handling for local_storage
# jobs.py is in app/backend/app/routers/ -> 5 levels deep
PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
PROJECT_ROOT = Path("/pipeline") if Path("/pipeline").exists() else PROJECT_ROOT_DIR
LOCAL_STORAGE_DIR = os.getenv("LOCAL_STORAGE_DIR", str(PROJECT_ROOT / "local_storage"))
os.makedirs(LOCAL_STORAGE_DIR, exist_ok=True)

@router.get("/{job_id}/stream")
async def stream_job_status(job_id: str, request: Request, db: AsyncSession = Depends(get_db)):
    """SSE endpoint that polls the database every 1 second for job updates."""
    async def event_generator():
        while True:
            # Check if client closed connection
            if await request.is_disconnected():
                break

            try:
                result = await db.execute(text("""
                    SELECT id, status, current_stage, error_message, pipeline_stages 
                    FROM jobs WHERE id = :id
                """), {"id": job_id})
                job = result.mappings().first()
                
                if not job:
                    yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
                    break

                # Prepare payload
                payload = {
                    "job_id": str(job["id"]),
                    "lecture_status": job["status"],
                    "current_stage": job["current_stage"],
                    "error_message": job["error_message"],
                    "pipeline_stages": job["pipeline_stages"] or []
                }
                
                yield f"data: {json.dumps(payload)}\n\n"
                
                # Stop streaming if the job reached a terminal state
                if job["status"] in ("done", "error"):
                    break
                    
            except Exception as e:
                logger.error(f"SSE polling error for job {job_id}: {e}")
                
            await asyncio.sleep(1)
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.get("")
async def list_jobs(db: AsyncSession = Depends(get_db)):
    result = await db.execute(text("""
        SELECT id, status, input_path, current_stage, error_message, created_at, updated_at
        FROM jobs ORDER BY created_at DESC
    """))
    jobs = result.mappings().all()
    # Serialize UUID and datetime objects safely
    return [
        {
            **job,
            "id": str(job["id"]) if job["id"] else None,
            "created_at": job["created_at"].isoformat() if job["created_at"] else None,
            "updated_at": job["updated_at"].isoformat() if job["updated_at"] else None,
        }
        for job in jobs
    ]

@router.post("")
async def create_job(
    video: UploadFile = File(...),
    title: str = Form(None),
    category: str = Form(None),
    description: str = Form(None),
    db: AsyncSession = Depends(get_db)
):
    print(f"--- [API] Received upload request: {video.filename} ---", flush=True)
    job_id = uuid.uuid4()
    job_id_str = str(job_id)
    input_dir = os.path.join(LOCAL_STORAGE_DIR, "inputs", job_id_str)
    os.makedirs(input_dir, exist_ok=True)
    
    input_path = os.path.join(input_dir, video.filename)
    print(f"--- [API] Saving file to: {input_path} ---", flush=True)
    
    try:
        # Save to local disk
        with open(input_path, "wb") as f:
            shutil.copyfileobj(video.file, f)
        print(f"--- [API] File saved. Inserting into DB... ---", flush=True)
        
        # Save to DB
        await db.execute(text("""
            INSERT INTO jobs (id, input_path, status)
            VALUES (:id, :input_path, 'pending')
        """), {
            "id": job_id, 
            "input_path": input_path
        })
        await db.commit()
        print(f"--- [API] Job {job_id_str} created successfully in DB ---", flush=True)
    except Exception as e:
        print(f"--- [API] ERROR: {str(e)} ---", flush=True)
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    return {"job_id": job_id_str}

@router.get("/{job_id}")
async def get_job_status(job_id: str, request: Request, db: AsyncSession = Depends(get_db)):
    result = await db.execute(text("""
        SELECT id, status, current_stage, error_message, input_path, output_dir, created_at, updated_at
        FROM jobs WHERE id = :id
    """), {"id": job_id})
    job = result.mappings().first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    # Generate video_url using the mounted /files directory
    video_url = None
    if job["input_path"]:
        try:
            # Get relative path from LOCAL_STORAGE_DIR
            input_p = Path(job["input_path"])
            base_p = Path(LOCAL_STORAGE_DIR)
            rel_path = input_p.relative_to(base_p)
            
            # Construct absolute URL using request.base_url (e.g. http://127.0.0.1:8000/)
            video_url = f"{request.base_url}files/{rel_path.as_posix()}"
        except ValueError:
            logger.warning(f"Job {job_id} input_path not under LOCAL_STORAGE_DIR: {job['input_path']}")
        
    return {
        **job,
        "id": str(job["id"]) if job["id"] else None,
        "video_url": video_url,
        "title": Path(job["input_path"]).name if job["input_path"] else "Unknown",
        "created_at": job["created_at"].isoformat() if job["created_at"] else None,
        "updated_at": job["updated_at"].isoformat() if job["updated_at"] else None,
    }

@router.delete("/{job_id}")
async def delete_job(job_id: str, db: AsyncSession = Depends(get_db)):
    """Deletes a job from the database and removes its files from local_storage."""
    try:
        # 1. Get job info to find paths
        result = await db.execute(text("SELECT id, input_path, output_dir FROM jobs WHERE id = :id"), {"id": job_id})
        job = result.mappings().first()
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        # 2. Delete files from local_storage
        # Delete input file's parent directory (which contains only this job's input)
        if job["input_path"]:
            input_dir = Path(job["input_path"]).parent
            if input_dir.exists() and "inputs" in str(input_dir):
                shutil.rmtree(input_dir, ignore_errors=True)
                print(f"--- [API] Deleted input dir: {input_dir} ---")

        # Delete output directory
        if job["output_dir"]:
            output_dir = Path(job["output_dir"])
            if output_dir.exists() and "results" in str(output_dir):
                shutil.rmtree(output_dir, ignore_errors=True)
                print(f"--- [API] Deleted output dir: {output_dir} ---")

        # 3. Delete from DB
        await db.execute(text("DELETE FROM jobs WHERE id = :id"), {"id": job_id})
        await db.commit()
        
        return {"status": "success", "message": f"Job {job_id} deleted"}
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{job_id}/retry")
async def retry_job(job_id: str, db: AsyncSession = Depends(get_db)):
    """Resets an error job back to pending status."""
    try:
        # Check if job exists
        result = await db.execute(text("SELECT id, status FROM jobs WHERE id = :id"), {"id": job_id})
        job = result.mappings().first()
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        # Reset status to pending
        await db.execute(text("""
            UPDATE jobs 
            SET status = 'pending', current_stage = 'Retrying...', error_message = NULL, updated_at = now()
            WHERE id = :id
        """), {"id": job_id})
        await db.commit()
        
        return {"status": "success", "message": f"Job {job_id} set to pending"}
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
