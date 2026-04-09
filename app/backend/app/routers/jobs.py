from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from neo4j import GraphDatabase, exceptions as neo4j_exceptions
from app.db import get_db
import uuid
import os
import shutil
import logging
import json
import asyncio
import httpx
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
            # 절대 경로에서 local_storage 이후의 경로만 추출 (OS 구분자 무관하게 처리)
            p_str = Path(job["input_path"]).as_posix()
            if "local_storage/" in p_str:
                rel_path = p_str.split("local_storage/")[1]
                video_url = f"/files/{rel_path}"
        except Exception as e:
            logger.warning(f"Error generating video_url for job {job_id}: {e}")
        
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

# ==========================================
# Graph DB (Neo4j) Status & Retry endpoints
# ==========================================

from neo4j import GraphDatabase, exceptions as neo4j_exceptions

def _get_neo4j_driver():
    """도커 환경 및 로컬 환경의 Neo4j 설정 로드"""
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USER")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    if not all([neo4j_uri, neo4j_user, neo4j_password]):
        raise RuntimeError("NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD 환경변수가 필요합니다.")
    return GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

@router.get("/{job_id}/graph_status")
async def check_graph_status(job_id: str, db: AsyncSession = Depends(get_db)):
    """Neo4j에 해당 작업의 그래프 데이터가 존재하는지 확인합니다."""
    # 1. DB에서 job_id의 input_path를 가져와 stem 추출
    result = await db.execute(text("SELECT input_path FROM jobs WHERE id = :id"), {"id": job_id})
    job = result.mappings().first()
    
    if not job or not job["input_path"]:
        raise HTTPException(status_code=404, detail="Job or input file not found")
        
    stem = Path(job["input_path"]).stem
    
    # 2. Neo4j에 해당 stem을 가진 노드가 있는지 쿼리
    node_count = 0
    driver = None
    try:
        driver = _get_neo4j_driver()
        # Verify connection quickly
        driver.verify_connectivity() 
        with driver.session() as session:
            # stem 속성을 가진 노드 개수 카운트
            record = session.run("MATCH (n {stem: $stem}) RETURN count(n) AS count", stem=stem).single()
            if record:
                node_count = record["count"]
                
        is_ingested = node_count > 0
        return {
            "job_id": job_id,
            "stem": stem,
            "graph_exists": is_ingested,
            "node_count": node_count,
            "neo4j_reachable": True
        }
    except Exception as e:
        logger.error(f"Neo4j connection error for job {job_id}: {e}")
        return {
            "job_id": job_id,
            "stem": stem,
            "graph_exists": False,
            "node_count": 0,
            "neo4j_reachable": False,
            "error": str(e)
        }
    finally:
        if driver:
            driver.close()

@router.post("/{job_id}/retry_graph")
async def retry_graph_ingestion(job_id: str, db: AsyncSession = Depends(get_db)):
    """
    분석은 스킵하고 그래프 적재만 재시도합니다.
    기존 워커가 main.py를 재실행하게 만들면, 로컬 스토리지에 파일이 있으므로 
    Stage 1~5는 스킵되고 Neo4j 적재만 실행됩니다.
    """
    try:
        result = await db.execute(text("SELECT id, status, output_dir FROM jobs WHERE id = :id"), {"id": job_id})
        job = result.mappings().first()
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        # 워커가 이 작업을 다시 집어가서 파이프라인(main.py)을 실행하게 함
        # _is_done 로직에 의해 무거운 분석은 자동으로 스킵됨
        await db.execute(text("""
            UPDATE jobs 
            SET status = 'pending', current_stage = 'Retrying Graph Ingestion...', error_message = NULL, updated_at = now()
            WHERE id = :id
        """), {"id": job_id})
        await db.commit()
        
        return {"status": "success", "message": f"Graph ingestion retry triggered for Job {job_id}"}
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# QA (Query Service Integration)
# ==========================================

import httpx

@router.post("/{job_id}/qa")
async def ask_question(job_id: str, request: Request, db: AsyncSession = Depends(get_db)):
    """
    사용자의 질문을 받아 query_service(8001)에 전달하고 답변을 반환합니다.
    """
    body = await request.json()
    question = body.get("question")
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")

    # 1. DB에서 job_id의 input_path를 가져와 stem 추출
    result = await db.execute(text("SELECT input_path FROM jobs WHERE id = :id"), {"id": job_id})
    job = result.mappings().first()
    if not job or not job["input_path"]:
        raise HTTPException(status_code=404, detail="Job not found")
        
    stem = Path(job["input_path"]).stem

    # 2. query_service URL 결정
    # 도커 내부라면 http://query_service:8001, 로컬이라면 http://localhost:8001
    query_service_url = os.getenv("QUERY_SERVICE_URL", "http://localhost:8001")
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{query_service_url}/internal/query",
                json={
                    "stem": stem,
                    "question": question
                }
            )
            
            if resp.status_code != 200:
                logger.error(f"Query service error: {resp.text}")
                raise HTTPException(status_code=resp.status_code, detail="Query service failed to respond")
                
            query_result = resp.json()
            
            # 3. 답변 및 그래프 근거 반환
            return {
                "answer": query_result.get("answer"),
                "timestamps": query_result.get("timestamps", []),
                "graph": query_result.get("graph", {"nodes": [], "edges": []}),
                "retrieved_chunks": query_result.get("retrieved_chunks", [])
            }
            
    except httpx.RequestError as e:
        logger.error(f"Failed to connect to query_service: {e}")
        raise HTTPException(status_code=503, detail="Query service is unreachable")

