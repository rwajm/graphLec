import os
import json
import hashlib
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from app.db import get_db

import pandas as pd

router = APIRouter(prefix="/jobs")

def _hex_color(key: str) -> str:
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return f"#{h[:6]}"

def _str_cell(x) -> str:
    if x is None: return ""
    try:
        if pd.isna(x): return ""
    except Exception:
        pass
    return str(x)

@router.get("/debug/list")
async def list_all_jobs_debug(db: AsyncSession = Depends(get_db)):
    result = await db.execute(text("SELECT id, status, input_path, created_at FROM jobs ORDER BY created_at DESC"))
    jobs = result.mappings().all()
    return jobs

@router.get("/{job_id}/results")
async def get_job_results(job_id: str, request: Request, db: AsyncSession = Depends(get_db)):
    result = await db.execute(text("""
        SELECT output_dir FROM jobs WHERE id = :id AND status = 'done'
    """), {"id": job_id})
    job = result.mappings().first()
    
    if not job or not job["output_dir"]:
        raise HTTPException(status_code=404, detail="Results not found or job not completed")
    
    output_dir = Path(job["output_dir"])
    
    if not output_dir.exists() or not output_dir.is_dir():
        raise HTTPException(status_code=404, detail="Result directory not found")
        
    results = []
    for filepath in output_dir.glob("**/*"):
        if filepath.is_file():
            # Get relative path for the URL
            # LOCAL_STORAGE_DIR mapping in main.py is assumed to be /pipeline/local_storage or PROJECT_ROOT/local_storage
            # We need to find the relative path from local_storage
            try:
                # Find 'local_storage' in path and get everything after it
                parts = list(filepath.parts)
                if "local_storage" in parts:
                    idx = parts.index("local_storage")
                    rel_path = Path(*parts[idx+1:])
                    file_url = f"{request.base_url}files/{rel_path.as_posix()}"
                    results.append({
                        "name": filepath.name,
                        "url": file_url
                    })
            except Exception:
                continue
        
    return results

@router.get("/{job_id}/timeline")
async def get_job_timeline(job_id: str, request: Request, db: AsyncSession = Depends(get_db)):
    result = await db.execute(text("""
        SELECT output_dir FROM jobs WHERE id = :id AND status = 'done'
    """), {"id": job_id})
    job = result.mappings().first()
    
    if not job or not job["output_dir"]:
        raise HTTPException(status_code=404, detail="Timeline results not found or job not completed")
    
    output_dir = Path(job["output_dir"])
    
    # Find *_slide_classified.json
    json_files = list(output_dir.glob("*_slide_classified.json"))
    if not json_files:
        raise HTTPException(status_code=404, detail="slide_classified.json not found")
    
    json_path = json_files[0]
    
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        scenes = []
        for s in data.get("slides", []):
            # Convert absolute image_path to URL
            img_url = None
            if "image_path" in s:
                img_path = Path(s["image_path"])
                parts = list(img_path.parts)
                if "local_storage" in parts:
                    idx = parts.index("local_storage")
                    rel_img_path = Path(*parts[idx+1:])
                    img_url = f"{request.base_url}files/{rel_img_path.as_posix()}"
            
            # Map roles to types for SceneList.jsx
            # core/elaborated -> slide, transitional -> (optional), silent_new -> (optional)
            scene_type = "slide"
            if s.get("role") == "elaborated":
                scene_type = "emphasis" # Or keep as slide
            
            # Formatted timestamp from 00:00:03.07 to 00:03 (frontend expects shorter often, but let's send what we have)
            ts = s.get("timestamp_formatted", "00:00")
            if "." in ts:
                ts = ts.split(".")[0] # Remove milliseconds
            if ts.startswith("00:"): # Remove leading hours if zero
                ts = ts[3:]
                
            scenes.append({
                "timestamp": ts,
                "type": scene_type,
                "text": s.get("title") or f"Slide {s.get('slide_number')}",
                "image_url": img_url,
                "slide_number": s.get("slide_number")
            })
            
        return scenes
