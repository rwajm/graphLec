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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading timeline: {str(e)}")

@router.get("/{job_id}/graph")
async def get_job_graph(job_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(text("""
        SELECT output_dir FROM jobs WHERE id = :id AND status = 'done'
    """), {"id": job_id})
    job = result.mappings().first()
    
    if not job or not job["output_dir"]:
        raise HTTPException(status_code=404, detail="Graph results not found or job not completed")
    
    output_dir = Path(job["output_dir"])
    
    nodes_paths = list(output_dir.glob("*_nodes.parquet"))
    edges_paths = list(output_dir.glob("*_edges.parquet"))
    
    if not nodes_paths:
        raise HTTPException(status_code=404, detail="Graph nodes parquet not found")
        
    nodes_path = nodes_paths[0]
    edges_path = edges_paths[0] if edges_paths else None
    
    try:
        ndf = pd.read_parquet(nodes_path)
        edf = pd.read_parquet(edges_path) if edges_path and edges_path.is_file() else pd.DataFrame(columns=["src_id", "rel_type", "tgt_id", "properties_json"])
        
        nodes_out = []
        for _, row in ndf.iterrows():
            nid = _str_cell(row.get("node_id"))
            if not nid: continue
            label = _str_cell(row.get("label")) or nid
            props = _str_cell(row.get("properties_json")) or "{}"
            
            try:
                obj = json.loads(props) if props.strip() else {}
                ntype = _str_cell(obj.get("type")) if isinstance(obj, dict) else ""
            except Exception:
                ntype = ""
                
            nodes_out.append({
                "id": nid,
                "label": label[:120],
                "title": props[:800],
                "type": ntype or "node",
                "color": _hex_color(label)
            })
            
        seen_ids = {n["id"] for n in nodes_out}
        
        edges_out = []
        for _, row in edf.iterrows():
            src = _str_cell(row.get("src_id"))
            tgt = _str_cell(row.get("tgt_id"))
            rel = _str_cell(row.get("rel_type")) or "related"
            
            if not src or not tgt: continue
            edges_out.append({"from": src, "to": tgt, "label": rel})
            
            for x in (src, tgt):
                if x not in seen_ids:
                    seen_ids.add(x)
                    nodes_out.append({
                        "id": x, "label": x[:80], "title": "", "type": "orphan", "color": "#9ca3af"
                    })
                    
        return {
            "node_count": len(nodes_out),
            "edge_count": len(edges_out),
            "graph": {
                "nodes": nodes_out,
                "edges": edges_out
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading graph: {str(e)}")

import httpx
from pydantic import BaseModel

class QAQuery(BaseModel):
    question: str

@router.post("/{job_id}/qa")
async def ask_qa(job_id: str, query: QAQuery, db: AsyncSession = Depends(get_db)):
    # Get stem (input_path filename without extension)
    result = await db.execute(text("SELECT input_path FROM jobs WHERE id = :id"), {"id": job_id})
    job = result.mappings().first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    input_path = Path(job["input_path"])
    stem = input_path.stem
    
    query_service_url = os.getenv("QUERY_SERVICE_URL", "http://127.0.0.1:8001")
    url = f"{query_service_url.rstrip('/')}/internal/query"
    payload = {"stem": stem, "question": query.question}
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            return r.json()
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Query service unreachable: {str(e)}")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Query service error: {e.response.text}")

