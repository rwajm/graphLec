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
                    file_url = f"/files/{rel_path.as_posix()}"
                    results.append({
                        "name": filepath.name,
                        "url": file_url
                    })
            except Exception:
                continue
        
    return results

def _format_seconds(sec: float) -> str:
    """초를 MM:SS 형식의 문자열로 변환합니다."""
    if sec is None:
        return "00:00"
    try:
        s = int(float(sec))
        minutes, seconds = divmod(s, 60)
        return f"{minutes:02d}:{seconds:02d}"
    except (ValueError, TypeError):
        return "00:00"

@router.get("/{job_id}/timeline")
async def get_job_timeline(job_id: str, request: Request, db: AsyncSession = Depends(get_db)):
    result = await db.execute(text("""
        SELECT output_dir FROM jobs WHERE id = :id AND status = 'done'
    """), {"id": job_id})
    job = result.mappings().first()
    
    if not job or not job["output_dir"]:
        raise HTTPException(status_code=404, detail="Timeline results not found or job not completed")
    
    output_dir = Path(job["output_dir"])
    
    fused_files = list(output_dir.glob("*_fused.json"))
    if not fused_files:
        raise HTTPException(status_code=404, detail="Fused result file not found (*_fused.json)")
    fused_path = fused_files[0]

    # --- 썸네일 보충을 위해 slide_classified.json 경로 확보 ---
    classified_files = list(output_dir.glob("*_slide_classified.json"))
    classified_path = classified_files[0] if classified_files else None

    try:
        with open(fused_path, "r", encoding="utf-8") as f:
            fused_data = json.load(f)
        
        # --- 썸네일 경로 맵 생성 ---
        image_path_map = {}
        if classified_path and classified_path.exists():
            with open(classified_path, "r", encoding="utf-8") as f:
                classified_data = json.load(f)
            for s in classified_data.get("slides", []):
                if "slide_number" in s and "image_path" in s:
                    image_path_map[s["slide_number"]] = s["image_path"]

        slides = fused_data.get("slides", [])
        if not slides:
            return []

        scores = []
        for s in slides:
            score = s.get("emphasis_score", {}).get("total", 0) or s.get("emphasis_total", 0)
            if score > 0:
                scores.append(score)
        emphasis_threshold = (sum(scores) / len(scores)) if scores else 0

        scenes = []
        for s in slides:
            # --- 썸네일 경로 보충 ---
            slide_num = s.get("slide_number")
            image_path = image_path_map.get(slide_num)
            
            img_url = None
            if image_path:
                img_p = Path(image_path)
                p_str = img_p.as_posix()
                if "local_storage/" in p_str:
                    rel_p = p_str.split("local_storage/")[1]
                    img_url = f"/files/{rel_p}"
            
            start_sec = s.get("start_sec") or s.get("start")
            ts = _format_seconds(start_sec)
                
            final_score = s.get("emphasis_score", {}).get("total", 0) or s.get("emphasis_total", 0)
            is_emphasis = final_score > emphasis_threshold if emphasis_threshold > 0 else False

            scenes.append({
                "timestamp": ts,
                "type": "emphasis" if is_emphasis else "slide",
                "text": s.get("title") or s.get("slide_text", "")[:30] or f"Slide {s.get('slide_number')}",
                "image_url": img_url,
                "slide_number": slide_num,
                "start": start_sec or 0
            })
            
        return sorted(scenes, key=lambda x: x["start"])
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
    
    # 한글 파일명 대응을 위해 더 유연하게 찾기
    nodes_paths = [p for p in output_dir.glob("*.parquet") if "nodes" in p.name.lower()]
    edges_paths = [p for p in output_dir.glob("*.parquet") if "edges" in p.name.lower()]
    
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

