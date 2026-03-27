import json
import logging

import httpx
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from lectures.models import Lecture

from .graph_loader import load_kg_from_parquet

logger = logging.getLogger(__name__)


def query_home(request):
    """강의 선택 + 질문 화면 (브라우저는 Django만 호출)."""
    lectures = Lecture.objects.all()
    return render(request, "query_home.html", {"lectures": lectures})


@require_GET
def api_lectures(request):
    """등록된 강의 목록 (lecture_id, stem, title)."""
    data = [
        {"lecture_id": lec.id, "stem": lec.stem, "title": lec.title or lec.stem}
        for lec in Lecture.objects.all()
    ]
    return JsonResponse({"lectures": data})


@csrf_exempt
@require_POST
def api_query(request):
    """
    POST JSON: { "lecture_id": <int>, "question": "<str>" }
    Django가 lecture_id → stem 변환 후 FastAPI /internal/query 로 프록시한다.
    """
    try:
        body = json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError:
        return JsonResponse({"detail": "Invalid JSON"}, status=400)

    lecture_id = body.get("lecture_id")
    question = (body.get("question") or "").strip()
    if not question:
        return JsonResponse({"detail": "question is required"}, status=400)
    if lecture_id is None:
        return JsonResponse({"detail": "lecture_id is required"}, status=400)

    try:
        lec = Lecture.objects.get(pk=int(lecture_id))
    except (ValueError, Lecture.DoesNotExist):
        return JsonResponse({"detail": "lecture not found"}, status=404)

    url = f"{settings.QUERY_SERVICE_URL.rstrip('/')}/internal/query"
    payload = {"stem": lec.stem, "question": question}

    try:
        with httpx.Client(timeout=settings.QUERY_SERVICE_TIMEOUT) as client:
            r = client.post(url, json=payload)
    except httpx.RequestError as e:
        logger.exception("FastAPI 연결 실패")
        return JsonResponse(
            {"detail": f"query service unreachable: {e}"},
            status=502,
        )

    try:
        data = r.json()
    except json.JSONDecodeError:
        return JsonResponse(
            {"detail": "query service returned non-JSON", "raw": r.text[:500]},
            status=502,
        )

    if r.status_code >= 400:
        return JsonResponse(
            data if isinstance(data, dict) else {"detail": data},
            status=r.status_code,
        )

    return JsonResponse(data)


@csrf_exempt
@require_POST
def api_full_graph(request):
    """
    POST JSON: { "lecture_id": <int> }
    output 디렉터리의 {stem}_nodes.parquet / {stem}_edges.parquet 를 읽어 전체 KG를 반환한다.
    """
    try:
        body = json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError:
        return JsonResponse({"detail": "Invalid JSON"}, status=400)

    lecture_id = body.get("lecture_id")
    if lecture_id is None:
        return JsonResponse({"detail": "lecture_id is required"}, status=400)

    try:
        lec = Lecture.objects.get(pk=int(lecture_id))
    except (ValueError, Lecture.DoesNotExist):
        return JsonResponse({"detail": "lecture not found"}, status=404)

    try:
        payload = load_kg_from_parquet(settings.GRAPHLEC_OUTPUT_DIR, lec.stem)
    except ValueError as e:
        return JsonResponse({"detail": str(e)}, status=400)
    except FileNotFoundError as e:
        return JsonResponse({"detail": str(e)}, status=404)
    except Exception as e:
        logger.exception("전체 그래프 Parquet 로드 실패")
        return JsonResponse({"detail": f"parquet 로드 실패: {e}"}, status=500)

    return JsonResponse(
        {
            "stem": payload["stem"],
            "node_count": payload["node_count"],
            "edge_count": payload["edge_count"],
            "graph": {
                "nodes": payload["nodes"],
                "edges": payload["edges"],
            },
        }
    )
