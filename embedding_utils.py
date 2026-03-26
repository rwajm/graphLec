"""
Gemini 임베딩 헬퍼 (문서/질의용 task_type 분리, 429 재시도).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Optional, Sequence, Union

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(Path(__file__).resolve().parent / ".env")

RETRY_DELAYS_SEC = [0, 5, 15, 30]

DEFAULT_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")


def gemini_api_key() -> str:
    key = os.getenv("GOOGLE_API_KEY_2") or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY (또는 GOOGLE_API_KEY_2) 환경 변수가 필요합니다.")
    return key


def get_genai_client() -> genai.Client:
    return genai.Client(api_key=gemini_api_key())


def _embed_with_retry(
    client: genai.Client,
    *,
    model: str,
    contents: Union[Sequence[str], str],
    task_type: str,
) -> List[List[float]]:
    last_err: Optional[Exception] = None
    for delay in RETRY_DELAYS_SEC:
        if delay:
            time.sleep(delay)
        try:
            r = client.models.embed_content(
                model=model,
                contents=contents,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            return [list(e.values) for e in r.embeddings]
        except Exception as e:
            last_err = e
            err_s = str(e)
            if "429" not in err_s and "RESOURCE_EXHAUSTED" not in err_s:
                raise
    assert last_err is not None
    raise last_err


def embed_documents(
    client: genai.Client,
    texts: Sequence[str],
    *,
    model: str = DEFAULT_EMBEDDING_MODEL,
) -> List[List[float]]:
    if not texts:
        return []
    return _embed_with_retry(
        client, model=model, contents=list(texts), task_type="RETRIEVAL_DOCUMENT"
    )


def embed_query(
    client: genai.Client,
    text: str,
    *,
    model: str = DEFAULT_EMBEDDING_MODEL,
) -> List[float]:
    vecs = _embed_with_retry(
        client, model=model, contents=text, task_type="RETRIEVAL_QUERY"
    )
    return vecs[0]
