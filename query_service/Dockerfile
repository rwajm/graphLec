FROM python:3.12-slim

WORKDIR /app

# 필요한 시스템 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 파이프라인 종속성 설치 (query_service에서 lance_ingest 등을 참조하므로)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# PYTHONPATH 설정
ENV PYTHONPATH=/pipeline

# 소스 코드 복사
COPY query_service/ ./query_service/
COPY lance_ingest.py .

# 서버 실행 (8001 포트)
CMD ["uvicorn", "query_service.main:app", "--host", "0.0.0.0", "--port", "8001"]
