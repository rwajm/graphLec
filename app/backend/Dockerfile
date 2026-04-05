FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 파이프라인 및 백엔드 전용 요구사항 파일 복사
COPY requirements.txt ./pipeline_reqs.txt
COPY app/backend/requirements.txt ./backend_reqs.txt

# 종속성 모두 설치
RUN pip install --no-cache-dir -r pipeline_reqs.txt -r backend_reqs.txt

# 파이프라인 모듈 경로 설정
ENV PYTHONPATH=/pipeline

# 백엔드 코드 복사 (context가 root이므로 상대 경로로 지정)
COPY app/backend/ .

# FastAPI 서버 실행 (PYTHONPATH가 설정되어 있으므로 app.main으로 직접 실행 가능)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
