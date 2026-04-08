FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=/app

COPY query_service/ ./query_service/
COPY lance_ingest.py .
COPY embedding_utils.py .

CMD ["uvicorn", "query_service.main:app", "--host", "0.0.0.0", "--port", "8001"]