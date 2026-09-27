FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr libgl1 libglib2.0-0 fonts-dejavu-core && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt "psycopg[binary]>=3.1"
COPY backend backend
COPY web web
COPY data/corpus data/corpus
COPY data/eval data/eval
COPY scripts scripts
WORKDIR /app/backend
EXPOSE 8000
CMD ["uvicorn", "groundai.api:app", "--host", "0.0.0.0", "--port", "8000"]
