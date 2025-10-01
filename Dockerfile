FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps (curl for healthcheck; build-essential for any light builds)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
 && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision --no-cache-dir

COPY main.py .

EXPOSE 8000

# Run the API; Railway sets $PORT
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 2"]