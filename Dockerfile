FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies for OpenCV and common libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    wget \
  && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for better caching
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Run as a non-root user
RUN useradd -m appuser
USER appuser

EXPOSE 8000
ENV PORT=8000

# Start the FastAPI app
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}


