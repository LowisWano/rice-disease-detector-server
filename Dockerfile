FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies needed for OpenCV and others
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    wget \
  && rm -rf /var/lib/apt/lists/*

# Copy only requirements.txt first to leverage Docker layer caching
COPY requirements.txt ./

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create a non-root user and change ownership of app directory
RUN useradd -m appuser \
  && chown -R appuser /app

# Switch to non-root user
USER appuser

# Expose port 10000 as expected by Render
EXPOSE 10000

# Use the PORT environment variable set by Render at runtime
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}