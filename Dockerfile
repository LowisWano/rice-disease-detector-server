# Use slim Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies (CPU-only PyTorch + everything else)
RUN pip install --no-cache-dir \
    torch==2.5.1+cpu torchvision==0.20.1+cpu \
    -f https://download.pytorch.org/whl/cpu \
    fastapi uvicorn timm pillow python-multipart

# Copy your app code and model
COPY . .

# Run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]