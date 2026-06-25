# Multi-stage build for MobileSAM + SAM 2 segmentation service
# Supports both GPU (CUDA) and CPU-only inference

FROM python:3.11-slim AS base

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies for OpenCV and image processing
# -o Acquire::Check-Valid-Until=false is used because the system clock (2026)
# might be ahead of the repository metadata, causing "Release file expired" errors.
RUN apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false update && \
    apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download MobileSAM checkpoint
RUN mkdir -p /app/models && \
    wget -q https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt \
    -O /app/models/mobile_sam.pt

# Download SAM 2 checkpoint (sam2.1_hiera_small)
RUN wget -q https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt \
    -O /app/models/sam2_hiera_small.pt

# Copy application code
COPY app/ ./app/

# Environment variables
ENV HOST=0.0.0.0
ENV PORT=8001
ENV REDIS_URL=redis://redis:6379

# Default to SAM 2 Small (set SEGMENTATION_ENGINE=mobilesam for MobileSAM)
ENV SEGMENTATION_ENGINE=sam2
ENV MOBILE_SAM_PATH=/app/models/mobile_sam.pt
ENV SAM2_MODEL_PATH=/app/models/sam2_hiera_small.pt

EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')" || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
