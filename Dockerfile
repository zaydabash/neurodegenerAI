# Production-grade Consolidated Dockerfile
FROM python:3.11-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy all project files to satisfy shared deps and migration
COPY . .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/reports

# --- API Stage ---
FROM base as api
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/app"
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://127.0.0.1:8000/health || exit 1
CMD ["python", "-m", "uvicorn", "core_api.src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# --- UI Stage ---
FROM base as ui
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/app"
EXPOSE 8501
CMD ["streamlit", "run", "dashboard_app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
