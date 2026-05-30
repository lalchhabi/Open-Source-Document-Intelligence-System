# 1. Lightweight base image
FROM python:3.11-slim

# 2. Prevent Python buffering logs
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 3. Set working directory
WORKDIR /app

# 4. Install system dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy requirements 
COPY requirements.txt .
COPY requirements_ml.txt .

# 6. Upgrade pip + install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -r requirements_ml.txt

# 7. Copy application code
COPY . .

# 8. Expose Flask port
EXPOSE 5000

# 9. Run application
CMD ["python", "app.py"]