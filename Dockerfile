
# Base Image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies because of C/C++ compilation
# (important for ML / FAISS / numpy builds)

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# Copy project files
COPY . .

# Environment variables (optional default placeholders)
# print logs immediately (without buffering)
ENV PYTHONUNBUFFERED=1

# Expose Flask port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]