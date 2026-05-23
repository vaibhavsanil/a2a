# Use an official Python slim base image for high efficiency
FROM python:3.9-slim

# Set strict production environment defaults
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Install system dependencies if any are needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements first to maximize Docker build layer caching
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all repository code files into /app
COPY . .

# Expose port 8000
EXPOSE 8000

# Start Uvicorn to host the FastAPI application
CMD ["uvicorn", "d27_fast_api_service:app", "--host", "0.0.0.0", "--port", "8000"]
