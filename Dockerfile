# Use Python slim base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port 8000
ENV PYTHONPATH=/app

# Expose port 8000 for FastAPI/Uvicorn
EXPOSE 8000

# Run the FastAPI app using module syntax to avoid import errors
CMD ["python", "-m", "src.main"]
