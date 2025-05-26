
FROM python:3.10-slim


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


COPY . .

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["python", "-m", "src.main"]
