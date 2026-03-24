FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for lightgbm/xgboost
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and data
COPY src/ src/
COPY Data/ Data/
COPY models/ models/

# Create recommendations directory
RUN mkdir -p models/Recumendations

EXPOSE 8080

CMD uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8080}
