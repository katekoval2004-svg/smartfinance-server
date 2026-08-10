FROM python:3.11-slim

# Системні бібліотеки для OpenCV / ultralytics (без них — libxcb.so.1 error)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libxcb1 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Railway передає $PORT; 1 воркер + більший таймаут (модель + Claude)
CMD gunicorn server:app --bind 0.0.0.0:${PORT:-8080} --workers 1 --timeout 120
