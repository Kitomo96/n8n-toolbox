# Use a modern Python base image based on Debian Bookworm
FROM python:3.11-slim-bookworm

# Argument for the application version, passed from docker-compose.yml
ARG APP_VERSION=3.0.0
ENV APP_VERSION=${APP_VERSION}

WORKDIR /app

# Prevent apt-get from asking interactive questions
ENV DEBIAN_FRONTEND=noninteractive

# Install all necessary system-level dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    exiftool \
    git \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-spa \
    tesseract-ocr-fra \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the TESSDATA_PREFIX environment variable so Tesseract can find the language packs
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install the browser binaries required by Playwright (for crawl4ai)
RUN playwright install chromium

# Copy the application code into the container
COPY main.py .

# Expose the port the app will run on internally
EXPOSE 8000

# The command to run when the container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]