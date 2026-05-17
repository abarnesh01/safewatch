# Build Stage
FROM python:3.10-slim AS builder

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Final Stage
FROM python:3.10-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy installed dependencies from builder
COPY --from=builder /root/.local /root/.local

# Ensure local bin is on PATH
ENV PATH=/root/.local/bin:$PATH

# Copy project files
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV SAFEWATCH_ENV=production

# Define volumes for persistent storage
VOLUME ["/app/logs", "/app/snapshots", "/app/recordings", "/app/runtime_cache"]

# Expose Dashboard Port
EXPOSE 8501

# Run the dashboard and main pipeline using a startup script or concurrently
CMD ["python", "main.py"]
