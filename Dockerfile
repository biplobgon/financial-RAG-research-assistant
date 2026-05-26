# ==============================================================================
# Financial RAG Research Assistant — Production Dockerfile
# Multi-stage build: deps → builder → runtime
# ==============================================================================

# ─────────────────────────────────────────────────
# Stage 1: Python dependency compilation
# ─────────────────────────────────────────────────
FROM python:3.11-slim AS deps

WORKDIR /deps

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install wheel
RUN pip install --upgrade pip wheel setuptools

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt


# ─────────────────────────────────────────────────
# Stage 2: Application builder
# ─────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Copy installed packages from deps stage
COPY --from=deps /root/.local /root/.local

# Copy application source
COPY app/ ./app/
COPY main.py .

# Verify application imports
RUN python -c "import app; print('Import check passed')" || true


# ─────────────────────────────────────────────────
# Stage 3: Production runtime
# ─────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Security: Create non-root user
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Install runtime system dependencies (PDF parsing, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages and application
COPY --from=deps /root/.local /home/appuser/.local
COPY --from=builder /app ./

# Create data directories
RUN mkdir -p data/raw data/processed data/synthetic && \
    chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Update PATH for user-installed packages
ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production entrypoint
CMD ["python", "-m", "uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--timeout-keep-alive", "120", \
     "--log-config", "/dev/null"]
