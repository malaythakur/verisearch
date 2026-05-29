# Multi-stage Dockerfile for Verisearch Backend
FROM python:3.11-slim AS production

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry==2.4.1

# Copy dependency files first (for caching)
COPY pyproject.toml poetry.lock* ./

# Install dependencies without virtualenv
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --only main --no-root || true

# Install AI package
RUN pip install --no-cache-dir openai

# Copy application code
COPY . /app/

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Use shell form so $PORT is expanded at runtime
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh
CMD ["/app/start.sh"]
