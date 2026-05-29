FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry==2.4.1

# Copy backend files
COPY pyproject.toml poetry.lock* ./

RUN poetry config virtualenvs.create false && poetry install --no-interaction --only main --no-root || true

RUN pip install --no-cache-dir openai asyncpg

# Copy backend code into /app/backend/
COPY . /app/backend/

# Set working directory to parent so 'backend.api_gateway' resolves
WORKDIR /app

EXPOSE 8000

CMD ["uvicorn", "backend.api_gateway.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
