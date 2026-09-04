FROM python:3.12-slim

# System deps needed for some packages (e.g., building certain wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first — leverages Docker layer caching so code changes
# don't force a full reinstall of everything
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the actual application code
COPY app/ ./app/
COPY data/docs/ ./data/docs/

# Ensure the vector store directory exists (will be mounted as a volume at runtime)
RUN mkdir -p /app/data/chroma_db

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]