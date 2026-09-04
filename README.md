# Engineering AI Systems — AI Assistant

Production-ready conversational AI assistant with RAG, tool calling, and structured output.

## Architecture

See `ARCHITECTURE.txt` for the full system diagram.

```
Client → Streamlit UI (:8501) → FastAPI Backend (:8000) → LLM (OpenAI / vLLM / Ollama)
                                         │
                                    ChromaDB (RAG)
```

### Reliability Features

| Feature | Implementation |
|---|---|
| **Retry** | Exponential backoff, 3 attempts, capped wait (tenacity-style) |
| **Rate limiting** | 30 req/min per client IP (sliding window) |
| **Circuit breaker** | Opens after 5 failures, 30s recovery window |
| **Fallback provider** | Auto-switches to Ollama/local vLLM on primary failure |
| **Error handling** | Structured JSON errors, middleware catch-all |
| **Response caching** | LRU with TTL (256 entries, 5min) |
| **Background processing** | Ingestion runs in background tasks |
| **Request tracking** | X-Request-ID + X-Response-Time headers |

## Quick Start

### Prerequisites

- Docker & Docker Compose
- An OpenAI API key (or local Ollama instance)

### 1. Configure environment

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 2. Run with Docker Compose

```bash
# Full stack (backend + UI)
docker compose up --build -d

# With local Ollama fallback (uncomment ollama service in docker-compose.yml first)
docker compose --profile ollama up --build -d
```

### 3. Access the application

- **UI:** http://localhost:8501
- **API docs:** http://localhost:8000/docs
- **Health check:** http://localhost:8000/health

### 4. Ingest documents

Place PDF/TXT/MD files in `data/docs/`, then:

```bash
# Via UI sidebar: click "Re-ingest documents"
# Or via API:
curl -X POST http://localhost:8000/ingest
```

## Local Development (without Docker)

```bash
# Backend
python -m venv myenv && source myenv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # edit with your keys
uvicorn app.main:app --reload

# UI (separate terminal)
pip install -r requirements-ui.txt
BACKEND_URL=http://localhost:8000 streamlit run ui/app.py
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check with circuit breaker + cache stats |
| `POST` | `/chat` | Plain LLM chat |
| `POST` | `/chat/tools` | Chat with tool calling (calculator, KB query) |
| `POST` | `/chat/rag` | RAG-augmented chat with sources + confidence |
| `POST` | `/ingest` | Background document ingestion |
| `GET` | `/cache/stats` | Cache hit/miss statistics |
| `POST` | `/cache/invalidate` | Clear response cache |

## Configuration

All settings are via environment variables (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai`, `vllm`, or `ollama` |
| `OPENAI_API_KEY` | — | Required for OpenAI provider |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model name |
| `MAX_RETRIES` | `3` | Retry attempts per LLM call |
| `RATE_LIMIT_PER_MINUTE` | `30` | Max requests per client IP |
| `CACHE_MAX_SIZE` | `256` | Max cached responses |
| `CACHE_TTL_SECONDS` | `300` | Cache entry lifetime |
| `FALLBACK_PROVIDER` | — | Secondary provider (e.g., `ollama`) |

## Cloud Deployment

### Azure Container Apps

```bash
az containerapp up \
  --name ai-assistant \
  --resource-group myResourceGroup \
  --image <your-registry>/ai-assistant:latest \
  --target-port 8000 \
  --ingress external \
  --env-vars "OPENAI_API_KEY=<key>" "LLM_PROVIDER=openai"
```

### AWS ECS (Fargate)

1. Push image to ECR
2. Create ECS cluster + task definition with the container
3. Set environment variables in the task definition
4. Attach an Application Load Balancer

### Google Cloud Run

```bash
gcloud run deploy ai-assistant \
  --image <your-registry>/ai-assistant:latest \
  --port 8000 \
  --set-env-vars "OPENAI_API_KEY=<key>" \
  --allow-unauthenticated
```

## Project Structure

```
├── app/
│   ├── main.py              # FastAPI app + middleware + endpoints
│   ├── config.py            # Environment-based settings
│   ├── llm_client.py        # LLM client with retry/fallback/cache
│   ├── schemas.py           # Pydantic request/response models
│   ├── tools.py             # Tool definitions + execution
│   ├── cache.py             # LRU response cache with TTL
│   ├── circuit_breaker.py   # Circuit breaker pattern
│   ├── middleware.py         # Rate limiting, request tracking, error handling
│   └── rag/
│       ├── vectorstore.py   # ChromaDB wrapper
│       ├── retriever.py     # Query + format context
│       └── ingest.py        # PDF/TXT/MD chunking + ingestion
├── ui/
│   └── app.py               # Streamlit chat interface
├── data/docs/               # Source documents for RAG
├── tests/                   # Test scripts
├── Dockerfile               # Multi-stage: backend + ui
├── docker-compose.yml       # Full stack orchestration
├── requirements.txt         # Backend dependencies
├── requirements-ui.txt      # UI dependencies
├── ARCHITECTURE.txt         # System architecture diagram
└── .env.example             # Configuration template
```
