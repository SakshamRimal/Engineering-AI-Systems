from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

from app.config import settings
from app.llm_client import llm_client
from app.schemas import ChatRequest, ChatResponse, AssistantAnswer
from app.rag.retriever import retrieve, format_context
from app.rag import ingest as ingest_module

app = FastAPI(
    title="AI Assistant API",
    description="LLM assistant with RAG, tool calling, and structured output.",
    version="1.0.0",
)


@app.get("/health")
def health_check():
    return {"status": "ok", "provider": settings.LLM_PROVIDER}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        history_dicts = [h.model_dump() for h in request.history] if request.history else None
        answer = await llm_client.chat(request.message, history=history_dicts)
        return ChatResponse(answer=answer, sources=[], confidence=1.0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {e}")


@app.post("/chat/tools")
async def chat_with_tools(request: ChatRequest):
    try:
        history_dicts = [h.model_dump() for h in request.history] if request.history else None
        result = await llm_client.chat_with_tools(request.message, history=history_dicts)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tool-calling request failed: {e}")


@app.post("/chat/rag", response_model=ChatResponse)
async def chat_rag(request: ChatRequest):
    try:
        chunks = retrieve(request.message, top_k=4)
        context = format_context(chunks)

        raw = await llm_client.chat_structured(request.message, context=context)
        validated = AssistantAnswer(**raw)

        return ChatResponse(
            answer=validated.answer,
            sources=validated.sources,
            confidence=validated.confidence,
        )
    except ValidationError as e:
        raise HTTPException(status_code=502, detail=f"Model returned invalid structured output: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG request failed: {e}")


@app.post("/ingest")
def trigger_ingest():
    # Kept synchronous deliberately — ingestion is CPU-bound (embedding computation),
    # not I/O-bound, so async offers no benefit here; FastAPI runs it in a threadpool automatically.
    try:
        ingest_module.ingest_all()
        return {"status": "ingestion complete"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.APP_HOST, port=settings.APP_PORT, reload=True)