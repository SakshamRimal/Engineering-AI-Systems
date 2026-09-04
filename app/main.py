from fastapi import FastAPI , HTTPException 
from pydantic import ValidationError 

from app.config import settings 
from app.llm_client import llm_client 
from app.schemas import ChatRequest , ChatResponse , AssistantAnswer
from app.rag.retriever import retrieve, format_context 
from app.rag import ingest as ingest_module 

app = FastAPI(
    title="AI Assistant API",
    description="An API for an AI assistant that can answer questions based on retrieved context and use tools.",
    version="1.0.0",
)

@app.get("/health")
def health_check():
    """Basic Liveness check -used by docker healthcheck and monitoring """
    return {
        "status": "healthy",
        "llm_provider": settings.LLM_PROVIDER,
        "llm_model": settings.OPENAI_MODEL if settings.LLM_PROVIDER == "openai" else settings.VLLM_MODEL,
    }
    
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Plain chat endpoint.No tools , no forced JSON , just conventional response"""
    try:
        answer = llm_client.chat(request.message , history=request.history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.post("/chat/tools")
def chat_with_tools(request: ChatRequest):
    """
    Tool -calling endpoint - model can invoke calculator , websearch , or query a vector database for context. The model is expected to return a JSON object matching the AssistantAnswer schema.
    
    """
    try:
        result = llm_client.chat_with_tools(
            request.message , history=request.history 
        )
        return result 
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/chat/rag" , response_model=ChatResponse)
def chat_with_rag(request: ChatRequest):
    """
    Explicit RAG endpoint - always retireves context first then generates a strucutred validated JSON answer citing sources"""
    
    try:
        chunks = retrieve(request.message , top_k=3)
        context = format_context(chunks)
        
        raw = llm_client.chat_structured(request.message , context=context )
        validated = AssistantAnswer(**raw)
        
        return ChatResponse(
            answer = validated.answer,
            sources = validated.sources,
            confidence = validated.confidence,
        )
    except ValidationError as ve:
        raise HTTPException(status_code=500, detail=f"LLM returned invalid JSON: {ve}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/ingest")
def trigger_ingest():
    """Re-runs document ingestion from data/docs userful for adding new document without restarting the server"""
    try:
        ingest_module.ingest_all()
        return {"status":"ingestion completed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.APP_HOST, port=settings.APP_PORT, reload=True)
    
        
    

    