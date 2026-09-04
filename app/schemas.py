from pydantic import BaseModel, Field
from typing import Optional


class SourceRef(BaseModel):
    document: str
    chunk_id: str


class AssistantAnswer(BaseModel):
    answer: str = Field(..., description="The direct answer to the user's question")
    sources: list[SourceRef] = Field(default_factory=list, description="Chunks used to answer, if any")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model's self-rated confidence 0-1")


class ChatRequest(BaseModel):
    message: str
    history: Optional[list[dict]] = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceRef] = []
    confidence: float