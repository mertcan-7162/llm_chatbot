from __future__ import annotations

import uuid
from enum import Enum

from pydantic import BaseModel, Field


class ExtractionMethod(str, Enum):
    NATIVE = "native"
    OCR = "ocr"


class OCRResult(BaseModel):
    text: str
    confidence: float
    bbox: list[list[float]] = Field(description="Four corner points [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]")


class PageResult(BaseModel):
    page: int
    text: str
    method: ExtractionMethod


class Chunk(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    text: str
    metadata: ChunkMetadata


class ChunkMetadata(BaseModel):
    source: str
    page: int
    method: ExtractionMethod
    chunk_index: int


class IngestRequest(BaseModel):
    collection_name: str = "default"


class IngestResponse(BaseModel):
    source: str
    total_pages: int
    total_chunks: int
    collection_name: str
    methods_used: list[ExtractionMethod]


class ChatMessage(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    query: str
    collection_name: str = "default"
    n_results: int = Field(default=5, ge=1, le=20)
    min_score: float = Field(default=0.3, ge=0.0, le=1.0)
    conversation_history: list[ChatMessage] = Field(default_factory=list)


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[SourceReference]


class SourceReference(BaseModel):
    text: str
    source: str
    page: int
    score: float


Chunk.model_rebuild()
QueryResponse.model_rebuild()
