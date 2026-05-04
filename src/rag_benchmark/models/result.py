from typing import Optional

from pydantic import BaseModel

from .document import DocumentMetadata


class SearchResult(BaseModel):
    doc_id: str
    rank: int
    score: float
    metadata: DocumentMetadata
    text: str


class BenchmarkResult(BaseModel):
    index_type: str
    n_vectors: int
    latency_ms: float
    latency_p95_ms: float
    recall_at_k: float
    k: int
    params: dict
    n_queries: int


class GenerationResult(BaseModel):
    answer: str
    model: str
    tokens_used: int
    latency_ms: Optional[float] = None
