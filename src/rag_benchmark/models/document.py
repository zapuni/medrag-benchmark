import math
from typing import Optional

from pydantic import BaseModel, field_validator


class DocumentMetadata(BaseModel):
    id: str
    title: str
    source: str = "wikipedia"
    topic: Optional[str] = None

    @field_validator("id", "title", "source", mode="before")
    @classmethod
    def _sanitize_string(cls, value):
        if value is None:
            return ""
        if isinstance(value, float) and math.isnan(value):
            return ""
        return str(value).strip()

    @field_validator("title", mode="after")
    @classmethod
    def _ensure_title(cls, value: str) -> str:
        return value or "Unknown Title"


class Document(BaseModel):
    id: str
    content: str
    metadata: DocumentMetadata

    class Config:
        arbitrary_types_allowed = True


class Chunk(BaseModel):
    doc_id: str
    chunk_index: int
    text: str
    metadata: DocumentMetadata
