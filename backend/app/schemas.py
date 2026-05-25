from pydantic import BaseModel
from typing import Optional


class IngestJobResponse(BaseModel):
    job_id:   str
    filename: str
    status:   str


class JobStatusResponse(BaseModel):
    job_id:       str
    filename:     str
    status:       str
    chunks_done:  int
    total_chunks: int
    figure_count: int
    table_count:  int
    error:        Optional[str]


class Source(BaseModel):
    text:        str
    source:      str
    chunk_index: int
    score:       float
    type:        str
    chunk_type:  str = "text"
    media_type:  Optional[str] = None
    image_url:   Optional[str] = None
    figure_type: Optional[str] = None
    caption:     Optional[str] = None
    page_number: Optional[int] = None


class QueryRequest(BaseModel):
    question: str
    top_k:    int = 5


class QueryResponse(BaseModel):
    answer:  str
    sources: list[Source]
