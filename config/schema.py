# config/schema.py

from pydantic import BaseModel
from typing import Dict, List, Optional
from enum import Enum
from datetime import datetime

class DocumentType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    JSON = "json"
    JPG = "jpg"
    PNG = "png"

class ChunkSchema(BaseModel):
    id: str
    document_id: str
    content: str
    chunk_index: int
    metadata: Dict

class DocumentSchema(BaseModel):
    id: str
    filename: str
    file_path: str
    file_type: str
    file_size: int
    upload_date: datetime
    processing_status: str
    metadata: Dict

class SystemConfigSchema(BaseModel):
    system: Dict
    llm: Dict
    embeddings: Dict
    vector_store: Dict
    rag_config: Dict
    agent: Dict
    evaluation: Dict
    paths: Dict
