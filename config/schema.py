# config/schema.py

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime
class DocumentType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    JSON = "json"
class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
class QueryType(str, Enum):
    SINGLE_DOCUMENT = "single_document"
    MULTI_DOCUMENT = "multi_document"
    HOLISTIC_ANALYSIS = "holistic_analysis"
class DocumentSchema(BaseModel):
    id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    file_path: str = Field(..., description="Path to stored document")
    file_type: DocumentType = Field(..., description="Document type")
    file_size: int = Field(..., description="File size in bytes")
    upload_date: datetime = Field(default_factory=datetime.now)
    processing_status: ProcessingStatus = Field(default=ProcessingStatus.PENDING)
    metadata: Dict[str, Any] = Field(default_factory=dict)
   
    class Config:
        use_enum_values = True
class ChunkSchema(BaseModel):
    id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document ID")
    content: str = Field(..., description="Chunk text content")
    chunk_index: int = Field(..., description="Position in document")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")
   
    @validator('content')
    def content_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Chunk content cannot be empty')
        return v.strip()
class QuerySchema(BaseModel):
    query_text: str = Field(..., description="User query text")
    query_type: QueryType = Field(..., description="Type of query")
    document_ids: Optional[List[str]] = Field(None, description="Specific documents to query")
    max_results: int = Field(default=5, ge=1, le=20)
   
    @validator('query_text')
    def query_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Query text cannot be empty')
        return v.strip()
class AgentStateSchema(BaseModel):
    session_id: str = Field(..., description="Unique session identifier")
    query: QuerySchema = Field(..., description="Current query")
    documents: List[DocumentSchema] = Field(default_factory=list)
    retrieved_chunks: List[ChunkSchema] = Field(default_factory=list)
    context: str = Field("", description="Aggregated context for generation")
    response: str = Field("", description="Agent response")
    history: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
   
    class Config:
        arbitrary_types_allowed = True
class EvaluationMetricsSchema(BaseModel):
    answer_relevancy: float = Field(..., ge=0, le=1, description="Answer relevancy score")
    context_precision: float = Field(..., ge=0, le=1, description="Context precision score")
    faithfulness: float = Field(..., ge=0, le=1, description="Faithfulness score")
    context_recall: float = Field(..., ge=0, le=1, description="Context recall score")
    overall_score: float = Field(..., ge=0, le=1, description="Overall evaluation score")
   
    @validator('overall_score')
    def calculate_overall(cls, v, values):
        if 'answer_relevancy' in values and 'faithfulness' in values:
            return (values['answer_relevancy'] + values['faithfulness']) / 2
        return v
class SystemConfigSchema(BaseModel):
    llm_config: Dict[str, Any] = Field(..., description="LLM configuration")
    embedding_config: Dict[str, Any] = Field(..., description="Embedding configuration")
    rag_config: Dict[str, Any] = Field(..., description="RAG configuration")
    agent_config: Dict[str, Any] = Field(..., description="Agent configuration")
    evaluation_config: Dict[str, Any] = Field(..., description="Evaluation configuration")