# core/document_processor.py

import fitz # PyMuPDF
from docx import Document as DocxDocument
import json
from typing import List, Optional, Dict, Any
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import hashlib
from config.schema import DocumentSchema, ChunkSchema, DocumentType, ProcessingStatus
from config.config import config_manager
class DocumentProcessor:
    """Advanced document processor with semantic chunking"""
   
    def __init__(self):
        self.config = config_manager.get_validated_config()
        self.embedding_model = SentenceTransformer(
            self.config.embedding_config['model'],
            device=self.config.embedding_config['device']
        )
       
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.rag_config.get('chunk_size', 512),
            chunk_overlap=self.config.rag_config.get('chunk_overlap', 50),
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
   
    def generate_document_id(self, file_path: str) -> str:
        """Generate unique document ID from file path and content"""
        file_hash = hashlib.md5(Path(file_path).read_bytes()).hexdigest()
        return f"doc_{file_hash[:16]}"
   
    def extract_text(self, file_path: str, file_type: DocumentType) -> str:
        """Extract text from various document formats"""
        try:
            if file_type == DocumentType.PDF:
                return self._extract_pdf_text(file_path)
            elif file_type == DocumentType.DOCX:
                return self._extract_docx_text(file_path)
            elif file_type == DocumentType.TXT:
                return self._extract_txt_text(file_path)
            elif file_type == DocumentType.JSON:
                return self._extract_json_text(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
        except Exception as e:
            raise Exception(f"Text extraction failed for {file_path}: {str(e)}")
   
    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF with formatting preservation"""
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        doc.close()
        return text
   
    def _extract_docx_text(self, file_path: str) -> str:
        """Extract text from DOCX document"""
        doc = DocxDocument(file_path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
   
    def _extract_txt_text(self, file_path: str) -> str:
        """Extract text from plain text file"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
   
    def _extract_json_text(self, file_path: str) -> str:
        """Extract text from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return json.dumps(data, indent=2)
   
    def chunk_document(self, text: str, document_id: str) -> List[ChunkSchema]:
        """Chunk document text semantically"""
        chunks = self.text_splitter.split_text(text)
       
        chunk_schemas = []
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{document_id}_chunk_{i:04d}"
           
            chunk_schema = ChunkSchema(
                id=chunk_id,
                document_id=document_id,
                content=chunk_text,
                chunk_index=i,
                metadata={
                    "chunk_length": len(chunk_text),
                    "word_count": len(chunk_text.split()),
                    "has_entities": bool(self._detect_entities(chunk_text))
                }
            )
            chunk_schemas.append(chunk_schema)
       
        return chunk_schemas
   
    def _detect_entities(self, text: str) -> List[str]:
        """Simple entity detection (can be enhanced with NER)"""
        # Basic implementation - can be replaced with spaCy or similar
        entities = []
        # Add simple entity detection logic here
        return entities
   
    def process_document(self, file_path: str) -> DocumentSchema:
        """Complete document processing pipeline"""
        file_path = Path(file_path)
        file_type = DocumentType(file_path.suffix.lower()[1:])
       
        # Generate document schema
        doc_id = self.generate_document_id(str(file_path))
        document = DocumentSchema(
            id=doc_id,
            filename=file_path.name,
            file_path=str(file_path),
            file_type=file_type,
            file_size=file_path.stat().st_size,
            processing_status=ProcessingStatus.PROCESSING
        )
       
        try:
            # Extract text
            text = self.extract_text(str(file_path), file_type)
            document.metadata["extracted_text_length"] = len(text)
           
            # Chunk document
            chunks = self.chunk_document(text, doc_id)
            document.metadata["chunk_count"] = len(chunks)
            document.processing_status = ProcessingStatus.COMPLETED
           
            return document, chunks
           
        except Exception as e:
            document.processing_status = ProcessingStatus.FAILED
            document.metadata["error"] = str(e)
            raise e