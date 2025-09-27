# core/document_store.py

import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Optional, Dict, Any
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
from config.schema import DocumentSchema, ChunkSchema, QuerySchema
from config.config import config_manager
class EnhancedDocumentStore:
    """Enhanced document store with vector search capabilities"""
   
    def __init__(self):
        self.config = config_manager.get_validated_config()
        self.embedding_model = SentenceTransformer(
            self.config.embedding_config['model']
        )
       
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=self.config.vector_store_config['persist_directory'],
            settings=ChromaSettings(anonymized_telemetry=False)
        )
       
        self.collection = self.client.get_or_create_collection(
            name=self.config.vector_store_config['collection_name'],
            metadata={"description": "Document chunks with embeddings"}
        )
       
        self.documents: Dict[str, DocumentSchema] = {}
        self.chunks: Dict[str, ChunkSchema] = {}
   
    def add_document(self, document: DocumentSchema, chunks: List[ChunkSchema]):
        """Add document and its chunks to the store"""
        self.documents[document.id] = document
       
        # Generate embeddings and add to vector store
        chunk_texts = []
        chunk_embeddings = []
        chunk_metadatas = []
        chunk_ids = []
       
        for chunk in chunks:
            self.chunks[chunk.id] = chunk
           
            # Generate embedding
            embedding = self.embedding_model.encode(chunk.content).tolist()
            chunk.embedding = embedding
           
            # Prepare for ChromaDB
            chunk_texts.append(chunk.content)
            chunk_embeddings.append(embedding)
            chunk_metadatas.append({
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                **chunk.metadata
            })
            chunk_ids.append(chunk.id)
       
        # Add to ChromaDB collection
        self.collection.add(
            embeddings=chunk_embeddings,
            documents=chunk_texts,
            metadatas=chunk_metadatas,
            ids=chunk_ids
        )
   
    def search(self, query: QuerySchema, top_k: Optional[int] = None) -> List[ChunkSchema]:
        """Semantic search with filtering"""
        if top_k is None:
            top_k = self.config.rag_config['top_k']
       
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query.query_text).tolist()
       
        # Build filters
        where_filter = {}
        if query.document_ids and query.query_type != "holistic_analysis":
            where_filter["document_id"] = {"$in": query.document_ids}
       
        # Perform search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter if where_filter else None
        )
       
        # Convert to ChunkSchema objects
        retrieved_chunks = []
        for i, chunk_id in enumerate(results['ids'][0]):
            if chunk_id in self.chunks:
                chunk = self.chunks[chunk_id]
                chunk.metadata["similarity_score"] = results['distances'][0][i]
                retrieved_chunks.append(chunk)
       
        return retrieved_chunks
   
    def get_document(self, document_id: str) -> Optional[DocumentSchema]:
        """Get document by ID"""
        return self.documents.get(document_id)
   
    def list_documents(self) -> List[DocumentSchema]:
        """List all documents"""
        return list(self.documents.values())
   
    def delete_document(self, document_id: str):
        """Delete document and its chunks"""
        if document_id in self.documents:
            del self.documents[document_id]
       
        # Remove chunks from vector store
        chunk_ids_to_remove = [
            chunk_id for chunk_id, chunk in self.chunks.items()
            if chunk.document_id == document_id
        ]
       
        for chunk_id in chunk_ids_to_remove:
            del self.chunks[chunk_id]
       
        self.collection.delete(ids=chunk_ids_to_remove)