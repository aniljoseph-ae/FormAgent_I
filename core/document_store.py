# core/document_store.py


from typing import List, Dict, Optional
from config.config import config_manager
from config.schema import DocumentSchema, ChunkSchema
import logging
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import warnings
import os
os.environ['CHROMADB_TELEMETRY_ENABLED'] = 'false'
warnings.filterwarnings("ignore", message=".*torch.classes.*")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedDocumentStore:
    def __init__(self):
        self.config = config_manager.get_validated_config()
        self.documents: Dict[str, DocumentSchema] = {}
        self.chunks: Dict[str, List[ChunkSchema]] = {}
        
        self.embedding_model = SentenceTransformer(self.config.embeddings['model'], trust_remote_code=False)  # Ltest update
        
        self.client = chromadb.PersistentClient(
            path=self.config.vector_store['persist_directory'],
            settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )
        self.collection = self.client.get_or_create_collection(
            name=self.config.vector_store['collection_name'],
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_document(self, document: DocumentSchema, chunks: List[ChunkSchema]):
        try:
            self.documents[document.id] = document
            self.chunks[document.id] = chunks
            
            chunk_texts = [chunk.content for chunk in chunks]
            chunk_ids = [chunk.id for chunk in chunks]
            embeddings = self.embedding_model.encode(chunk_texts, batch_size=self.config.embeddings['batch_size']).tolist()
            
            self.collection.add(
                documents=chunk_texts,
                embeddings=embeddings,
                ids=chunk_ids,
                metadatas=[{"document_id": chunk.document_id, "chunk_index": chunk.chunk_index} for chunk in chunks]
            )
            logger.info(f"Added document {document.id} with {len(chunks)} chunks to ChromaDB")
        except Exception as e:
            logger.error(f"Failed to add document {document.id}: {str(e)}")
            raise
    
    def list_documents(self) -> List[DocumentSchema]:
        return list(self.documents.values())
    
    def get_document(self, document_id: str) -> Optional[DocumentSchema]:
        return self.documents.get(document_id)
    
    def search_chunks(self, query: str, top_k: int = 5, document_ids: Optional[List[str]] = None) -> List[dict]:
        try:
            query_embedding = self.embedding_model.encode([query])[0].tolist()
            filter_dict = {"document_id": {"$in": document_ids}} if document_ids else None
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filter_dict,
                include=["documents", "metadatas", "distances"]
            )
            return [
                {
                    "id": id,
                    "content": doc,
                    "document_id": meta["document_id"],
                    "chunk_index": meta["chunk_index"],
                    "distance": dist
                }
                for id, doc, meta, dist in zip(
                    results["ids"][0],
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                )
            ]
        except Exception as e:
            logger.error(f"Chunk search failed: {str(e)}")
            return []