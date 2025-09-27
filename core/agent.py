# core/agent.py

from typing import List, Dict, Optional
from core.workflows import Workflows
import logging
from langchain_groq import ChatGroq
from config.config import config_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGAugmentedAgent:
    def __init__(self, document_store):
        self.document_store = document_store
        self.config = config_manager.get_validated_config()
        self.llm = ChatGroq(
            model_name=self.config.llm['model'],
            api_key=self.config.llm['api_key'],
            temperature=self.config.llm['temperature'],
            max_tokens=self.config.llm['max_tokens']
        )
        self.workflows = Workflows(document_store, self.llm)
    
    def query(self, query: str, document_ids: Optional[List[str]] = None) -> Dict:
        try:
            query_lower = query.lower()
            if "summarize" in query_lower:
                task_type = "summarization"
            elif "compare" in query_lower or "across" in query_lower:
                task_type = "holistic"
            else:
                task_type = "qa"
            
            workflow = self.workflows.build_workflow(task_type)
            state = {
                "query": query,
                "document_ids": document_ids or [doc.id for doc in self.document_store.list_documents()],
                "retrieved_chunks": [],
                "response": "",
                "metadata": {}
            }
            result = workflow.invoke(state)
            
            return {
                "response": result['response'],
                "retrieved_chunks": result['retrieved_chunks'],
                "metadata": result['metadata']
            }
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            return {
                "response": f"Error processing query: {str(e)}",
                "retrieved_chunks": [],
                "metadata": {"error": str(e)}
            }