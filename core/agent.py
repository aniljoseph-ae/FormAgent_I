# core/agent.py

from typing import Dict, Any, List, Optional
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
import json
from config.schema import AgentStateSchema, QuerySchema, DocumentSchema, ChunkSchema
from config.config import config_manager
from .prompts import PromptManager
from .document_store import EnhancedDocumentStore
class RAGAugmentedAgent:
    """RAG-Augmented Agentic System with LangGraph"""
   
    def __init__(self, document_store: EnhancedDocumentStore):
        self.config = config_manager.get_validated_config()
        self.document_store = document_store
        self.prompt_manager = PromptManager()
       
        # Initialize LLM
        self.llm = ChatGroq(
            groq_api_key=self.config.llm_config['api_key'],
            model_name=self.config.llm_config['model'],
            temperature=self.config.llm_config['temperature'],
            max_tokens=self.config.llm_config['max_tokens']
        )
       
        # Build the agent graph
        self.graph = self._build_agent_graph()
   
    def _classify_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Classify query type using LLM"""
        available_docs = [doc.id for doc in self.document_store.list_documents()]
       
        prompt = self.prompt_manager.format_prompt(
            "query_classification",
            query=state['query'].query_text,
            available_docs=available_docs
        )
       
        response = self.llm.invoke(prompt)
       
        try:
            classification = json.loads(response.content)
            state['query'].query_type = classification['query_type']
            state['metadata']['classification'] = classification
        except:
            # Fallback classification
            query_text = state['query'].query_text.lower()
            if 'all' in query_text or 'compare' in query_text or 'across' in query_text:
                state['query'].query_type = "holistic_analysis"
            else:
                state['query'].query_type = "single_document"
       
        return state
   
    def _retrieve_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve relevant context using RAG"""
        retrieved_chunks = self.document_store.search(state['query'])
        state['retrieved_chunks'] = retrieved_chunks
       
        # Aggregate context
        context_parts = []
        for chunk in retrieved_chunks:
            context_parts.append(f"Document: {chunk.document_id}\nContent: {chunk.content}")
       
        state['context'] = "\n\n".join(context_parts)
        return state
   
    def _generate_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response based on query type and context"""
        query_type = state['query'].query_type
       
        if query_type == "single_document":
            prompt_template = "qa_prompt"
        elif query_type == "multi_document":
            prompt_template = "qa_prompt" # Can be specialized
        else: # holistic_analysis
            prompt_template = "holistic_analysis_prompt"
       
        prompt = self.prompt_manager.format_prompt(
            prompt_template,
            context=state['context'],
            question=state['query'].query_text
        )
       
        response = self.llm.invoke(prompt)
        state['response'] = response.content
       
        return state
   
    def _reflect_and_correct(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Reflect on response and correct if needed"""
        if not self.config.agent_config['enable_reflection']:
            return state
       
        prompt = self.prompt_manager.format_prompt(
            "reflection_prompt",
            question=state['query'].query_text,
            context=state['context'],
            response=state['response']
        )
       
        reflection = self.llm.invoke(prompt)
       
        # Simple reflection: if the reflection suggests major issues, regenerate
        if "inaccurate" in reflection.content.lower() or "missing" in reflection.content.lower():
            # Regenerate with reflection context
            corrected_prompt = f"Original response had issues: {reflection.content}\n\nPlease provide corrected response for: {state['query'].query_text}"
            corrected_response = self.llm.invoke(corrected_prompt)
            state['response'] = corrected_response.content
            state['metadata']['corrected'] = True
       
        return state
   
    def _build_agent_graph(self) -> StateGraph:
        """Build the LangGraph state machine"""
        workflow = StateGraph(AgentStateSchema)
       
        # Add nodes
        workflow.add_node("classify", self._classify_query)
        workflow.add_node("retrieve", self._retrieve_context)
        workflow.add_node("generate", self._generate_response)
        workflow.add_node("reflect", self._reflect_and_correct)
       
        # Define edges
        workflow.set_entry_point("classify")
        workflow.add_edge("classify", "retrieve")
        workflow.add_edge("retrieve", "generate")
       
        if self.config.agent_config['enable_reflection']:
            workflow.add_edge("generate", "reflect")
            workflow.add_edge("reflect", END)
        else:
            workflow.add_edge("generate", END)
       
        return workflow.compile()
   
    def query(self, query_text: str, document_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Execute a query through the agentic system"""
        query = QuerySchema(
            query_text=query_text,
            query_type="single_document", # Will be classified
            document_ids=document_ids
        )
       
        initial_state = AgentStateSchema(
            session_id=f"session_{hash(query_text)}",
            query=query,
            documents=self.document_store.list_documents(),
            retrieved_chunks=[],
            context="",
            response="",
            history=[],
            metadata={}
        )
       
        # Execute the graph
        final_state = self.graph.invoke(initial_state)
       
        return {
            "response": final_state.response,
            "retrieved_chunks": final_state.retrieved_chunks,
            "query_type": final_state.query.query_type,
            "metadata": final_state.metadata
        }