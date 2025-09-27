# core/workflows.py

from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict
from core.prompts import qa_prompt, summarization_prompt, holistic_prompt
from langchain_groq import ChatGroq
from config.config import config_manager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WorkflowState(TypedDict):
    query: str
    document_ids: List[str]
    retrieved_chunks: List[Dict]
    response: str
    metadata: Dict

class Workflows:
    def __init__(self, document_store, llm=None):
        self.document_store = document_store
        self.config = config_manager.get_validated_config()
        self.llm = llm or ChatGroq(
            model_name=self.config.llm['model'],
            api_key=self.config.llm['api_key'],
            temperature=self.config.llm['temperature'],
            max_tokens=self.config.llm['max_tokens']
        )
        self.max_iterations = self.config.agent['max_iterations']
    
    def retrieve_chunks(self, state: WorkflowState) -> WorkflowState:
        try:
            chunks = self.document_store.search_chunks(
                query=state['query'],
                top_k=self.config.rag_config['top_k'],
                document_ids=state['document_ids']
            )
            state['retrieved_chunks'] = chunks
            state['metadata'] = {'retrieved_chunk_ids': [chunk['id'] for chunk in chunks]}
            logger.info(f"Retrieved {len(chunks)} chunks for query: {state['query']}")
            return state
        except Exception as e:
            logger.error(f"Chunk retrieval failed: {str(e)}")
            state['response'] = f"Error retrieving chunks: {str(e)}"
            return state
    
    def qa_task(self, state: WorkflowState) -> WorkflowState:
        try:
            context = "\n".join([chunk['content'] for chunk in state['retrieved_chunks']])
            prompt = qa_prompt.format(context=context, question=state['query'])
            response = self.llm.invoke(prompt).content
            state['response'] = response
            state['metadata']['task'] = 'qa'
            return state
        except Exception as e:
            logger.error(f"QA task failed: {str(e)}")
            state['response'] = f"Error in QA task: {str(e)}"
            return state
    
    def summarization_task(self, state: WorkflowState) -> WorkflowState:
        try:
            context = "\n".join([chunk['content'] for chunk in state['retrieved_chunks']])
            prompt = summarization_prompt.format(context=context, document_id=state['document_ids'][0] if state['document_ids'] else "unknown")
            response = self.llm.invoke(prompt).content
            state['response'] = response
            state['metadata']['task'] = 'summarization'
            return state
        except Exception as e:
            logger.error(f"Summarization task failed: {str(e)}")
            state['response'] = f"Error in summarization: {str(e)}"
            return state
    
    def holistic_task(self, state: WorkflowState) -> WorkflowState:
        try:
            context = "\n".join([chunk['content'] for chunk in state['retrieved_chunks']])
            prompt = holistic_prompt.format(context=context, document_ids=", ".join(state['document_ids'] or ["all"]))
            response = self.llm.invoke(prompt).content
            state['response'] = response
            state['metadata']['task'] = 'holistic'
            return state
        except Exception as e:
            logger.error(f"Holistic task failed: {str(e)}")
            state['response'] = f"Error in holistic analysis: {str(e)}"
            return state
    
    def build_workflow(self, task_type: str) -> StateGraph:
        workflow = StateGraph(WorkflowState)
        workflow.add_node("retrieve", self.retrieve_chunks)
        if task_type == "qa":
            workflow.add_node("process", self.qa_task)
        elif task_type == "summarization":
            workflow.add_node("process", self.summarization_task)
        elif task_type == "holistic":
            workflow.add_node("process", self.holistic_task)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        workflow.add_edge("retrieve", "process")
        workflow.add_edge("process", END)
        workflow.set_entry_point("retrieve")
        return workflow.compile()
