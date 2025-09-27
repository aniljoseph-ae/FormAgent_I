# core/prompts.py

from typing import Dict, Any
class PromptTemplates:
    """Advanced prompt templates for the agentic system"""
   
    # Query Classification
    QUERY_CLASSIFICATION = """
    Analyze the following query and classify it into one of these categories:
    - single_document: Query about a specific document or requires information from one document
    - multi_document: Query comparing or relating multiple specific documents
    - holistic_analysis: Query requiring analysis across all available documents
   
    Query: {query}
   
    Available Documents: {available_docs}
   
    Think step by step and provide your classification in JSON format:
    {{
        "query_type": "single_document|multi_document|holistic_analysis",
        "reasoning": "step-by-step reasoning",
        "relevant_documents": ["doc_id1", "doc_id2"] # if applicable
    }}
    """
   
    # QA Prompt with Chain-of-Thought
    QA_PROMPT = """
    You are a expert document analyst. Use the following context to answer the question.
   
    Context:
    {context}
   
    Question: {question}
   
    Think step by step:
    1. Identify key entities and concepts in the question
    2. Locate relevant information in the context
    3. Synthesize the information into a coherent answer
    4. If information is missing, clearly state what cannot be answered
   
    Provide a concise, accurate answer based solely on the provided context.
    """
   
    # Summarization Prompt
    SUMMARIZATION_PROMPT = """
    Summarize the following document content in a structured format:
   
    Document Content:
    {content}
   
    Provide a summary with:
    - Overview: Brief purpose and main topic
    - Key Details: Important facts, figures, and entities
    - Insights: Analytical observations and implications
   
    Keep the summary under 200 words and maintain factual accuracy.
    """
   
    # Holistic Analysis Prompt
    HOLISTIC_ANALYSIS_PROMPT = """
    Analyze the following documents holistically and provide comprehensive insights:
   
    Documents Context:
    {context}
   
    Analysis Request: {question}
   
    Provide analysis covering:
    1. Cross-document similarities and patterns
    2. Key differences and contradictions
    3. Overall trends and insights
    4. Actionable recommendations
   
    Support your analysis with specific evidence from the documents.
    """
   
    # Reflection Prompt for Self-Correction
    REFLECTION_PROMPT = """
    Review the following response and context to identify any issues:
   
    Original Question: {question}
    Context: {context}
    Initial Response: {response}
   
    Identify:
    1. Factual inaccuracies or hallucinations
    2. Missing information from context
    3. Logical inconsistencies
    4. Opportunities for improvement
   
    Provide corrected response if needed.
    """
class PromptManager:
    """Manage and format prompts dynamically"""
   
    def __init__(self):
        self.templates = PromptTemplates()
   
    def format_prompt(self, template_name: str, **kwargs) -> str:
        """Format a prompt template with provided variables"""
        template = getattr(self.templates, template_name.upper())
        return template.format(**kwargs)