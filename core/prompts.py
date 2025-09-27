# core/prompts.py

from langchain.prompts import PromptTemplate

qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""Given the following context from documents:
{context}

Answer the question: {question}
Provide a concise and accurate answer based on the context."""
)

summarization_prompt = PromptTemplate(
    input_variables=["context", "document_id"],
    template="""Summarize the following document (ID: {document_id}):
{context}

Provide a concise summary including key fields, entities, and transactions."""
)

holistic_prompt = PromptTemplate(
    input_variables=["context", "document_ids"],
    template="""Compare the following documents (IDs: {document_ids}):
{context}

Provide a holistic analysis, highlighting similarities, differences, and key insights across fields, entities, and transactions."""
)
