# main.py


import streamlit as st
import os
from pathlib import Path
import tempfile
import json
from config.config import config_manager
from core.document_processor import DocumentProcessor
from core.document_store import EnhancedDocumentStore
from core.agent import RAGAugmentedAgent
from evaluation.test_suite import ComprehensiveTestSuite
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentAISystem:
    def __init__(self):
        self.config = config_manager.get_validated_config()
        self.document_store = EnhancedDocumentStore()
        self.document_processor = DocumentProcessor()
        self.agent = RAGAugmentedAgent(self.document_store)
        self.test_suite = ComprehensiveTestSuite(self.agent)
        
        if 'documents' not in st.session_state:
            st.session_state.documents = {}
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        self.setup_streamlit()
    
    def setup_streamlit(self):
        st.set_page_config(
            page_title="RAG-Augmented Document AI System",
            page_icon="📊",
            layout="wide"
        )
        
        st.title("RAG-Augmented Agentic Document AI System")
        st.markdown("Advanced document processing with agentic RAG capabilities")
    
    def generate_markdown_preview(self, doc):
        extracted_text = doc.metadata.get('extracted_text', 'No text available')
        metadata = doc.metadata.get('structured_data', {})
        
        md = [f"**Preview: {doc.filename}**", f"**Document ID:** {doc.id}"]
        preview_text = extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text
        md.append(f"**Text:**\n{preview_text}")
        
        if 'fields' in metadata:
            md.append("\n**Fields**")
            md.append("| Field Name | Value |")
            md.append("|------------|-------|")
            for key, value in metadata['fields'].items():
                md.append(f"| {key} | {value} |")
        
        if 'entities' in metadata:
            md.append("\n**Entities**")
            md.append("| Entity Type | Values |")
            md.append("|-------------|--------|")
            for ent_type, values in metadata['entities'].items():
                md.append(f"| {ent_type} | {', '.join(values) if values else ''} |")
        
        if 'transactions' in metadata:
            md.append("\n**Transactions**")
            md.append("| Date | Description | Amount |")
            md.append("|------|-------------|--------|")
            for tx in metadata['transactions']:
                md.append(f"| {tx['date']} | {tx['description']} | {tx['amount']} |")
        
        md.append("\n**Errors**")
        md.append("None" if not metadata.get('errors') else "\n".join(metadata.get('errors', [])))
        
        return "\n".join(md)
    
    def run(self):
        with st.sidebar:
            st.header("📁 Document Management")
            uploaded_files = st.file_uploader(
                "Upload Documents",
                type=['pdf', 'docx', 'txt', 'json', 'jpg', 'jpeg', 'png'],
                accept_multiple_files=True
            )
            
            if uploaded_files:
                self.process_uploaded_files(uploaded_files)
            
            documents = self.document_store.list_documents()
            if documents:
                st.subheader("Loaded Documents")
                for doc in documents:
                    with st.expander(f"📄 {doc.filename} (ID: {doc.id})"):
                        st.markdown(self.generate_markdown_preview(doc))
                        if 'extracted_text' in doc.metadata:
                            st.download_button(
                                label="Download Full Text",
                                data=doc.metadata['extracted_text'],
                                file_name=f"{doc.filename}_extracted.txt",
                                mime="text/plain",
                                key=f"download_{doc.id}"
                            )
                        st.info(f"Status: {doc.processing_status} | Chunks: {doc.metadata.get('chunk_count', 0)}")
        
        st.header("💬 Document Chat")
        
        query = st.text_input("Ask a question about your documents:", placeholder="e.g., Summarize financial_statement.pdf")
        
        documents = self.document_store.list_documents()
        doc_options = [doc.id for doc in documents]
        selected_docs = st.multiselect("Select documents to query (leave empty for all):", doc_options)
        
        if query:
            with st.chat_message("user"):
                st.markdown(query)
            st.session_state.chat_history.append({"role": "user", "content": query})
            
            with st.spinner("Processing your query..."):
                try:
                    result = self.agent.query(query, selected_docs if selected_docs else None)
                    with st.chat_message("assistant"):
                        st.markdown(result['response'])
                    st.session_state.chat_history.append({"role": "assistant", "content": result['response']})
                    
                    with st.expander("View Retrieved Context"):
                        for chunk in result['retrieved_chunks']:
                            st.write(f"**Document {chunk['document_id']}:**")
                            st.write(chunk['content'])
                            st.divider()
                    
                    with st.expander("View Query Metadata"):
                        st.json(result['metadata'])
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        for msg in st.session_state.chat_history:
            with st.chat_message(msg['role']):
                st.markdown(msg['content'])
        
        if st.sidebar.button("Run Comprehensive Evaluation"):
            self.run_evaluation()
    
    def process_uploaded_files(self, uploaded_files):
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                document, chunks = self.document_processor.process_document(tmp_path)
                self.document_store.add_document(document, chunks)
                st.session_state.documents[document.id] = document
                st.sidebar.success(f"✅ Processed {uploaded_file.name}")
                logger.info(f"Processed {uploaded_file.name} with {len(document.metadata.get('extracted_text', ''))} characters")
            except Exception as e:
                st.sidebar.error(f"❌ Failed to process {uploaded_file.name}: {str(e)}")
                logger.error(f"Processing failed for {uploaded_file.name}: {str(e)}")
            finally:
                os.unlink(tmp_path)
    
    def run_evaluation(self):
        st.sidebar.info("Running comprehensive evaluation...")
        
        documents = self.document_store.list_documents()
        if not documents:
            st.warning("No documents loaded. Upload forms first.")
            return
        
        test_cases = [
            {
                "question": "What is the account holder's name in financial_statement.pdf?",
                "expected_answer": "Client #35192",
                "expected_chunks": [],
                "document_ids": [doc.id for doc in documents if "financial_statement" in doc.filename]
            },
            {
                "question": "Summarize financial_statement.pdf",
                "expected_answer": "Summary of financial details including transactions and balances.",
                "document_ids": [doc.id for doc in documents if "financial_statement" in doc.filename]
            },
            {
                "question": "Compare key fields across all documents",
                "expected_answer": "Holistic comparison of forms.",
                "document_ids": [doc.id for doc in documents]
            }
        ]
        
        try:
            results = self.test_suite.run_comprehensive_evaluation(test_cases)
            
            st.header("📊 Evaluation Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("RAGAS Metrics")
                ragas_metrics = results['ragas_metrics']
                st.metric("Answer Relevancy", f"{ragas_metrics['answer_relevancy']:.3f}")
                st.metric("Faithfulness", f"{ragas_metrics['faithfulness']:.3f}")
                st.metric("Context Precision", f"{ragas_metrics['context_precision']:.3f}")
                st.metric("Context Recall", f"{ragas_metrics['context_recall']:.3f}")
            
            with col2:
                st.subheader("Retrieval Metrics")
                retrieval_metrics = results['retrieval_metrics']
                st.metric("Retrieval Precision", f"{retrieval_metrics['retrieval_precision']:.3f}")
                st.metric("Retrieval Recall", f"{retrieval_metrics['retrieval_recall']:.3f}")
                st.metric("Retrieval F1", f"{retrieval_metrics['retrieval_f1']:.3f}")
            
            st.subheader("Summary")
            summary = results['summary']
            st.metric("Overall Score", f"{summary['overall_score']:.3f}")
            
            st.write("**Strengths:**")
            for strength in summary['strengths']:
                st.write(f"✅ {strength}")
            
            st.write("**Improvement Areas:**")
            for area in summary['improvement_areas']:
                st.write(f"🔧 {area}")
            
            st.write("**Recommendations:**")
            for rec in summary['recommendations']:
                st.write(f"💡 {rec}")
        except Exception as e:
            st.error(f"Evaluation failed: {str(e)}")
            logger.error(f"Evaluation error: {str(e)}")

if __name__ == "__main__":
    app = DocumentAISystem()
    app.run()
