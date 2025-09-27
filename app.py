# app.py

import streamlit as st
from config.config import config_manager
from core.document_processor import DocumentProcessor
from core.document_store import EnhancedDocumentStore
from core.agent import RAGAugmentedAgent
from evaluation.test_suite import ComprehensiveTestSuite
import logging
from pathlib import Path
import tempfile
import os
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentAISystem:
    def __init__(self):
        self.config = config_manager.get_validated_config()
        try:
            self.document_store = EnhancedDocumentStore()
            self.document_processor = DocumentProcessor()
            self.agent = RAGAugmentedAgent(self.document_store)
            self.test_suite = ComprehensiveTestSuite(self.agent)
        except Exception as e:
            st.error(f"Initialization failed: {str(e)}")
            logger.error(f"Initialization failed: {str(e)}")
            raise
        
        # Initialize session state
        if 'documents' not in st.session_state:
            st.session_state.documents = {}
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'selected_doc_ids' not in st.session_state:
            st.session_state.selected_doc_ids = []
    
    def setup_streamlit(self):
        st.set_page_config(
            page_title="RAG-Augmented Document AI System",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
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
                logger.info(f"Processed {uploaded_file.name} with {len(document.metadata.get('extracted_text', ''))} characters, {len(chunks)} chunks")
            except Exception as e:
                st.sidebar.error(f"❌ Failed to process {uploaded_file.name}: {str(e)}")
                logger.error(f"Processing failed for {uploaded_file.name}: {str(e)}")
            finally:
                os.unlink(tmp_path)
    
    def render_sidebar(self):
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
                # Select All/Deselect All Checkbox
                select_all = st.checkbox("Select All Documents", key="select_all_docs")
                if select_all:
                    st.session_state.selected_doc_ids = [doc.id for doc in documents]
                else:
                    if st.session_state.selected_doc_ids == [doc.id for doc in documents]:
                        st.session_state.selected_doc_ids = []
                
                # Individual Document Checkboxes
                for doc in documents:
                    is_selected = st.checkbox(
                        f"{doc.filename} (ID: {doc.id})",
                        value=doc.id in st.session_state.selected_doc_ids,
                        key=f"doc_{doc.id}"
                    )
                    if is_selected and doc.id not in st.session_state.selected_doc_ids:
                        st.session_state.selected_doc_ids.append(doc.id)
                    elif not is_selected and doc.id in st.session_state.selected_doc_ids:
                        st.session_state.selected_doc_ids.remove(doc.id)
                
                # Download button for each document
                for doc in documents:
                    if 'extracted_text' in doc.metadata:
                        st.download_button(
                            label=f"Download {doc.filename} Text",
                            data=doc.metadata['extracted_text'],
                            file_name=f"{doc.filename}_extracted.txt",
                            mime="text/plain",
                            key=f"download_{doc.id}"
                        )
                    st.info(f"Status: {doc.processing_status} | Chunks: {doc.metadata.get('chunk_count', 0)}")
                
                # Previews for Selected Documents
                if st.session_state.selected_doc_ids:
                    st.subheader("Selected Document Previews")
                    for doc_id in st.session_state.selected_doc_ids:
                        doc = self.document_store.get_document(doc_id)
                        if doc:
                            with st.expander(f"Preview: {doc.filename}"):
                                st.markdown(self.generate_markdown_preview(doc))

if __name__ == "__main__":
    app = DocumentAISystem()
    app.setup_streamlit()
    # Multi-page navigation
    pages = {
        "Chat": "pages/chat.py",
        "Evaluation & Debugging": "pages/evaluation.py"
    }
    st.navbar = st.radio(
        "Navigate",
        list(pages.keys()),
        horizontal=True,
        label_visibility="collapsed"
    )
    st.switch_page(pages[st.navbar])