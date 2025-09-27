# pages / chat.py

import streamlit as st
from core.agent import RAGAugmentedAgent
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    st.title("💬 Document Chat")
    st.markdown("Interact with your documents using QA, summarization, or holistic analysis.")
    
    # Initialize agent from session state or create new
    if 'agent' not in st.session_state:
        st.error("Document AI System not initialized. Please restart the app.")
        return
    
    agent = st.session_state.get('agent')
    documents = st.session_state.get('document_store').list_documents()
    
    # Chat container
    chat_container = st.container()
    
    # Text input at the bottom
    with st.form(key="query_form", clear_on_submit=True):
        query = st.text_input(
            "Ask a question about your documents:",
            placeholder="e.g., Summarize financial_statement.pdf",
            key="chat_input"
        )
        doc_options = [f"{doc.filename} (ID: {doc.id})" for doc in documents]
        selected_docs = st.multiselect(
            "Select documents to query (leave empty for all):",
            doc_options,
            default=[f"{doc.filename} (ID: {doc.id})" for doc_id in st.session_state.get('selected_doc_ids', [])]
        )
        submit_button = st.form_submit_button("Send")
    
    # Process query
    if submit_button and query:
        selected_doc_ids = [doc.id for doc in documents if f"{doc.filename} (ID: {doc.id})" in selected_docs]
        with chat_container:
            with st.chat_message("user"):
                st.markdown(query)
            st.session_state.chat_history.append({"role": "user", "content": query})
            
            with st.spinner("Processing your query..."):
                try:
                    result = agent.query(query, selected_doc_ids if selected_doc_ids else None)
                    with st.chat_message("assistant"):
                        st.markdown(result['response'])
                    st.session_state.chat_history.append({"role": "assistant", "content": result['response']})
                    
                    with st.expander("View Retrieved Context"):
                        if result['retrieved_chunks']:
                            for chunk in result['retrieved_chunks']:
                                st.write(f"**Document {chunk['document_id']}:**")
                                st.write(chunk['content'])
                                st.write(f"**Distance:** {chunk['distance']:.3f}")
                                st.divider()
                        else:
                            st.write("No chunks retrieved.")
                    
                    with st.expander("View Query Metadata"):
                        st.json(result['metadata'])
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    logger.error(f"Query error: {str(e)}")
    
    # Display chat history
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg['role']):
                st.markdown(msg['content'])

if __name__ == "__main__":
    main()