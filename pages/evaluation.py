# pages/evalauation.py

import streamlit as st
from evaluation.test_suite import ComprehensiveTestSuite
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    st.title("📊 Evaluation & Debugging")
    st.markdown("View evaluation metrics, retrieved context, metadata, and logs for debugging.")
    
    # Initialize test suite from session state
    if 'test_suite' not in st.session_state:
        st.error("Document AI System not initialized. Please restart the app.")
        return
    
    test_suite = st.session_state.get('test_suite')
    document_store = st.session_state.get('document_store')
    documents = document_store.list_documents()
    
    if not documents:
        st.warning("No documents loaded. Upload forms first.")
        return
    
    # Run evaluation button
    if st.button("Run Comprehensive Evaluation"):
        with st.spinner("Running comprehensive evaluation..."):
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
                results = test_suite.run_comprehensive_evaluation(test_cases)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("RAGAS Metrics")
                    ragas_metrics = results['ragas_metrics']
                    st.metric("Answer Relevancy", f"{ragas_metrics.get('answer_relevancy', 0.0):.3f}")
                    st.metric("Faithfulness", f"{ragas_metrics.get('faithfulness', 0.0):.3f}")
                    st.metric("Context Precision", f"{ragas_metrics.get('context_precision', 0.0):.3f}")
                    st.metric("Context Recall", f"{ragas_metrics.get('context_recall', 0.0):.3f}")
                
                with col2:
                    st.subheader("Retrieval Metrics")
                    retrieval_metrics = results['retrieval_metrics']
                    st.metric("Retrieval Precision", f"{retrieval_metrics.get('retrieval_precision', 0.0):.3f}")
                    st.metric("Retrieval Recall", f"{retrieval_metrics.get('retrieval_recall', 0.0):.3f}")
                    st.metric("Retrieval F1", f"{retrieval_metrics.get('retrieval_f1', 0.0):.3f}")
                
                st.subheader("Summary")
                summary = results['summary']
                st.metric("Overall Score", f"{summary['overall_score']:.3f}")
                
                st.write("**Strengths:**")
                for strength in summary['strengths']:
                    if strength:
                        st.write(f"✅ {strength}")
                
                st.write("**Improvement Areas:**")
                for area in summary['improvement_areas']:
                    if area:
                        st.write(f"🔧 {area}")
                
                st.write("**Recommendations:**")
                for rec in summary['recommendations']:
                    if rec:
                        st.write(f"💡 {rec}")
            except Exception as e:
                st.error(f"Evaluation failed: {str(e)}")
                logger.error(f"Evaluation error: {str(e)}")
    
    # Display Retrieved Context and Metadata from Chat History
    st.subheader("Query History: Context & Metadata")
    if st.session_state.chat_history:
        for i, msg in enumerate(st.session_state.chat_history):
            if msg['role'] == 'assistant' and i > 0 and st.session_state.chat_history[i-1]['role'] == 'user':
                with st.expander(f"Query: {st.session_state.chat_history[i-1]['content']}"):
                    st.markdown(f"**Response:** {msg['content']}")
                    # Retrieve context/metadata from last query result
                    query = st.session_state.chat_history[i-1]['content']
                    selected_doc_ids = st.session_state.get('selected_doc_ids', [])
                    try:
                        result = test_suite.agent.query(query, selected_doc_ids if selected_doc_ids else None)
                        st.write("**Retrieved Context:**")
                        if result['retrieved_chunks']:
                            for chunk in result['retrieved_chunks']:
                                st.write(f"**Document {chunk['document_id']}:**")
                                st.write(chunk['content'])
                                st.write(f"**Distance:** {chunk['distance']:.3f}")
                                st.divider()
                        else:
                            st.write("No chunks retrieved.")
                        st.write("**Query Metadata:**")
                        st.json(result['metadata'])
                    except Exception as e:
                        st.error(f"Error retrieving context: {str(e)}")
    else:
        st.info("No queries in history.")
    
    # Display Logs
    st.subheader("Application Logs")
    log_file = Path("logs/app.log")
    if log_file.exists():
        with open(log_file, 'r') as f:
            logs = f.read()
        st.text_area("Logs", logs, height=300)
    else:
        st.info("No logs found. Ensure logging is configured to write to logs/app.log.")

if __name__ == "__main__":
    main()