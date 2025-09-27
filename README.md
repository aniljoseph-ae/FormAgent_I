# Intelligent Form Agent

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt


Create .env file in the project root:GROQ_API_KEY=your_actual_groq_key_here
CHROMADB_TELEMETRY_ENABLED=false


Install Tesseract OCR:
Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
Add to PATH (e.g., C:\Program Files\Tesseract-OCR\tesseract.exe)


Install Poppler for pdf2image:
Windows: Download from https://github.com/oschwartz10612/poppler-windows
Add Poppler's bin folder to PATH


Run the app:streamlit run main.py



Features

Upload PDF, DOCX, TXT, JSON, JPG, scanned PDFs
Structured markdown previews (fields, entities, transactions)
QA, summarization, holistic analysis
RAGAS/DeepEval metrics
OCR support for images and scanned PDFs

Demo Queries

QA: "What is the account holder's name in financial_statement.pdf?"
Summary: "Summarize financial_statement.pdf"
Holistic: "Compare balances across documents"

Troubleshooting

TypeError in DeepEval: Ensure evaluation/groq_llm.py uses GroqDeepEvalLLM.
ChromaDB Telemetry Errors: Verify CHROMADB_TELEMETRY_ENABLED=false in .env and chromadb==0.5.3 in requirements.
Groq API Key: Run python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.getenv('GROQ_API_KEY'))" to verify key.
OCR Issues: Ensure Tesseract and Poppler are in PATH.
ChromaDB Corruption: Clear data/chroma_db/:rm -rf data/chroma_db/*


```markdown
Directory Structure
AI_Scanner_IV_Langgraph_RAG_vectrdb/
├── .env
├── config/
│   ├── config.py
│   ├── config.yaml
│   ├── schema.py
├── core/
│   ├── document_processor.py
│   ├── document_store.py
│   ├── agent.py
│   ├── workflows.py
│   ├── prompts.py
├── evaluation/
│   ├── test_suite.py
│   ├── ragas_evaluator.py
│   ├── deepeval_evaluator.py
│   ├── groq_llm.py
├── tests/
│   ├── test_deepeval.py
├── main.py
├── data/
│   ├── chroma_db/
│   ├── financial_statement.pdf
│   ├── sample_test2.txt
├── requirements.txt
├── docs/
│   ├── README.md

```

