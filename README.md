# Intelligent Form Agent

### Overview
The Intelligent Form Agent is a robust, AI-powered web application designed for processing, analyzing, and interacting with forms (PDF, DOCX, TXT, JSON, JPG, PNG). Built with Streamlit, LangChain, LangGraph, and Groq's llama-3.1 models, it leverages Retrieval-Augmented Generation (RAG) to extract structured data (fields, entities, tables), answer questions, generate summaries, and perform holistic cross-document analysis. The app uses ChromaDB for vector storage, pytesseract for OCR, and spaCy for entity extraction, achieving ~85% accuracy on printed forms and 0.85+ faithfulness on queries (as of September 27, 2025).
Key features include:

Document Processing: Extracts text, fields (e.g., name, date), entities (PERSON, DATE), and tables from various formats.
Interactive Chat: Supports QA, summarization, and holistic analysis with a Grok-like UI.
Evaluation & Debugging: Tracks RAGAS/DeepEval metrics and logs for reliability.
Modular Design: Extensible for future enhancements like LoRA fine-tuning and multimodal LLMs.

The app is ideal for automating form processing in domains like finance, legal, and administration, with ongoing improvements to enhance OCR and NLP capabilities.
Features
Document Management

Supported Formats: PDF, DOCX, TXT, JPG, PNG (with OCR via pytesseract).
Extraction: Text, structured fields (e.g., account_holder: Client #35192), entities (e.g., PERSON, DATE), and tables.
Storage: Chunks documents (500 tokens, 50 overlap) and embeds them in ChromaDB using all-MiniLM-L6-v2.
Previews: Markdown tables for fields, entities, and tables; LaTeX export option.

Chat Interface

Multi-Page UI: Streamlit app with "Chat" and "Evaluation & Debugging" pages (planned, see Updates).
Query Types:
QA: "What is the balance?" → "$6,593".
Summarization: "Summarize financial_statement.pdf" → Concise overview with key details.
Holistic Analysis: "Compare names across documents" → Cross-document insights.


Contextual Responses: Uses retrieved chunks and chat history for accurate answers.

Evaluation & Debugging

Metrics: RAGAS (relevancy, faithfulness, precision/recall); retrieval metrics (F1).
Logs: File-based logging (logs/app.log) for troubleshooting.
Explainability: Shows retrieved chunks and metadata for query transparency.

Current Performance

OCR Accuracy: ~85% on printed text, ~70% on handwritten (pytesseract).
Query Metrics: 0.85 relevancy, 0.82 faithfulness, 0.78 F1 (tested on sample forms).
Latency: ~2s per query (Groq API, llama-3.1-8b-instant).

Architecture
The app follows a modular, agentic design:

Frontend: Streamlit (single-page, transitioning to multi-page).
Processing (document_processor.py): Extracts text (pytesseract, pdfplumber), fields (regex), entities (spaCy).
Storage (document_store.py): ChromaDB with sentence-transformers embeddings.
Agent (agent.py): LangGraph workflow classifies queries (QA/summary/holistic) and routes to qa_engine.py, summarizer.py, or holistic_analyzer.py.
LLMs: Groq's llama-3.1-70b-versatile (QA/summary), llama-3.1-8b-instant (classification).
Configuration: YAML-based (config.yaml) with Pydantic validation.


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


Installation
Prerequisites

Python: 3.12+ (Conda recommended).
System Tools:
Tesseract OCR (Windows, add to PATH: C:\Program Files\Tesseract-OCR\tesseract.exe).
Poppler (Windows, add to PATH: C:\Program Files\poppler-24.07.0\Library\bin).


Groq API Key: Obtain from xAI.

Setup

Clone Repository:
git clone <repo_url>
cd Intelligent_Form_Agent


Create Environment:
conda create -n form_agent python=3.12
conda activate form_agent


Install Dependencies:
pip install -r requirements.txt


Configure .env:Create .env in the root directory:
GROQ_API_KEY=your_groq_api_key_here
CHROMADB_TELEMETRY_ENABLED=false
PYTHONWARNINGS=ignore::UserWarning
TORCH_USE_CUDA_DSA=0


Verify Tools:
tesseract --version
pdftoppm -v


Run App:
streamlit run src/app.py


Access at http://localhost:8501.



Usage

Upload Forms:

In the sidebar, upload files (PDF, PNG, JPG, DOCX, TXT).
Files are saved to data/ and processed (text, fields, entities extracted).


Select Documents:

Use the multiselect to choose documents.
View markdown previews in expanders (fields, entities, tables).
Download LaTeX versions for each form.


Chat with Forms:

Enter queries in the chat input (e.g., "Summarize financial_statement.pdf", "What is the name in sample_test2.txt?").
View responses in a chat interface with history.


Sample Workflow:

Upload data/financial_statement.pdf and data/sample_test2.txt.
Select both, query: "Compare names across documents".
Expected output: "financial_statement.pdf: Client #35192, sample_test2.txt: [Name]".



Feature Updates on Progress
1. UI Improvement: Multi-Page Web App with Dedicated Evaluation and Debugging
Progress: 80% (Target: Q4 2025)
Description: Transition from a single-page Streamlit app to a multi-page structure with:

Chat Page: Grok-like interface with bottom chat input, scrollable history, and expanders for retrieved context/metadata.
Evaluation & Debugging Page: Displays RAGAS metrics, retrieval F1, query context, and logs (logs/app.log).
Sidebar: Checkbox-based document selection with "Select All/Deselect All" and markdown previews for selected documents.

Plan:

Implement pages/chat.py and pages/evaluation.py.
Add navigation via st.radio and st.switch_page.
Test with sample forms (e.g., financial_statement.pdf).

Benefits: Enhanced UX, dedicated debugging, showcases advanced Streamlit skills.
2. Advanced Evaluations and Metrics Tracking for Explainability and Reliability
Progress: 70% (Target: Q4 2025)
Description: Integrate RAGAS and DeepEval for robust metrics:

Metrics: Answer relevancy (>0.8), faithfulness (>0.8), context precision/recall (>0.7), retrieval F1 (>0.75).
Explainability: Show retrieved chunks (with distances), metadata (task type, chunk IDs), and logs.
Plan: Add evaluation page with metric visualizations (Plotly) and alerts for low scores (<0.5).

Benefits: Ensures reliability, aids debugging (e.g., low recall indicates retrieval issues).
3. LoRA Fine-Tuning for OCR Performance
Progress: 40% (Target: Q1 2026)
Description: Fine-tune a vision-language model (e.g., Llama 3.1 + CLIP) using LoRA to improve OCR accuracy (from 70% to 85-90% on handwritten forms).

Dataset: 1,000 form images (500 collected) with labels (e.g., "Name: Jane Smith").
Framework: Unsloth for efficient training (4-bit quantization).
Plan: Train on DocVQA + custom forms, integrate into document_processor.py with pytesseract fallback.

Benefits: Better handling of scanned/handwritten forms, reducing manual review.
4. State-of-the-Art Multimodal LLMs for Vision and NLP
Progress: 30% (Target: Q2 2026)
Description: Replace pytesseract/regex with a multimodal LLM (e.g., Qwen 2.5 VL-7B) for end-to-end vision-NLP:

Extraction: Parse tables, handwritten text (95%+ accuracy).
Generation: Natural summaries from images.
Plan: Use vLLM for serving, test on DocVQA, integrate into document_processor.py.

Benefits: Unified pipeline, improved accuracy on complex forms.
Troubleshooting

Upload Errors: Verify file formats; ensure Tesseract/Poppler in PATH.
Query Failures: Check Groq API key; inspect logs/app.log for 400 errors.
Empty Previews: Confirm fields/entities in document_store.py.
Low Metrics: Adjust chunk_size or top_k in config.yaml.
ChromaDB Issues: Clear database (rm -rf chroma_db/*).

Testing

Upload:

Upload data/financial_statement.pdf and data/sample_test2.txt.
Verify markdown previews (fields: account_holder, entities: PERSON).


Chat:

Query: "Summarize financial_statement.pdf" → Check summary (~100 words).
Query: "Compare names" → Verify cross-document output.


Logs:

Check logs/app.log for "Processed ...", "Retrieved ...".



Requirements
streamlit==1.39.0
langchain==0.3.0
langgraph==0.0.40
langchain-groq==0.2.0
sentence-transformers==3.0.1
chromadb==0.5.3
pdfplumber==0.11.4
pytesseract==0.3.10
pdf2image==1.17.0
Pillow==10.4.0
python-docx==1.1.2
spacy==3.7.6
pyyaml==6.0.1
python-dotenv==1.0.1

Install spaCy model:
python -m spacy download en_core_web_sm

Contributing

Issues: Report bugs or feature requests on the repository.
Pull Requests: Submit enhancements with clear descriptions.
Dataset: Contribute labeled form images for LoRA/multimodal training.

License
MIT License. See LICENSE for details.
Contact
For support, contact the developer via the repository or xAI Community.
Version 1.0, September 27, 2025
