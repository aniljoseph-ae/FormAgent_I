# core/document_processor.py


from typing import Tuple, List
from config.config import config_manager
from config.schema import DocumentSchema, ChunkSchema, DocumentType
from pathlib import Path
import logging
import re
import uuid
from datetime import datetime
import pypdf
import docx
import json
import pytesseract
from PIL import Image
import io
import os



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.config = config_manager.get_validated_config()
        try:
            self.chunk_size = self.config.rag_config.get('chunk_size', 512)
            self.chunk_overlap = self.config.rag_config.get('chunk_overlap', 50)
            self.max_chunks = self.config.rag_config.get('max_chunks_per_doc', 100)
        except AttributeError as e:
            logger.warning(f"Missing rag_config: {str(e)}. Using defaults.")
            self.chunk_size = 512
            self.chunk_overlap = 50
            self.max_chunks = 100
    
    def extract_text(self, file_path: str, file_type: str) -> str:
        try:
            file_path = Path(file_path)
            if file_type == DocumentType.PDF:
                with open(file_path, 'rb') as file:
                    reader = pypdf.PdfReader(file)
                    text = "".join(page.extract_text() or "" for page in reader.pages)
                    if not text.strip():
                        logger.info(f"No text extracted from {file_path}. Attempting OCR.")
                        text = self._ocr_pdf(file_path)
            elif file_type == DocumentType.DOCX:
                doc = docx.Document(file_path)
                text = "\n".join(paragraph.text for paragraph in doc.paragraphs if paragraph.text)
            elif file_type == DocumentType.TXT:
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
            elif file_type == DocumentType.JSON:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    text = json.dumps(data, indent=2)
            elif file_type == DocumentType.JPG:
                text = self._ocr_image(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            return text
        except Exception as e:
            logger.error(f"Text extraction failed for {file_path}: {str(e)}")
            raise
    
    def _ocr_image(self, file_path: Path) -> str:
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            logger.error(f"OCR failed for {file_path}: {str(e)}")
            return ""
    
    def _ocr_pdf(self, file_path: Path) -> str:
        try:
            from pdf2image import convert_from_path
            images = convert_from_path(file_path)
            text = ""
            for image in images:
                text += pytesseract.image_to_string(image) + "\n"
            return text
        except Exception as e:
            logger.error(f"PDF OCR failed for {file_path}: {str(e)}")
            return ""
    
    def extract_structured_data(self, text: str, filename: str) -> dict:
        structured_data = {
            'fields': {},
            'entities': {'DATE': [], 'PERSON': [], 'ORGANIZATION': []},
            'transactions': [],
            'errors': []
        }
        
        field_patterns = {
            'statement_period': r"Statement Period: ([\d-]+ to [\d-]+)",
            'account_number': r"Account Number: (\*+\d+)",
            'account_holder': r"Account Holder: ([\w\s#]+)",
            'opening_balance': r"Opening Balance: \$([\d,.]+)",
            'closing_balance': r"Closing Balance: \$([\d,.]+)",
            'total_deposits': r"Total Deposits: \$([\d,.]+)",
            'total_withdrawals': r"Total Withdrawals: \$([\d,.]+)",
            'credit_score': r"Credit Score: (\d+)",
            'name': r"Name: (\w+\s\w+)",
            'email': r"Email: ([\w\.-]+@[\w\.-]+)",
            'amount': r"Amount: \$([\d,.]+)"
        }
        
        for key, pattern in field_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                structured_data['fields'][key] = match.group(1).strip()
        
        transaction_pattern = r"Date: ([\d-]+)\s*\|\s*Description: ([^\|]+)\s*\|\s*Amount: ([+-]\$[\d,.]+)"
        transactions = re.findall(transaction_pattern, text, re.IGNORECASE)
        for date, desc, amount in transactions:
            structured_data['transactions'].append({
                'date': date.strip(),
                'description': desc.strip(),
                'amount': amount.strip()
            })
        
        date_pattern = r"\d{4}-\d{2}-\d{2}"
        person_pattern = r"Client #(\d+)|Name: (\w+\s\w+)"
        org_pattern = r"(?:LLC|Inc|Corp)\.?\b"
        
        structured_data['entities']['DATE'] = re.findall(date_pattern, text)
        person_matches = re.findall(person_pattern, text)
        structured_data['entities']['PERSON'] = [m[0] or m[1] for m in person_matches if m[0] or m[1]]
        structured_data['entities']['ORGANIZATION'] = re.findall(org_pattern, text)
        
        return structured_data
    
    def process_document(self, file_path: str) -> Tuple[DocumentSchema, List[ChunkSchema]]:
        try:
            file_path = Path(file_path)
            file_type = self._get_file_type(file_path)
            file_size = file_path.stat().st_size
            
            text = self.extract_text(str(file_path), file_type)
            structured_data = self.extract_structured_data(text, file_path.name)
            
            document_id = f"doc_{uuid.uuid4().hex[:16]}"
            document = DocumentSchema(
                id=document_id,
                filename=file_path.name,
                file_path=str(file_path),
                file_type=file_type,
                file_size=file_size,
                upload_date=datetime.now(),
                processing_status="completed",
                metadata={
                    'extracted_text': text,
                    'structured_data': structured_data,
                    'chunk_count': 0
                }
            )
            
            chunks = self._chunk_text(text, document_id)
            document.metadata['chunk_count'] = len(chunks)
            
            logger.info(f"Processed {file_path.name}: {len(chunks)} chunks")
            return document, chunks
        
        except Exception as e:
            logger.error(f"Document processing failed for {file_path}: {str(e)}")
            document = DocumentSchema(
                id=f"doc_{uuid.uuid4().hex[:16]}",
                filename=file_path.name,
                file_path=str(file_path),
                file_type=file_type,
                file_size=file_size,
                upload_date=datetime.now(),
                processing_status="failed",
                metadata={'error': str(e)}
            )
            return document, []
    
    def _get_file_type(self, file_path: Path) -> str:
        ext = file_path.suffix.lower()
        if ext == '.pdf':
            return DocumentType.PDF
        elif ext == '.docx':
            return DocumentType.DOCX
        elif ext == '.txt':
            return DocumentType.TXT
        elif ext == '.json':
            return DocumentType.JSON
        elif ext in ['.jpg', '.jpeg']:
            return DocumentType.JPG
        elif ext in ['.png',]:
            return DocumentType.JPG
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
    
    def _chunk_text(self, text: str, document_id: str) -> List[ChunkSchema]:
        try:
            chunks = []
            words = text.split()
            current_chunk = []
            current_length = 0
            chunk_index = 0
            
            for word in words:
                word_length = len(word) + 1
                if current_length + word_length <= self.chunk_size:
                    current_chunk.append(word)
                    current_length += word_length
                else:
                    chunk_text = " ".join(current_chunk)
                    if chunk_text.strip():
                        chunks.append(ChunkSchema(
                            id=f"chunk_{uuid.uuid4().hex[:16]}",
                            document_id=document_id,
                            content=chunk_text,
                            chunk_index=chunk_index,
                            metadata={}
                        ))
                        chunk_index += 1
                    current_chunk = [word]
                    current_length = word_length
                
                if len(chunks) >= self.max_chunks:
                    break
            
            if current_chunk and len(chunks) < self.max_chunks:
                chunk_text = " ".join(current_chunk)
                if chunk_text.strip():
                    chunks.append(ChunkSchema(
                        id=f"chunk_{uuid.uuid4().hex[:16]}",
                        document_id=document_id,
                        content=chunk_text,
                        chunk_index=chunk_index,
                        metadata={}
                    ))
            
            return chunks
        except Exception as e:
            logger.error(f"Chunking failed: {str(e)}")
            return []
