from typing import List, Dict
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

class PDFProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract text from a PDF file with page numbers and positions."""
        try:
            reader = PdfReader(pdf_path)
            text_chunks = []
            
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                # Store the start position of each line
                lines = text.split('\n')
                current_pos = 0
                
                for line in lines:
                    if line.strip():  # Only store non-empty lines
                        text_chunks.append({
                            "text": line,
                            "page": page_num + 1,
                            "start_pos": current_pos,
                            "end_pos": current_pos + len(line)
                        })
                    current_pos += len(line) + 1  # +1 for newline
            
            return text_chunks
        except Exception as e:
            raise Exception(f"Error processing PDF {pdf_path}: {str(e)}")
    
    def process_pdf(self, pdf_path: str, original_filename: str = None) -> List[Dict]:
        """Process a PDF file and return chunks with metadata."""
        # Extract text with detailed metadata
        text_chunks = self.extract_text_from_pdf(pdf_path)
        
        # Process chunks
        processed_chunks = []
        for chunk in text_chunks:
            processed_chunks.append({
                "text": chunk["text"],
                "metadata": {
                    "source": pdf_path,
                    "original_filename": original_filename or os.path.basename(pdf_path),
                    "page": chunk["page"],
                    "start_pos": chunk["start_pos"],
                    "end_pos": chunk["end_pos"]
                }
            })
        
        return processed_chunks 