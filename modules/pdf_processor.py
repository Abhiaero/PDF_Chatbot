import PyPDF2
import io
from typing import List, Dict, Tuple
import re

class PDFProcessor:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_text_from_pdf(self, pdf_file) -> List[Tuple[str, int]]:
        """Extract text from PDF with page numbers"""
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text_with_pages = []
        
        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text.strip():
                # Clean up text and preserve structure
                text = self._clean_text(text)
                text_with_pages.append((text, page_num + 1))
        
        return text_with_pages
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove excessive whitespace but preserve line breaks for structure
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Preserve paragraph breaks
        return text.strip()
    
    def _is_structured_data(self, text: str) -> bool:
        """Check if text contains structured data like tables, charts, or lists"""
        patterns = [
            r'\d+%',  # Percentages
            r'\b\d+\.\d+\b',  # Decimal numbers
            r'\b\d+\s*[-–]\s*\d+\b',  # Number ranges
            r'\|.*\|',  # Table-like structures with pipes
            r'\b(graph|chart|table|figure)\b',  # Data visualization references
            r'.*:.*%',  # Key-value pairs with percentages
        ]
        
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _chunk_structured_data(self, text: str, page_num: int) -> List[Dict]:
        """Special chunking for structured data to keep it intact"""
        chunks = []
        
        # Split by likely section boundaries but keep structured data together
        sections = re.split(r'\n\s*\n', text)  # Split by blank lines
        
        current_chunk = ""
        current_sections = []
        
        for section in sections:
            if self._is_structured_data(section):
                # Structured data section - treat as its own chunk
                if current_chunk:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'page': page_num,
                        'is_structured': False
                    })
                    current_chunk = ""
                
                # Add structured section as separate chunk
                chunks.append({
                    'text': section.strip(),
                    'page': page_num,
                    'is_structured': True
                })
            else:
                # Regular text section
                if len(current_chunk) + len(section) > self.chunk_size:
                    if current_chunk:
                        chunks.append({
                            'text': current_chunk.strip(),
                            'page': page_num,
                            'is_structured': False
                        })
                    current_chunk = section
                else:
                    current_chunk += " " + section if current_chunk else section
        
        # Add the last chunk
        if current_chunk:
            chunks.append({
                'text': current_chunk.strip(),
                'page': page_num,
                'is_structured': False
            })
        
        return chunks
    
    def chunk_text(self, text: str, page_num: int) -> List[Dict]:
        """Split text into chunks with special handling for structured data"""
        # First check if this page contains structured data
        if self._is_structured_data(text):
            return self._chunk_structured_data(text, page_num)
        
        # Regular chunking for non-structured text
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append({
                'text': chunk_text,
                'page': page_num,
                'is_structured': False,
                'start_word': i,
                'end_word': i + len(chunk_words)
            })
        
        return chunks
    
    def process_pdf(self, pdf_file) -> List[Dict]:
        """Main method to process PDF into chunks with better structure handling"""
        text_with_pages = self.extract_text_from_pdf(pdf_file)
        all_chunks = []
        
        for text, page_num in text_with_pages:
            chunks = self.chunk_text(text, page_num)
            all_chunks.extend(chunks)
        
        print(f"✓ Processed {len(all_chunks)} chunks with structured data handling")
        
        # Count structured chunks
        structured_count = sum(1 for chunk in all_chunks if chunk.get('is_structured'))
        if structured_count > 0:
            print(f"✓ Found {structured_count} structured data chunks (tables, charts, etc.)")
        
        return all_chunks