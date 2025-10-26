"""
Utility Functions for HVAC GraphRAG System
"""
import re
from typing import List, Dict

def extract_section_number(text: str) -> str:
    """Extract section number like 901, 901.1, 901.1.1"""
    match = re.search(r'\b(\d{3}(?:\.\d+)*)\b', text)
    return match.group(1) if match else None

def build_materialized_path(section_number: str) -> Dict:
    """
    Build materialized path for hierarchical queries
    e.g., "901.1.1" → path="/9/901/901.1/901.1.1", level=4
    """
    parts = section_number.split('.')
    level = len(parts)
    
    # Build path
    if level == 1:
        path = f"/{parts[0]}"
    else:
        path = f"/{parts[0]}"
        for i in range(1, len(parts)):
            path += f"/{'.'.join(parts[:i+1])}"
    
    return {
        "number": section_number,
        "path": path,
        "level": level
    }

def clean_text(text: str) -> str:
    """Clean extracted text while preserving structure"""
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Remove excessive blank lines (more than 2 consecutive newlines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove excessive spaces on same line (but keep newlines)
    text = re.sub(r' +', ' ', text)
    
    # Remove page numbers
    text = re.sub(r'Page \d+', '', text)
    
    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    return text.strip()

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks for embeddings
    """
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    
    return chunks
