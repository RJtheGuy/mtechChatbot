import os
from pathlib import Path
import logging
from typing import List, Dict
import re

logger = logging.getLogger(__name__)

def clean_markdown(text: str) -> str:
    """Enhanced markdown cleaning that preserves structure"""
    # Remove images and links but keep text
    text = re.sub(r'!\[(.*?)\]\(.*?\)', r'\1', text)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    # Clean other markdown syntax
    text = re.sub(r'#{1,6}\s*', '', text)  # Remove headings but keep text
    text = re.sub(r'\*{1,2}(.*?)\*{1,2}', r'\1', text)  # Remove bold/italic
    text = re.sub(r'`{3}.*?`{3}', '', text, flags=re.DOTALL)  # Remove code blocks
    text = re.sub(r'`(.*?)`', r'\1', text)  # Remove inline code
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'\n{3,}', '\n\n', text)  # Normalize newlines
    return text.strip()

def read_markdown_files(readme_dir: str) -> List[Dict]:
    """Reads markdown files with improved metadata handling"""
    readme_path = Path(readme_dir)
    if not readme_path.exists():
        logger.error(f"Directory {readme_dir} does not exist")
        return []

    readmes = []
    for file in readme_path.glob("**/*.md"):
        try:
            content = file.read_text(encoding='utf-8')
            cleaned = clean_markdown(content)
            if cleaned:
                readmes.append({
                    "project": file.stem,
                    "content": cleaned,
                    "source": str(file.relative_to(readme_path))
                })
        except Exception as e:
            logger.warning(f"Error reading {file.name}: {str(e)}")
    
    logger.info(f"Loaded {len(readmes)} markdown files")
    return readmes

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 100) -> List[str]:
    """Improved chunking that preserves semantic sections"""
    # First split by major sections
    sections = [s.strip() for s in re.split(r'\n##+\s', text) if s.strip()]
    chunks = []
    
    for section in sections:
        # Preserve whole section if small enough
        if len(section.split()) <= chunk_size * 1.5:
            chunks.append(section)
            continue
            
        # Otherwise split into paragraphs
        paragraphs = [p.strip() for p in section.split('\n\n') if p.strip()]
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            words = len(para.split())
            if current_length + words > chunk_size and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = current_chunk[-overlap//50:] if overlap else []
                current_length = sum(len(p.split()) for p in current_chunk)
            
            current_chunk.append(para)
            current_length += words
        
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
    
    return chunks

def load_and_chunk_readmes(readme_dir: str, chunk_size: int = 400, overlap: int = 100) -> List[Dict]:
    """Main loading function with enhanced metadata"""
    documents = read_markdown_files(readme_dir)
    chunked_data = []
    
    for doc in documents:
        chunks = chunk_text(doc["content"], chunk_size, overlap)
        for i, chunk in enumerate(chunks):
            # Extract section title if available
            first_line = chunk.split('\n')[0]
            section = first_line if len(first_line) < 50 else f"section_{i+1}"
            
            chunked_data.append({
                "text": chunk,
                "metadata": {
                    "project": doc["project"],
                    "section": section,
                    "source": doc["source"],
                    "chunk_size": len(chunk.split()),
                    "chunk_id": i+1
                }
            })
    
    logger.info(f"Created {len(chunked_data)} chunks from {len(documents)} documents")
    return chunked_data