"""
Text Processing Utilities
Handles document chunking, cleaning, and preprocessing for the RAG pipeline.
"""

from typing import List, Dict
import re


class TextProcessor:
    """
    Processes raw text documents for embedding and indexing.
    
    Key responsibilities:
    - Split documents into manageable chunks
    - Add overlap between chunks for context preservation
    - Clean and normalize text
    - Extract metadata
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the text processor.
        
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Split text into overlapping chunks for better context preservation.
        
        Why chunking?
        - LLMs have token limits
        - Smaller chunks = more precise retrieval
        - Overlap ensures context isn't lost at boundaries
        
        Args:
            text: Raw text to chunk
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text:
            return []
        
        # Clean the text first
        text = self.clean_text(text)
        
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size
            
            # If not at the end, try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings near the chunk boundary
                sentence_end = self._find_sentence_boundary(text, end)
                if sentence_end != -1:
                    end = sentence_end
            
            # Extract chunk
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunk_data = {
                    "text": chunk_text,
                    "chunk_id": chunk_id,
                    "start_index": start,
                    "end_index": end,
                }
                
                # Add metadata if provided
                if metadata:
                    chunk_data["metadata"] = metadata.copy()
                    chunk_data["metadata"]["chunk_id"] = chunk_id
                
                chunks.append(chunk_data)
                chunk_id += 1
            
            # Move to next chunk with overlap
            start = end - self.chunk_overlap
            
            # Prevent infinite loop
            if start >= len(text):
                break
        
        return chunks
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text for better embedding quality.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that don't add semantic value
        # (Keep punctuation for sentence structure)
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text.strip()
    
    def _find_sentence_boundary(self, text: str, position: int) -> int:
        """
        Find the nearest sentence boundary near the given position.
        
        Args:
            text: Full text
            position: Target position to find boundary near
            
        Returns:
            Position of sentence boundary, or -1 if not found
        """
        # Define sentence endings
        sentence_endings = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
        
        # Search window (100 chars before and after)
        search_start = max(0, position - 100)
        search_end = min(len(text), position + 100)
        search_window = text[search_start:search_end]
        
        # Find all sentence endings in window
        boundaries = []
        for ending in sentence_endings:
            pos = search_window.find(ending)
            while pos != -1:
                # Calculate absolute position
                abs_pos = search_start + pos + len(ending)
                boundaries.append(abs_pos)
                pos = search_window.find(ending, pos + 1)
        
        if not boundaries:
            return -1
        
        # Return boundary closest to target position
        closest = min(boundaries, key=lambda x: abs(x - position))
        return closest
    
    def extract_metadata(self, text: str) -> Dict:
        """
        Extract basic metadata from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of extracted metadata
        """
        return {
            "character_count": len(text),
            "word_count": len(text.split()),
            "has_urls": bool(re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)),
            "has_emails": bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)),
        }


def create_text_processor(chunk_size: int = 500, chunk_overlap: int = 50) -> TextProcessor:
    """
    Factory function to create a TextProcessor instance.
    
    Args:
        chunk_size: Maximum characters per chunk
        chunk_overlap: Number of overlapping characters
        
    Returns:
        Configured TextProcessor instance
    """
    return TextProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
