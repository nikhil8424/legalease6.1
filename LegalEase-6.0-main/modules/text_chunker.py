# modules/text_chunker.py
"""
Optimized text chunking strategies for better performance and context preservation.
Implements semantic-aware chunking for legal documents.
"""

import re
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import math

logger = logging.getLogger(__name__)

@dataclass
class ChunkConfig:
    """Configuration for text chunking."""
    max_chunk_size: int = 2000
    min_chunk_size: int = 100
    overlap_size: int = 200
    preserve_sentences: bool = True
    preserve_paragraphs: bool = True
    legal_boundary_bonus: int = 50  # Extra characters for legal boundaries

@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""
    content: str
    start_pos: int
    end_pos: int
    chunk_id: int
    importance_score: float = 0.0
    contains_legal_terms: bool = False
    section_type: Optional[str] = None

class OptimizedTextChunker:
    """
    Advanced text chunker optimized for legal documents with semantic awareness.
    """
    
    def __init__(self, config: Optional[ChunkConfig] = None):
        self.config = config or ChunkConfig()
        
        # Legal document patterns for smart boundaries
        self.legal_patterns = {
            'section_header': re.compile(r'\b(?:SECTION|ARTICLE|CLAUSE|PARAGRAPH|SUBSECTION)\s+\d+', re.IGNORECASE),
            'definitions': re.compile(r'\b(?:DEFINITIONS?|DEFINED TERMS?)\b', re.IGNORECASE),
            'whereas_clause': re.compile(r'\bWHEREAS\b', re.IGNORECASE),
            'party_reference': re.compile(r'\b(?:PARTY|PARTIES|PLAINTIFF|DEFENDANT|PETITIONER|RESPONDENT)\b', re.IGNORECASE),
            'legal_action': re.compile(r'\b(?:AGREES?|COVENANTS?|WARRANTS?|REPRESENTS?|UNDERTAKES?)\b', re.IGNORECASE),
            'termination': re.compile(r'\b(?:TERMINATION|EXPIRY|EXPIRATION|END)\b', re.IGNORECASE),
        }
        
        # High-value legal terms that should stay together
        self.legal_keyphrases = [
            'force majeure', 'intellectual property', 'confidentiality agreement',
            'non-disclosure', 'governing law', 'dispute resolution', 'liquidated damages',
            'specific performance', 'material breach', 'notice period'
        ]
        
        logger.info(f"OptimizedTextChunker initialized with max_chunk_size={self.config.max_chunk_size}")
    
    def calculate_importance_score(self, text: str) -> float:
        """Calculate importance score for text chunk based on legal content."""
        score = 0.0
        text_lower = text.lower()
        
        # Score based on legal patterns
        for pattern_name, pattern in self.legal_patterns.items():
            matches = len(pattern.findall(text))
            if matches > 0:
                score += matches * 10  # 10 points per legal pattern match
        
        # Score based on legal keyphrases
        for phrase in self.legal_keyphrases:
            if phrase in text_lower:
                score += 15  # 15 points per keyphrase
        
        # Score based on text density (shorter, information-dense chunks score higher)
        if len(text.strip()) > 0:
            word_count = len(text.split())
            char_count = len(text.strip())
            density = word_count / max(char_count, 1) * 100
            score += density
        
        # Normalize score to 0-100 range
        return min(score, 100.0)
    
    def find_legal_boundaries(self, text: str) -> List[int]:
        """Find optimal chunk boundaries based on legal document structure."""
        boundaries = [0]  # Always start with beginning
        
        # Find paragraph boundaries
        if self.config.preserve_paragraphs:
            for match in re.finditer(r'\n\s*\n', text):
                boundaries.append(match.end())
        
        # Find section boundaries
        for pattern in self.legal_patterns.values():
            for match in pattern.finditer(text):
                # Add boundary at start of legal section
                boundaries.append(match.start())
        
        # Find sentence boundaries if needed
        if self.config.preserve_sentences:
            sentence_ends = re.finditer(r'[.!?]\s+(?=[A-Z])', text)
            for match in sentence_ends:
                boundaries.append(match.end())
        
        # Remove duplicates and sort
        boundaries = sorted(set(boundaries))
        boundaries.append(len(text))  # Always end with text length
        
        return boundaries
    
    def create_chunk_with_overlap(self, text: str, start: int, end: int, chunk_id: int) -> TextChunk:
        """Create a text chunk with proper overlap handling."""
        # Adjust boundaries for overlap
        actual_start = max(0, start - self.config.overlap_size // 2)
        actual_end = min(len(text), end + self.config.overlap_size // 2)
        
        chunk_text = text[actual_start:actual_end]
        importance_score = self.calculate_importance_score(chunk_text)
        
        # Detect legal content
        contains_legal = any(pattern.search(chunk_text) for pattern in self.legal_patterns.values())
        
        # Detect section type
        section_type = None
        for section_name, pattern in self.legal_patterns.items():
            if pattern.search(chunk_text):
                section_type = section_name
                break
        
        return TextChunk(
            content=chunk_text,
            start_pos=actual_start,
            end_pos=actual_end,
            chunk_id=chunk_id,
            importance_score=importance_score,
            contains_legal_terms=contains_legal,
            section_type=section_type
        )
    
    def semantic_chunking(self, text: str) -> List[TextChunk]:
        """
        Advanced semantic chunking that respects legal document structure.
        """
        if not text.strip():
            return []
        
        chunks = []
        boundaries = self.find_legal_boundaries(text)
        
        current_start = 0
        chunk_id = 0
        
        while current_start < len(text):
            # Find the best end position
            target_end = current_start + self.config.max_chunk_size
            
            if target_end >= len(text):
                # Last chunk
                chunk = self.create_chunk_with_overlap(text, current_start, len(text), chunk_id)
                chunks.append(chunk)
                break
            
            # Find the best boundary near target_end
            best_boundary = target_end
            for boundary in boundaries:
                if current_start < boundary <= target_end + self.config.legal_boundary_bonus:
                    best_boundary = boundary
                elif boundary > target_end + self.config.legal_boundary_bonus:
                    break
            
            # Ensure minimum chunk size
            if best_boundary - current_start < self.config.min_chunk_size:
                best_boundary = min(current_start + self.config.max_chunk_size, len(text))
            
            chunk = self.create_chunk_with_overlap(text, current_start, best_boundary, chunk_id)
            chunks.append(chunk)
            
            current_start = best_boundary
            chunk_id += 1
        
        logger.info(f"Created {len(chunks)} semantic chunks from {len(text)} characters")
        return chunks
    
    def adaptive_chunking(self, text: str) -> List[TextChunk]:
        """
        Adaptive chunking that adjusts chunk size based on content density.
        """
        if not text.strip():
            return []
        
        # Analyze text density
        paragraphs = text.split('\n\n')
        avg_paragraph_length = sum(len(p) for p in paragraphs) / max(len(paragraphs), 1)
        
        # Adjust chunk size based on paragraph density
        if avg_paragraph_length > 500:
            # Dense text - use smaller chunks
            adapted_config = ChunkConfig(
                max_chunk_size=self.config.max_chunk_size // 2,
                min_chunk_size=self.config.min_chunk_size // 2,
                overlap_size=self.config.overlap_size,
                preserve_sentences=True,
                preserve_paragraphs=True
            )
        elif avg_paragraph_length < 100:
            # Sparse text - use larger chunks
            adapted_config = ChunkConfig(
                max_chunk_size=self.config.max_chunk_size * 2,
                min_chunk_size=self.config.min_chunk_size,
                overlap_size=self.config.overlap_size,
                preserve_sentences=True,
                preserve_paragraphs=True
            )
        else:
            adapted_config = self.config
        
        # Temporarily switch config
        original_config = self.config
        self.config = adapted_config
        
        try:
            chunks = self.semantic_chunking(text)
        finally:
            self.config = original_config
        
        return chunks
    
    def prioritized_chunking(self, text: str, max_priority_chunks: int = 3) -> Tuple[List[TextChunk], List[TextChunk]]:
        """
        Split text into high-priority and regular chunks based on legal importance.
        
        Returns:
            Tuple of (high_priority_chunks, regular_chunks)
        """
        all_chunks = self.semantic_chunking(text)
        
        # Sort by importance score
        sorted_chunks = sorted(all_chunks, key=lambda x: x.importance_score, reverse=True)
        
        high_priority = sorted_chunks[:max_priority_chunks]
        regular = sorted_chunks[max_priority_chunks:]
        
        # Re-sort by original position
        high_priority.sort(key=lambda x: x.start_pos)
        regular.sort(key=lambda x: x.start_pos)
        
        logger.info(f"Created {len(high_priority)} high-priority and {len(regular)} regular chunks")
        return high_priority, regular
    
    def get_chunk_summary(self, chunks: List[TextChunk]) -> Dict[str, any]:
        """Get summary statistics for chunks."""
        if not chunks:
            return {}
        
        total_chars = sum(len(chunk.content) for chunk in chunks)
        avg_importance = sum(chunk.importance_score for chunk in chunks) / len(chunks)
        legal_chunks = sum(1 for chunk in chunks if chunk.contains_legal_terms)
        
        section_types = {}
        for chunk in chunks:
            if chunk.section_type:
                section_types[chunk.section_type] = section_types.get(chunk.section_type, 0) + 1
        
        return {
            'total_chunks': len(chunks),
            'total_characters': total_chars,
            'avg_chunk_size': total_chars / len(chunks),
            'avg_importance_score': avg_importance,
            'legal_chunks_count': legal_chunks,
            'legal_chunks_percentage': (legal_chunks / len(chunks)) * 100,
            'section_types': section_types
        }

# Global chunker instance with optimized defaults for legal documents
default_chunker = OptimizedTextChunker(
    ChunkConfig(
        max_chunk_size=1500,  # Optimized for LLM context windows
        min_chunk_size=200,   # Ensure meaningful content
        overlap_size=150,     # Good balance for context preservation
        preserve_sentences=True,
        preserve_paragraphs=True,
        legal_boundary_bonus=100  # Prefer legal boundaries
    )
)

def chunk_text_for_processing(text: str, processing_type: str = "general") -> List[TextChunk]:
    """
    Convenience function to chunk text based on processing type.
    
    Args:
        text: Input text to chunk
        processing_type: Type of processing ('key_extraction', 'summarization', 'analysis')
    
    Returns:
        List of optimized text chunks
    """
    if processing_type == "key_extraction":
        # Use prioritized chunking for key extraction
        high_priority, regular = default_chunker.prioritized_chunking(text, max_priority_chunks=5)
        return high_priority + regular[:10]  # Limit total chunks for key extraction
    
    elif processing_type == "summarization":
        # Use adaptive chunking for summarization
        return default_chunker.adaptive_chunking(text)[:8]  # Limit for summarization
    
    else:
        # Use semantic chunking for general processing
        return default_chunker.semantic_chunking(text)
