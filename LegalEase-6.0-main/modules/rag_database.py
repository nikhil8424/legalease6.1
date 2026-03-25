# modules/rag_database.py
"""
High-performance RAG database manager with optimized vector storage and retrieval.
Uses FAISS for fast similarity search and implements efficient text chunking.
"""

import os
import json
import pickle
import hashlib
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import threading
from datetime import datetime

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available, using numpy for similarity search")

from sentence_transformers import SentenceTransformer
from modules.cache_manager import document_cache
from modules.text_chunker import default_chunker, ChunkConfig
from modules.model_manager import model_manager

logger = logging.getLogger(__name__)

class RAGDatabase:
    """
    High-performance RAG database with vector indexing and smart chunking.
    """
    
    def __init__(self, db_path: str = "uploads/database", embedding_model: str = "all-MiniLM-L6-v2"):
        # Use absolute path relative to the project root
        if not os.path.isabs(db_path):
            # Get the project root directory (where app.py is located)
            project_root = Path(__file__).parent.parent
            self.db_path = project_root / db_path
        else:
            self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.text_db_path = self.db_path / "documents.json"
        self.vector_db_path = self.db_path / "vectors.index"
        self.metadata_path = self.db_path / "metadata.json"
        
        # Threading
        self._lock = threading.RLock()
        
        # Initialize embedding model lazily
        self.embedding_model_name = embedding_model
        self._embedding_model = None
        
        # Database components
        self.documents = []  # List of document chunks
        self.metadata = {}   # Document metadata
        self.vector_index = None  # FAISS index or numpy array
        self.document_embeddings = None
        
        # Configuration
        self.chunk_config = ChunkConfig(
            max_chunk_size=500,  # Smaller chunks for better retrieval
            min_chunk_size=100,
            overlap_size=50,
            preserve_sentences=True,
            preserve_paragraphs=True
        )
        
        # Load existing database
        self._load_database()
        
        logger.info(f"RAG Database initialized with {len(self.documents)} chunks")
    
    @property
    def embedding_model(self):
        """Lazy loading of embedding model."""
        if self._embedding_model is None:
            try:
                logger.info(f"Loading embedding model: {self.embedding_model_name}")
                self._embedding_model = SentenceTransformer(self.embedding_model_name)
                logger.info("Embedding model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise
        return self._embedding_model
    
    def _generate_doc_id(self, text: str) -> str:
        """Generate unique document ID based on content hash."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _chunk_document(self, text: str, doc_id: str) -> List[Dict]:
        """
        Chunk document into smaller pieces for better retrieval.
        
        Args:
            text: Document text
            doc_id: Document identifier
            
        Returns:
            List of chunk dictionaries with metadata
        """
        # Use our optimized chunker
        chunker = default_chunker
        chunker.config = self.chunk_config
        
        text_chunks = chunker.semantic_chunking(text)
        
        chunks = []
        for i, chunk in enumerate(text_chunks):
            chunk_dict = {
                'id': f"{doc_id}_chunk_{i}",
                'doc_id': doc_id,
                'chunk_index': i,
                'content': chunk.content,
                'importance_score': chunk.importance_score,
                'contains_legal_terms': chunk.contains_legal_terms,
                'section_type': chunk.section_type,
                'start_pos': chunk.start_pos,
                'end_pos': chunk.end_pos,
                'created_at': datetime.now().isoformat()
            }
            chunks.append(chunk_dict)
        
        return chunks
    
    def _create_vector_index(self, embeddings: np.ndarray) -> Optional[object]:
        """
        Create FAISS index for fast similarity search.
        
        Args:
            embeddings: Document embeddings array
            
        Returns:
            FAISS index or None if FAISS not available
        """
        if not FAISS_AVAILABLE or len(embeddings) == 0:
            return None
        
        try:
            # Create FAISS index
            dimension = embeddings.shape[1]
            
            if len(embeddings) < 1000:
                # Use flat index for small datasets
                index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            else:
                # Use IVF index for larger datasets
                nlist = min(100, len(embeddings) // 10)
                quantizer = faiss.IndexFlatIP(dimension)
                index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
                
                # Train the index
                index.train(embeddings.astype(np.float32))
            
            # Add embeddings to index
            index.add(embeddings.astype(np.float32))
            
            logger.info(f"Created FAISS index with {index.ntotal} vectors")
            return index
            
        except Exception as e:
            logger.error(f"Failed to create FAISS index: {e}")
            return None
    
    def _load_database(self):
        """Load existing database from disk."""
        try:
            # Load documents
            if self.text_db_path.exists():
                with open(self.text_db_path, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)
            
            # Load metadata
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            
            # Load vector index
            if self.documents:
                self._rebuild_vector_index()
            
        except Exception as e:
            logger.error(f"Error loading database: {e}")
            self.documents = []
            self.metadata = {}
    
    def _save_database(self):
        """Save database to disk."""
        try:
            with self._lock:
                # Save documents
                with open(self.text_db_path, 'w', encoding='utf-8') as f:
                    json.dump(self.documents, f, indent=2)
                
                # Save metadata
                self.metadata['last_updated'] = datetime.now().isoformat()
                self.metadata['total_chunks'] = len(self.documents)
                
                with open(self.metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(self.metadata, f, indent=2)
                
                # Save vector index
                if FAISS_AVAILABLE and self.vector_index:
                    faiss.write_index(self.vector_index, str(self.vector_db_path))
                
                logger.info("Database saved successfully")
                
        except Exception as e:
            logger.error(f"Error saving database: {e}")
    
    def _rebuild_vector_index(self):
        """Rebuild vector index from existing documents."""
        if not self.documents:
            return
        
        try:
            # Extract content for embedding
            contents = [doc['content'] for doc in self.documents]
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(contents)} chunks...")
            embeddings = self.embedding_model.encode(contents, show_progress_bar=True)
            
            # Store embeddings
            self.document_embeddings = embeddings
            
            # Create vector index
            self.vector_index = self._create_vector_index(embeddings)
            
            logger.info("Vector index rebuilt successfully")
            
        except Exception as e:
            logger.error(f"Error rebuilding vector index: {e}")
    
    def add_document(self, text: str, doc_metadata: Optional[Dict] = None) -> str:
        """
        Add a new document to the database.
        
        Args:
            text: Document content
            doc_metadata: Optional metadata dictionary
            
        Returns:
            Document ID
        """
        with self._lock:
            # Generate document ID
            doc_id = self._generate_doc_id(text)
            
            # Check if document already exists
            existing_docs = [d for d in self.documents if d['doc_id'] == doc_id]
            if existing_docs:
                logger.info(f"Document {doc_id} already exists, skipping")
                return doc_id
            
            # Chunk document
            chunks = self._chunk_document(text, doc_id)
            
            # Generate embeddings for new chunks
            contents = [chunk['content'] for chunk in chunks]
            embeddings = self.embedding_model.encode(contents)
            
            # Add to documents
            self.documents.extend(chunks)
            
            # Update embeddings
            if self.document_embeddings is None:
                self.document_embeddings = embeddings
            else:
                self.document_embeddings = np.vstack([self.document_embeddings, embeddings])
            
            # Update metadata
            if doc_metadata is None:
                doc_metadata = {}
            
            doc_metadata.update({
                'doc_id': doc_id,
                'chunk_count': len(chunks),
                'added_at': datetime.now().isoformat()
            })
            self.metadata[doc_id] = doc_metadata
            
            # Rebuild vector index
            self.vector_index = self._create_vector_index(self.document_embeddings)
            
            # Save database
            self._save_database()
            
            logger.info(f"Added document {doc_id} with {len(chunks)} chunks")
            return doc_id
    
    def search(self, query: str, top_k: int = 5, score_threshold: float = 0.1) -> List[Dict]:
        """
        Search for relevant documents using semantic similarity.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            score_threshold: Minimum similarity score
            
        Returns:
            List of relevant document chunks with scores
        """
        if not self.documents:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0]
            
            if FAISS_AVAILABLE and self.vector_index:
                # Use FAISS for fast search
                scores, indices = self.vector_index.search(
                    query_embedding.reshape(1, -1).astype(np.float32), 
                    top_k
                )
                
                results = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx >= 0 and score >= score_threshold:  # Valid index and score
                        doc = self.documents[idx].copy()
                        doc['similarity_score'] = float(score)
                        results.append(doc)
                
            else:
                # Fallback to numpy-based search
                similarities = np.dot(self.document_embeddings, query_embedding)
                top_indices = similarities.argsort()[-top_k:][::-1]
                
                results = []
                for idx in top_indices:
                    score = similarities[idx]
                    if score >= score_threshold:
                        doc = self.documents[idx].copy()
                        doc['similarity_score'] = float(score)
                        results.append(doc)
            
            # Sort by similarity score (descending)
            results.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def get_context(self, query: str, max_tokens: int = 1500) -> str:
        """
        Get relevant context for RAG generation.
        
        Args:
            query: User query
            max_tokens: Maximum tokens in context
            
        Returns:
            Concatenated context string
        """
        # Search for relevant chunks
        results = self.search(query, top_k=10)
        
        if not results:
            return ""
        
        # Build context with token limit
        context_parts = []
        current_tokens = 0
        
        for result in results:
            content = result['content']
            # Rough token estimation (1 token ≈ 4 characters)
            estimated_tokens = len(content) // 4
            
            if current_tokens + estimated_tokens > max_tokens:
                break
            
            context_parts.append({
                'content': content,
                'score': result['similarity_score'],
                'section_type': result.get('section_type'),
                'importance': result.get('importance_score', 0)
            })
            current_tokens += estimated_tokens
        
        # Sort by importance and similarity
        context_parts.sort(key=lambda x: (x['score'], x['importance']), reverse=True)
        
        # Build final context string
        context_strings = []
        for part in context_parts:
            section_info = f" [{part['section_type']}]" if part.get('section_type') else ""
            context_strings.append(f"{part['content']}{section_info}")
        
        return "\n\n".join(context_strings)
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        with self._lock:
            return {
                'total_documents': len(set(doc['doc_id'] for doc in self.documents)),
                'total_chunks': len(self.documents),
                'vector_index_size': self.vector_index.ntotal if self.vector_index else 0,
                'embedding_dimension': self.document_embeddings.shape[1] if self.document_embeddings is not None else 0,
                'database_size_mb': sum(
                    os.path.getsize(path) for path in [
                        self.text_db_path, self.metadata_path, self.vector_db_path
                    ] if path.exists()
                ) / (1024 * 1024),
                'last_updated': self.metadata.get('last_updated', 'Never')
            }
    
    def clear_database(self):
        """Clear all data from the database."""
        with self._lock:
            self.documents = []
            self.metadata = {}
            self.document_embeddings = None
            self.vector_index = None
            
            # Remove files
            for path in [self.text_db_path, self.vector_db_path, self.metadata_path]:
                if path.exists():
                    path.unlink()
            
            logger.info("Database cleared")

# Global RAG database instance
rag_db = RAGDatabase()
