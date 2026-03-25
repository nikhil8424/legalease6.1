import os
import re
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import logging
from .llm_model import llama_model, llama_tokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGChatbot:
    def __init__(self, database_path='uploads/database/db.txt'):
        """
        Initialize RAG Chatbot with database
        
        Args:
            database_path (str): Path to the text database
        """
        self.database_path = database_path
        
        # Load embedding model for retrieval
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.error(f'Failed to load embedding model: {e}')
            raise
        
        # Load text database
        self.database_text = self._load_database()
        self.database_embeddings = self._embed_database()
        
        # Use models from llm_model.py
        self.model = llama_model
        self.tokenizer = llama_tokenizer
        
        if self.model is None or self.tokenizer is None:
            logger.error('Language model not available')
            raise RuntimeError('Language model initialization failed')
    
    def _load_database(self):
        """Load text from database file"""
        try:
            os.makedirs(os.path.dirname(self.database_path), exist_ok=True)
            with open(self.database_path, 'r', encoding='utf-8') as f:
                # Read the entire file as a single string
                text = f.read().strip()
                return [text] if text else []
        except FileNotFoundError:
            logger.warning(f'Database file not found at {self.database_path}. Creating an empty database.')
            with open(self.database_path, 'w', encoding='utf-8') as f:
                pass
            return []
        except Exception as e:
            logger.error(f'Error loading database: {e}')
            return []
    
    def _embed_database(self):
        """Create embeddings for database text"""
        return self.embedding_model.encode(self.database_text)
    
    def _retrieve_context(self, query, top_k=3):
        """
        Retrieve most relevant context from database using semantic search
        
        Args:
            query (str): User's query
            top_k (int): Number of top contexts to retrieve
        
        Returns:
            List of most relevant text contexts
        """
        if not self.database_text:
            return []
        
        # Embed query
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Compute cosine similarities
        similarities = np.dot(self.database_embeddings, query_embedding)
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        return [self.database_text[i] for i in top_indices]
    
    import re
    def _extract_dates(self, text):
        # Extract dates in common formats
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # 31-12-2024, 04/11/2024
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)[ ,.-]*\d{1,2}[, ]*\d{4}\b',
            r'\b\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\b',
            r'\b\d{4}\b'  # just years
        ]
        dates = set()
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            dates.update(matches)
        return sorted(dates)

    def generate_response(self, query):
        """Generate context-aware response for various queries"""
        if not query or not isinstance(query, str):
            return "Please provide a valid question."
            
        if self.model is None or self.tokenizer is None:
            return "The language model is not available. Please try again later."
            
        try:
            # Get relevant context
            context = self._retrieve_context(query)
            if not context:
                return "I couldn't find any relevant information to answer your question. Please try rephrasing or asking a different question."
                
            context_text = ' '.join(context)
            
            # Preprocess query
            query = query.strip()
            
            # Construct prompt based on query type
            if any(word in query.lower() for word in ['summarize', 'summary']):
                dates = self._extract_dates(context_text)
                dates_str = ', '.join(dates) if dates else 'No explicit dates found.'
                prompt = (
                    f"You are a legal assistant. Analyze this document and provide a clear, structured summary:\n\n"
                    f"Document dates: {dates_str}\n"
                    f"Document content:\n{context_text}\n\n"
                    f"Instructions:\n"
                    f"1. Extract and list the key points\n"
                    f"2. Highlight important dates and deadlines\n"
                    f"3. Identify main parties or entities involved\n"
                    f"4. Note any specific requirements or conditions\n"
                )
            else:
                prompt = (
                    f"You are a legal document assistant. Based on the following document, answer the question DIRECTLY and CONCISELY.\n\n"
                    f"Document:\n{context_text}\n\n"
                    f"Question: {query}\n\n"
                    f"Rules for answering:\n"
                    f"1. If the answer is directly stated in the document, start with 'Answer: ' followed by the exact information\n"
                    f"2. If the answer requires combining multiple parts of the document, start with 'Answer: ' and clearly explain\n"
                    f"3. If the answer is not in the document, respond ONLY with 'The document does not contain information about [topic].'\n"
                    f"4. Keep responses under 50 words unless absolutely necessary\n"
                    f"5. Do not include any disclaimers or additional explanations unless specifically asked\n"
                    f"6. Do not restate or summarize the question"
                )
                        # Generate response with error handling
            try:
                # Encode prompt
                prompt_ids = self.tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=True)
                
                # Generate with stricter parameters for more focused responses
                output_ids = self.model.generate(
                    prompt_ids,
                    max_new_tokens=100,  # Limit length for conciseness
                    num_return_sequences=1,
                    no_repeat_ngram_size=3,
                    temperature=0.2,  # Lower temperature for more deterministic responses
                    top_p=0.95,
                    top_k=30,  # Limit vocabulary for more focused responses
                    do_sample=False,  # Deterministic generation
                    pad_token_id=self.tokenizer.eos_token_id,
                    attention_mask=torch.ones_like(prompt_ids),
                    eos_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True,
                    repetition_penalty=1.2  # Discourage repetition
                )
                
                # Extract only the new tokens
                response = self.tokenizer.decode(
                    output_ids[0][prompt_ids.shape[1]:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                ).strip()
                
                # Clean up the response
                if response.lower().startswith('answer:'):
                    response = response[7:].strip()
                
                # Ensure response isn't too long
                if len(response.split()) > 50:
                    response = ' '.join(response.split()[:50]) + '...'
                
                if not response:
                    return "I apologize, but I couldn't generate a meaningful response. Please try rephrasing your question."
                    
                return response
                
            except Exception as e:
                logger.error(f'Model generation error: {e}')
                return "I encountered an error while generating the response. Please try again with a simpler question."
                
        except Exception as e:
            logger.error(f'Response generation error: {e}')
            return "Sorry, I'm having trouble processing your request. Please try again later."

    
    def update_database(self, new_text):
        """
        Update database with new text and re-embed

        Args:
            new_text (str): New text to add to database
        """
        try:
            with open(self.database_path, 'a', encoding='utf-8') as f:
                f.write(new_text + '\n')
            
            # Reload and re-embed database
            self.database_text = self._load_database()
            self.database_embeddings = self._embed_database()
            
            logger.info('Database updated successfully')
        except Exception as e:
            logger.error(f'Failed to update database: {e}')

# Create a singleton instance
# Llama-2-7b will be loaded from the default path in __init__
rag_chatbot = RAGChatbot()  # Using local Llama-2-7B for improved performance