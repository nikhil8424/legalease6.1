# modules/fast_rag_chatbot.py
"""
High-performance RAG chatbot with optimized retrieval and generation.
Features caching, fast vector search, and intelligent response generation.
"""

import re
import time
import logging
from typing import List, Dict, Optional, Tuple
from functools import lru_cache
import threading

from modules.rag_database import rag_db
from modules.cache_manager import document_cache, cached_processing
from modules.model_manager import model_manager
from modules.llm_model import get_llm

logger = logging.getLogger(__name__)

class FastRAGChatbot:
    """
    High-performance RAG chatbot with optimized retrieval and caching.
    """
    
    def __init__(self, max_context_tokens: int = 1500, response_cache_size: int = 1000):
        self.max_context_tokens = max_context_tokens
        self.response_cache_size = response_cache_size
        
        # Threading
        self._lock = threading.RLock()
        
        # Query preprocessing patterns
        self.date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\b',
            r'\b\d{4}\b'
        ]
        
        # Legal entity patterns
        self.legal_patterns = {
            'parties': r'\b(?:plaintiff|defendant|petitioner|respondent|appellant|appellee)\b',
            'courts': r'\b(?:supreme court|high court|district court|magistrate|tribunal)\b',
            'acts': r'\b(indian penal code|code of criminal procedure|code of civil procedure|indian evidence act|indian divorce act|hindu marriage act|motor vehicle act|narcotics and psychotropic substances act)\b',
            'sections': r'\b(?:section|article|clause|sub-section|paragraph)\s+\d+[A-Z]?\b'
        }
        
        # Response templates for different query types
        self.response_templates = {
            'summary': """Based on the document, here's a summary:

**Key Points:**
{key_points}

**Important Dates:** {dates}
**Parties Involved:** {parties}
**Legal References:** {legal_refs}""",
            
            'specific': """**Answer:** {answer}

**Context:** {context}""",
            
            'not_found': "I couldn't find specific information about '{query}' in the available documents. Please try rephrasing your question or asking about topics covered in the uploaded documents."
        }
        
        logger.info("FastRAGChatbot initialized")
    
    @lru_cache(maxsize=100)
    def _extract_query_entities(self, query: str) -> Dict[str, List[str]]:
        """
        Extract entities from query for better context retrieval.
        
        Args:
            query: User query
            
        Returns:
            Dictionary of extracted entities
        """
        entities = {
            'dates': [],
            'legal_terms': [],
            'parties': [],
            'sections': []
        }
        
        query_lower = query.lower()
        
        # Extract dates
        for pattern in self.date_patterns:
            entities['dates'].extend(re.findall(pattern, query, re.IGNORECASE))
        
        # Extract legal entities
        for entity_type, pattern in self.legal_patterns.items():
            matches = re.findall(pattern, query_lower, re.IGNORECASE)
            if matches:
                entities['legal_terms'].extend(matches)
        
        return entities
    
    def _preprocess_query(self, query: str) -> Tuple[str, Dict]:
        """
        Preprocess query for better retrieval.
        
        Args:
            query: Raw user query
            
        Returns:
            Tuple of (processed_query, query_metadata)
        """
        # Clean query
        processed_query = re.sub(r'\s+', ' ', query.strip())
        
        # Extract entities
        entities = self._extract_query_entities(query)
        
        # Determine query type
        query_lower = processed_query.lower()
        query_type = 'specific'
        
        if any(word in query_lower for word in ['summarize', 'summary', 'overview']):
            query_type = 'summary'
        elif any(word in query_lower for word in ['when', 'date', 'time']):
            query_type = 'temporal'
        elif any(word in query_lower for word in ['who', 'party', 'parties']):
            query_type = 'parties'
        elif any(word in query_lower for word in ['what', 'define', 'meaning']):
            query_type = 'definition'
        
        metadata = {
            'type': query_type,
            'entities': entities,
            'length': len(processed_query.split())
        }
        
        return processed_query, metadata
    
    def _get_relevant_context(self, query: str, query_metadata: Dict) -> str:
        """
        Get relevant context with caching and optimization.
        
        Args:
            query: Processed query
            query_metadata: Query metadata
            
        Returns:
            Relevant context string
        """
        # Build enhanced query based on detected entities
        enhanced_query = query
        
        # Check for legal act mentions and enhance query
        query_lower = query.lower()
        if 'ipc' in query_lower or 'indian penal code' in query_lower:
            enhanced_query += " Indian Penal Code IPC criminal law"
        elif 'crpc' in query_lower or 'criminal procedure' in query_lower:
            enhanced_query += " Criminal Procedure CrPC criminal court"
        elif 'cpc' in query_lower or 'civil procedure' in query_lower:
            enhanced_query += " Civil Procedure CPC civil court"
        elif 'evidence' in query_lower:
            enhanced_query += " Indian Evidence Act IEA evidence"
        elif 'motor vehicle' in query_lower or 'mva' in query_lower:
            enhanced_query += " Motor Vehicle Act MVA traffic"
        
        # Add section-specific enhancement
        section_match = re.search(r'section\s+(\d+[A-Z]?)', query_lower)
        if section_match:
            section_num = section_match.group(1)
            enhanced_query += f" Section {section_num}"
        
        # Adjust context retrieval based on query type
        if query_metadata['type'] == 'summary':
            # For summaries, get broader context
            context = rag_db.get_context(enhanced_query, max_tokens=2000)
        elif query_metadata['type'] == 'temporal':
            # For temporal queries, focus on date-related content
            enhanced_query += " date time when period"
            context = rag_db.get_context(enhanced_query, max_tokens=self.max_context_tokens)
        else:
            # Standard context retrieval with enhancement
            context = rag_db.get_context(enhanced_query, max_tokens=self.max_context_tokens)
        
        return context
    
    def _extract_dates_from_context(self, context: str) -> List[str]:
        """Extract dates from context."""
        dates = set()
        for pattern in self.date_patterns:
            matches = re.findall(pattern, context, re.IGNORECASE)
            dates.update(matches)
        return sorted(list(dates))
    
    def _extract_parties_from_context(self, context: str) -> List[str]:
        """Extract parties from context."""
        parties = set()
        
        # Look for legal parties
        party_patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:vs?\.?|versus)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
            r'\b(?:plaintiff|defendant|petitioner|respondent):\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),?\s+(?:plaintiff|defendant|petitioner|respondent)\b'
        ]
        
        for pattern in party_patterns:
            matches = re.findall(pattern, context)
            if matches:
                if isinstance(matches[0], tuple):
                    parties.update([m for m in matches[0] if m])
                else:
                    parties.update(matches)
        
        return list(parties)[:5]  # Limit to top 5 parties
    
    def _extract_legal_references(self, context: str) -> List[str]:
        """Extract legal references from context."""
        references = set()
        
        # Legal reference patterns
        ref_patterns = [
            r'\b(?:Section|Article|Clause)\s+\d+(?:\([a-z]+\))?\b',
            r'\b[A-Z][a-zA-Z\s]+Act,?\s+\d{4}\b',
            r'\b(?:Constitution|Code|Statute)\s+[A-Z][a-zA-Z\s]*\b'
        ]
        
        for pattern in ref_patterns:
            matches = re.findall(pattern, context, re.IGNORECASE)
            references.update(matches)
        
        return list(references)[:10]  # Limit to top 10 references
    
    def _generate_structured_response(self, query: str, context: str, query_metadata: Dict) -> str:
        """
        Generate structured response based on query type.
        
        Args:
            query: User query
            context: Retrieved context
            query_metadata: Query metadata
            
        Returns:
            Generated response
        """
        if not context:
            return self.response_templates['not_found'].format(query=query)
        
        query_type = query_metadata['type']
        
        if query_type == 'summary':
            # Generate structured summary
            dates = self._extract_dates_from_context(context)
            parties = self._extract_parties_from_context(context)
            legal_refs = self._extract_legal_references(context)
            
            # Extract key points (first few sentences from high-relevance chunks)
            sentences = context.split('.')[:5]
            key_points = '\n'.join([f"• {s.strip()}" for s in sentences if len(s.strip()) > 20])
            
            return self.response_templates['summary'].format(
                key_points=key_points or "Document content available for review",
                dates=', '.join(dates) if dates else "No specific dates found",
                parties=', '.join(parties) if parties else "Not specified",
                legal_refs=', '.join(legal_refs) if legal_refs else "No specific legal references"
            )
        
        else:
            # Generate specific answer using LLM
            return self._generate_llm_response(query, context, query_type)
    
    def _generate_llm_response(self, query: str, context: str, query_type: str) -> str:
        """
        Generate response using LLM with optimized prompts and fallback.
        
        Args:
            query: User query
            context: Retrieved context
            query_type: Type of query
            
        Returns:
            Generated response
        """
        # First try to extract direct answer from context for simple queries
        direct_answer = self._try_direct_extraction(query, context)
        if direct_answer:
            return direct_answer
            
        try:
            # Get LLM with lazy loading
            llm = get_llm()
            if llm is None:
                return self._fallback_response(query, context)
            
            # Create simple, focused prompt to avoid repetition
            prompt = f"""Context: {context[:800]}...

Question: {query}

Provide a direct, factual answer based on the context above. If the information is not available, say "Information not found in the document."

Answer:"""
            
            # Generate response using langchain chain
            from langchain.prompts import PromptTemplate
            
            prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template=prompt
            )
            
            chain = prompt_template | llm
            
            # Generate response with strict controls to prevent repetition
            response_obj = chain.invoke({
                "context": context[:800],  # Limit context to prevent confusion
                "question": query
            })
            
            response = str(response_obj).strip()
            
            # Aggressive cleanup to prevent repetition
            response = self._clean_llm_response(response)
            
            # Check for repetitive patterns
            if self._is_repetitive(response):
                logger.warning("Detected repetitive LLM response, using fallback")
                return self._fallback_response(query, context)
            
            return response if response and len(response) > 10 else self._fallback_response(query, context)
                
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return self._fallback_response(query, context)
    
    def _try_direct_extraction(self, query: str, context: str) -> Optional[str]:
        """Try to extract direct answer from context without LLM for simple queries."""
        query_lower = query.lower()
        
        # Handle full form queries directly
        if 'full form' in query_lower or 'stands for' in query_lower or 'meaning' in query_lower:
            # Extract the acronym from query
            acronym_match = re.search(r'\b([A-Z]{2,6})\b', query)
            if acronym_match:
                acronym = acronym_match.group(1)
                
                # Special handling for common legal acronyms
                legal_expansions = {
                    'IPC': 'Indian Penal Code',
                    'CPC': 'Code of Civil Procedure', 
                    'CRPC': 'Code of Criminal Procedure',
                    'IEA': 'Indian Evidence Act',
                    'MVA': 'Motor Vehicle Act',
                    'HMA': 'Hindu Marriage Act',
                    'IDA': 'Indian Divorce Act',
                    'NIA': 'Narcotics and Psychotropic Substances Act'
                }
                
                if acronym in legal_expansions:
                    return f"{acronym} stands for {legal_expansions[acronym]}."
                
                # Look for expansions in context
                context_lines = context.replace('\n', ' ').split('.')
                for line in context_lines:
                    line_upper = line.upper()
                    if acronym in line_upper:
                        # Try to find the full expansion near the acronym
                        words = line.split()
                        for i, word in enumerate(words):
                            if acronym.upper() in word.upper():
                                # Look for capitalized words around the acronym
                                expansion_words = []
                                start = max(0, i-10)
                                end = min(len(words), i+10)
                                for j in range(start, end):
                                    if words[j][0].isupper() and len(words[j]) > 2:
                                        expansion_words.append(words[j])
                                if len(expansion_words) >= 2:
                                    expansion = ' '.join(expansion_words[:4])  # Take first few capitalized words
                                    return f"{acronym} likely stands for {expansion}."
        
        # Handle section queries
        section_match = re.search(r'section\s+(\d+[A-Z]?)', query_lower)
        if section_match:
            section_num = section_match.group(1)
            # Look for section content in context
            lines = context.split('\n')
            for i, line in enumerate(lines):
                if f"Section {section_num}" in line or f"section {section_num}" in line.lower():
                    # Get this line and next few lines
                    result = [line.strip()]
                    for j in range(i+1, min(i + 4, len(lines))):
                        if lines[j].strip():
                            result.append(lines[j].strip())
                        if len(' '.join(result)) > 200:  # Limit length
                            break
                    return ' '.join(result)
        
        # Handle "what is" questions
        if query_lower.startswith('what is'):
            # Extract the subject from the query
            subject_match = re.search(r'what is\s+(.*?)\??$', query_lower)
            if subject_match:
                subject = subject_match.group(1).strip()
                # Look for definitions in context
                sentences = context.split('.')
                for sentence in sentences:
                    if subject in sentence.lower():
                        sentence = sentence.strip()
                        if len(sentence) > 20 and len(sentence) < 300:
                            return sentence + '.'
        
        return None
    
    def _fallback_response(self, query: str, context: str) -> str:
        """Generate fallback response when LLM fails."""
        # Extract first few sentences from most relevant context
        sentences = context.split('.')[:3]
        relevant_text = '. '.join([s.strip() for s in sentences if len(s.strip()) > 20])
        
        if relevant_text:
            return f"Based on the available information: {relevant_text}."
        else:
            return "I found some information in the documents, but I'm having trouble processing it clearly. Please try rephrasing your question."
    
    def _clean_llm_response(self, response: str) -> str:
        """Clean up LLM response to remove artifacts and repetition."""
        # Remove common prefixes
        response = re.sub(r'^(Answer:|Response:|Based on|According to)\s*:?\s*', '', response, flags=re.IGNORECASE)
        
        # Remove instruction artifacts
        response = re.sub(r'(Instructions?|Context|Document|Question)\s*:.*$', '', response, flags=re.MULTILINE | re.IGNORECASE)
        
        # Remove repetitive patterns
        words = response.split()
        cleaned_words = []
        last_word = None
        repeat_count = 0
        
        for word in words:
            if word == last_word:
                repeat_count += 1
                if repeat_count < 2:  # Allow one repeat
                    cleaned_words.append(word)
            else:
                cleaned_words.append(word)
                repeat_count = 0
            last_word = word
        
        response = ' '.join(cleaned_words)
        
        # Remove numbered lists of repeated items
        response = re.sub(r'\d+\.\s*(The document|The form|The content)\s*', '', response, flags=re.IGNORECASE)
        
        # Limit to first 100 words for clarity
        words = response.split()
        if len(words) > 100:
            response = ' '.join(words[:100]) + '...'
        
        return response.strip()
    
    def _is_repetitive(self, response: str) -> bool:
        """Check if response contains repetitive patterns."""
        words = response.lower().split()
        if len(words) < 5:
            return False
        
        # Check for word repetition
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # If any word appears more than 30% of the time, it's repetitive
        for count in word_counts.values():
            if count > len(words) * 0.3:
                return True
        
        # Check for phrase repetition
        phrases = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
        phrase_counts = {}
        for phrase in phrases:
            phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
        
        # If any 3-word phrase appears more than twice, it's repetitive
        for count in phrase_counts.values():
            if count > 2:
                return True
        
        return False
    
    def generate_response(self, query: str) -> str:
        """
        Generate optimized response with full caching.
        
        Args:
            query: User query
            
        Returns:
            Generated response
        """
        start_time = time.time()
        
        try:
            # Validate input
            if not query or not isinstance(query, str) or len(query.strip()) < 3:
                return "Please provide a valid question."
            
            # Preprocess query
            processed_query, query_metadata = self._preprocess_query(query)
            
            # Get relevant context (cached)
            context = self._get_relevant_context(processed_query, query_metadata)
            
            if not context:
                return f"I couldn't find relevant information to answer your question about '{query}'. Please try asking about topics covered in the uploaded documents."
            
            # Generate structured response
            response = self._generate_structured_response(processed_query, context, query_metadata)
            
            # Log performance
            processing_time = time.time() - start_time
            logger.info(f"Generated response in {processing_time:.2f}s for query type: {query_metadata['type']}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I encountered an error while processing your request. Please try again with a different question."
    
    def add_document_to_database(self, text: str, metadata: Optional[Dict] = None) -> str:
        """
        Add document to RAG database.
        
        Args:
            text: Document content
            metadata: Optional metadata
            
        Returns:
            Document ID
        """
        try:
            doc_id = rag_db.add_document(text, metadata)
            logger.info(f"Added document {doc_id} to RAG database")
            return doc_id
        except Exception as e:
            logger.error(f"Error adding document to database: {e}")
            raise
    
    def get_database_stats(self) -> Dict:
        """Get RAG database statistics."""
        return rag_db.get_stats()
    
    def clear_cache(self):
        """Clear response cache."""
        # Clear function-level caches
        self._extract_query_entities.cache_clear()
        
        # Clear document cache for this chatbot
        document_cache.invalidate('rag_')
        
        logger.info("Chatbot cache cleared")

# Global optimized chatbot instance
fast_rag_chatbot = FastRAGChatbot()

# Convenience functions for easy integration
def chat_response(query: str) -> str:
    """Generate chat response - main entry point."""
    return fast_rag_chatbot.generate_response(query)

def add_document(text: str, metadata: Optional[Dict] = None) -> str:
    """Add document to knowledge base."""
    return fast_rag_chatbot.add_document_to_database(text, metadata)

def get_stats() -> Dict:
    """Get system statistics."""
    return fast_rag_chatbot.get_database_stats()
