#!/usr/bin/env python3
"""
Initialize RAG Database with Legal Documents
Loads all legal documents from JSON files into the optimized RAG database.
"""

import os
import json
import logging
from pathlib import Path
from modules.rag_database import rag_db

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_legal_document(json_path: str) -> dict:
    """Load and parse legal document from JSON file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Error loading {json_path}: {e}")
        return {}

def extract_text_from_legal_json(data: dict, document_name: str) -> str:
    """
    Extract readable text from legal JSON structure.
    
    Args:
        data: Legal document JSON data
        document_name: Name of the legal document
        
    Returns:
        Formatted text content
    """
    text_parts = []
    
    # Add document title/name
    if document_name:
        text_parts.append(f"=== {document_name.upper()} ===\n")
    
    # Handle different JSON structures
    if isinstance(data, list):
        # Array of sections/articles
        for item in data:
            if isinstance(item, dict):
                # Extract section information
                section_text = ""
                if 'section' in item:
                    section_text += f"Section {item['section']}: "
                elif 'article' in item:
                    section_text += f"Article {item['article']}: "
                elif 'rule' in item:
                    section_text += f"Rule {item['rule']}: "
                
                # Extract title/heading
                if 'title' in item:
                    section_text += f"{item['title']}\n"
                elif 'heading' in item:
                    section_text += f"{item['heading']}\n"
                elif 'name' in item:
                    section_text += f"{item['name']}\n"
                
                # Extract description/content
                if 'description' in item:
                    section_text += f"{item['description']}\n"
                elif 'content' in item:
                    section_text += f"{item['content']}\n"
                elif 'text' in item:
                    section_text += f"{item['text']}\n"
                
                # Extract punishment/penalty if available
                if 'punishment' in item:
                    section_text += f"Punishment: {item['punishment']}\n"
                elif 'penalty' in item:
                    section_text += f"Penalty: {item['penalty']}\n"
                
                # Extract fine information
                if 'fine' in item:
                    section_text += f"Fine: {item['fine']}\n"
                
                # Extract imprisonment information  
                if 'imprisonment' in item:
                    section_text += f"Imprisonment: {item['imprisonment']}\n"
                
                # Extract additional details
                if 'details' in item:
                    section_text += f"Details: {item['details']}\n"
                
                text_parts.append(section_text + "\n")
    
    elif isinstance(data, dict):
        # Object with sections
        for key, value in data.items():
            if isinstance(value, (list, dict)):
                # Recursive extraction for nested structures
                nested_text = extract_text_from_legal_json(value, key)
                text_parts.append(nested_text)
            elif isinstance(value, str) and len(value) > 10:
                text_parts.append(f"{key}: {value}\n")
    
    return "\n".join(text_parts)

def initialize_database():
    """Initialize RAG database with all legal documents."""
    logger.info("Starting RAG database initialization with legal documents...")
    
    # Get legal documents directory
    legal_docs_dir = Path("Indian-Law-Penal-Code-Json-main")
    
    if not legal_docs_dir.exists():
        logger.error(f"Legal documents directory not found: {legal_docs_dir}")
        return
    
    # Document mappings
    document_names = {
        'ipc.json': 'Indian Penal Code (IPC)',
        'crpc.json': 'Code of Criminal Procedure (CrPC)', 
        'cpc.json': 'Code of Civil Procedure (CPC)',
        'iea.json': 'Indian Evidence Act (IEA)',
        'ida.json': 'Indian Divorce Act (IDA)',
        'hma.json': 'Hindu Marriage Act (HMA)',
        'nia.json': 'Narcotics and Psychotropic Substances Act (NIA)',
        'MVA.json': 'Motor Vehicle Act (MVA)'
    }
    
    total_documents = 0
    successful_loads = 0
    
    # Load each legal document
    for json_file, doc_name in document_names.items():
        json_path = legal_docs_dir / json_file
        
        if not json_path.exists():
            logger.warning(f"Document not found: {json_path}")
            continue
        
        logger.info(f"Loading {doc_name} from {json_file}...")
        
        try:
            # Load JSON data
            legal_data = load_legal_document(str(json_path))
            if not legal_data:
                logger.warning(f"No data found in {json_file}")
                continue
            
            # Extract readable text
            document_text = extract_text_from_legal_json(legal_data, doc_name)
            
            if not document_text.strip():
                logger.warning(f"No readable text extracted from {json_file}")
                continue
            
            # Add to RAG database
            metadata = {
                'source_file': json_file,
                'document_name': doc_name,
                'document_type': 'legal_act',
                'language': 'english',
                'loaded_from': 'legal_json_collection'
            }
            
            doc_id = rag_db.add_document(document_text, metadata)
            logger.info(f"Added {doc_name} to database with ID: {doc_id}")
            successful_loads += 1
            
        except Exception as e:
            logger.error(f"Error processing {json_file}: {e}")
        
        total_documents += 1
    
    # Get final statistics
    stats = rag_db.get_stats()
    
    logger.info(f"""
=== RAG Database Initialization Complete ===
Total documents processed: {total_documents}
Successfully loaded: {successful_loads}
Final database stats:
- Total documents: {stats['total_documents']}
- Total chunks: {stats['total_chunks']}
- Vector index size: {stats['vector_index_size']}
- Database size: {stats['database_size_mb']:.2f} MB
""")
    
    if successful_loads > 0:
        logger.info("RAG database is now ready for chatbot queries!")
        
        # Test the database with a sample query
        logger.info("Testing database with sample query...")
        context = rag_db.get_context("theft punishment", max_tokens=200)
        if context:
            logger.info("✅ Database test successful - context retrieved!")
        else:
            logger.warning("❌ Database test failed - no context retrieved")
    else:
        logger.error("❌ No documents were successfully loaded!")

if __name__ == "__main__":
    initialize_database()
