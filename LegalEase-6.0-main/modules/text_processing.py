# modules/text_processing.py
"""
This module now supports advanced LLM-based key information extraction via:
    from modules.text_processing_llm_utils import extract_key_information, extract_unique_articles
"""
#key piint
from langchain.prompts import PromptTemplate
from modules.llm_model import get_llm, _ensure_models_loaded
import re
import spacy
from collections import defaultdict
from nltk.corpus import wordnet
import json
from modules.text_processing_llm_utils import extract_key_information, extract_unique_articles
from modules.cache_manager import cached_processing, document_cache
from modules.text_chunker import chunk_text_for_processing, default_chunker
from modules.model_manager import model_manager
import logging

# AI Model Imports
from modules.llm_model import llama_model, llama_tokenizer
# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def _get_default_key_info():
    """Return default key information structure when LLM fails."""
    return {
        "title": "",
        "parties": [],
        "effective_date": "",
        "definitions": "",
        "purpose_objective": "",
        "obligations_responsibilities": "",
        "terms_conditions": "",
        "duration_term": "",
        "termination_clause": "",
        "confidentiality_clause": "",
        "jurisdiction_governing_law": "",
        "signatures_witnesses": "",
        "dispute_resolution": "",
        "payment_terms": "",
        "force_majeure": "",
        "amendments": "",
        "annexures_attachments_schedules": "",
        "legal_clauses_boilerplate": "",
        "constitutional_articles": []
    }


@cached_processing('extract_key_info')
def extract_key_information(text):
    """Optimized key information extraction with caching and smart chunking."""
    print("\n=== Starting Optimized Key Information Extraction ===")
    print(f"Text length: {len(text)}")
    logging.info(f"Starting key information extraction for text of length {len(text)}")
    
    # Get the LLM with lazy loading
    print("Getting LLM (lazy loading)...")
    llm = get_llm()
    if llm is None:
        print("Error: Failed to initialize LLM")
        logging.error("Failed to initialize LLM")
        return _get_default_key_info()
    
    print("LLM ready - using optimized chunking...")
    # Use optimized chunking for key extraction
    chunks = chunk_text_for_processing(text, processing_type="key_extraction")
    
    print(f"Processing {len(chunks)} optimized chunks for key information")
    key_info_dicts = []
    articles_list = []
    
    # Process optimized chunks for key information  
    print("Processing optimized chunks for key information...\n")
    chunk_results = []
    for i, text_chunk in enumerate(chunks, 1):  
        chunk_content = text_chunk.content
        status = f"Processing chunk {i} of {len(chunks)} (importance: {text_chunk.importance_score:.1f})..."
        print(status)
        print(f"Chunk length: {len(chunk_content)} characters")
        
        # Create prompt for key information
        print("Sending chunk to LLM for analysis...")
        key_info_template = """
Analyze this document chunk and extract key information in JSON format. You must ONLY output a valid JSON object, nothing else. No explanations, no notes, just the JSON.

Document chunk to analyze:
{text}

Expected JSON format (replace values but keep the exact same structure):
{{
    "title": "Office Memorandum",
    "parties": ["Ministry of Home Affairs", "Government of India"],
    "effective_date": "January 1, 2025",
    "definitions": "MHA - Ministry of Home Affairs",
    "purpose_objective": "Issue of parking labels for 2025",
    "obligations_responsibilities": "",
    "terms_conditions": "",
    "duration_term": "Valid until December 31, 2025",
    "termination_clause": "",
    "confidentiality_clause": "",
    "jurisdiction_governing_law": "",
    "signatures_witnesses": "",
    "dispute_resolution": "",
    "payment_terms": "",
    "force_majeure": "",
    "amendments": "",
    "annexures_attachments_schedules": "",
    "legal_clauses_boilerplate": ""  
}}
"""

        articles_template = """
List any Indian Constitution articles mentioned in this text (e.g. "Article 14", "Article 21"). If none, respond "No constitutional articles found."

Text:
{text}
"""
        key_info_prompt = PromptTemplate(
            input_variables=["text"],
            template=key_info_template
        )
        # Ensure the chain is created with the current LLM
        key_info_chain = key_info_prompt | llm
        # Create prompt for constitutional articles
        articles_template = """
You are a legal expert assistant. Identify any references to articles from the Indian Constitution in the following text.
Only extract actual article references (e.g., "Article 14", "Article 21", etc.), not implied references.
Format your response as a simple list of the articles mentioned. If no articles are mentioned, respond with "No constitutional articles found."

Legal text:
{text}

Constitutional Articles Mentioned:
"""
        articles_prompt = PromptTemplate(
            input_variables=["text"],
            template=articles_template
        )
        articles_chain = articles_prompt | llm
        try:
            # Extract structured key information
            print("Preparing document chunk...")
            try:
                key_info_json = key_info_chain.invoke({
                    "text": chunk_content
                })
                print("Received response from LLM")
                
                if not isinstance(key_info_json, str):
                    print(f"Warning: Unexpected response type from LLM: {type(key_info_json)}")
                    key_info_json = str(key_info_json)
            except Exception as e:
                print(f"Error getting LLM response: {str(e)}")
                logging.error(f"Error getting LLM response: {str(e)}")
                key_info_json = "{}"
            print("Processing LLM response...")
            try:
                # First clean up any markdown formatting and find the JSON
                if isinstance(key_info_json, str):
                    key_info_json = key_info_json.strip()
                    
                    # Find JSON content between curly braces
                    start = key_info_json.find('{')
                    end = key_info_json.rfind('}')
                    
                    if start >= 0 and end > start:
                        key_info_json = key_info_json[start:end + 1].strip()
                    else:
                        print("Warning: No JSON object found in response")
                        key_info_json = "{}"
                else:
                    print(f"Warning: LLM response is not a string: {type(key_info_json)}")
                    key_info_json = "{}"
                
                print("Attempting to parse JSON...")
                try:
                    key_info_dict = json.loads(key_info_json)
                    print("Successfully parsed JSON response")
                    if not isinstance(key_info_dict, dict):
                        print("Warning: Parsed JSON is not a dictionary")
                        key_info_dict = {}
                    
                    # Store chunk result
                    chunk_results.append({
                        'chunk_number': i,
                        'chunk_text': chunk_content[:200] + '...' if len(chunk_content) > 200 else chunk_content,
                        'extracted_info': key_info_dict
                    })
                except json.JSONDecodeError as je:
                    print(f"Error: Failed to parse JSON: {je}")
                    logging.error(f"JSON decode error: {je}. Response: {key_info_json}")
                    key_info_dict = {}
            except Exception as e:
                print(f"Error processing LLM response: {str(e)}")
                logging.error(f"Error processing LLM response: {str(e)}")
                key_info_dict = {}
            
            # Clean up the JSON response
            try:
                # Initialize with default values
                expected_keys = [
                    "title", "parties", "effective_date", "definitions", "purpose_objective",
                    "obligations_responsibilities", "terms_conditions", "duration_term", "termination_clause",
                    "confidentiality_clause", "jurisdiction_governing_law", "signatures_witnesses", "dispute_resolution",
                    "payment_terms", "force_majeure", "amendments", "annexures_attachments_schedules", "legal_clauses_boilerplate"
                ]
                
                # Initialize cleaned dict with default values
                cleaned_dict = {k: [] if k == "parties" else "" for k in expected_keys}
                
                # Update with any valid values from the response
                if isinstance(key_info_dict, dict):
                    for k, v in key_info_dict.items():
                        k = str(k).strip().lower()
                        if k in cleaned_dict:
                            if k == "parties":
                                if isinstance(v, list):
                                    cleaned_dict[k].extend(x.strip('"').strip() if isinstance(x, str) else str(x) for x in v if x)
                                elif isinstance(v, str) and v.strip():
                                    cleaned_dict[k].append(v.strip('"').strip())
                            else:
                                if isinstance(v, str):
                                    v = v.strip('"').strip()
                                    if v:  # Only update if non-empty
                                        cleaned_dict[k] = v
                                else:
                                    v = str(v)
                                    if v and v != 'None':
                                        cleaned_dict[k] = v
                
                key_info_dict = cleaned_dict
                non_empty = {k: v for k, v in key_info_dict.items() if v}
                print(f"Found {len(non_empty)} non-empty fields")
                if non_empty:
                    print("Non-empty fields:")
                    for k, v in non_empty.items():
                        print(f"- {k}: {v[:100]}{'...' if len(str(v)) > 100 else ''}")
                
                # Add to key_info_dicts list
                key_info_dicts.append(key_info_dict)
            except Exception as e:
                print(f"Error cleaning JSON response: {str(e)}")
                key_info_dict = {k: [] if k == "parties" else "" for k in expected_keys}
            
            logging.debug(f"Raw LLM output (key_info_json): {key_info_json}")
            
            # Initialize with default values for this chunk
            expected_keys = [
                "title", "parties", "effective_date", "definitions", "purpose_objective",
                "obligations_responsibilities", "terms_conditions", "duration_term", "termination_clause",
                "confidentiality_clause", "jurisdiction_governing_law", "signatures_witnesses", "dispute_resolution",
                "payment_terms", "force_majeure", "amendments", "annexures_attachments_schedules", "legal_clauses_boilerplate"
            ]
            
            default_dict = {k: [] if k == "parties" else "" for k in expected_keys}
            print("Default values initialized")
            
            # Update with any valid values from the response
            if isinstance(key_info_dict, dict):
                for k in expected_keys:
                    if k in key_info_dict and key_info_dict[k]:
                        default_dict[k] = key_info_dict[k]
            
            key_info_dict = default_dict
            
            # Remove any quotes from values
            for k, v in key_info_dict.items():
                if isinstance(v, str):
                    key_info_dict[k] = v.strip('"').strip()
            print(f"Found {len(key_info_dict)} fields in response")
            
            # Ensure all expected keys are present with proper default values
            expected_keys = [
                "title", "parties", "effective_date", "definitions", "purpose_objective",
                "obligations_responsibilities", "terms_conditions", "duration_term", "termination_clause",
                "confidentiality_clause", "jurisdiction_governing_law", "signatures_witnesses", "dispute_resolution",
                "payment_terms", "force_majeure", "amendments", "annexures_attachments_schedules", "legal_clauses_boilerplate"
            ]
            
            # Initialize with default values
            print("Initializing default values...")
            default_dict = {k: [] if k == "parties" else "" for k in expected_keys}
            
            # Update with any valid values from the response
            print("Updating with extracted values...")
            if isinstance(key_info_dict, dict):
                for k in expected_keys:
                    if k in key_info_dict and key_info_dict[k]:
                        default_dict[k] = key_info_dict[k]
                        print(f"Updated {k}")
            
            key_info_dict = default_dict
            key_info_dicts.append(key_info_dict)
            logging.debug(f"Final processed keys: {list(key_info_dict.keys())}")
            
            # Extract constitutional articles
            print("\nExtracting constitutional articles...")
            try:
                articles = articles_chain.invoke({"text": chunk_content})
                if isinstance(articles, str):
                    if "No constitutional articles found" not in articles:
                        articles_list.append(articles.strip())
                        print("Found constitutional articles")
                    else:
                        print("No constitutional articles found in this chunk")
                else:
                    print(f"Unexpected response type from LLM: {type(articles)}")
            except Exception as e:
                print(f"Error extracting constitutional articles: {str(e)}")
                logging.error(f"Error extracting constitutional articles: {str(e)}")
        except Exception as e:
            print(f"Error processing chunk {i}: {str(e)}")
            logging.error(f"Error processing chunk {i}: {str(e)}")
            key_info_dict = {}
    # Process all extracted articles to remove duplicates
    print("\nProcessing extracted information...")
    unique_articles = extract_unique_articles(articles_list)
    print("Merging extracted information...")
    
    # Initialize final info with default values
    final_info = {
        "title": "",
        "parties": [],
        "effective_date": "",
        "definitions": "",
        "purpose_objective": "",
        "obligations_responsibilities": "",
        "terms_conditions": "",
        "duration_term": "",
        "termination_clause": "",
        "confidentiality_clause": "",
        "jurisdiction_governing_law": "",
        "signatures_witnesses": "",
        "dispute_resolution": "",
        "payment_terms": "",
        "force_majeure": "",
        "amendments": "",
        "annexures_attachments_schedules": "",
        "legal_clauses_boilerplate": ""
    }
    
    # Merge information from all chunks
    for info_dict in key_info_dicts:
        if not isinstance(info_dict, dict):
            continue
            
        for key, value in info_dict.items():
            if not value:  # Skip empty values
                continue
                
            if key == "parties":  # Special handling for parties list
                if isinstance(value, list):
                    final_info["parties"].extend(x for x in value if x not in final_info["parties"])
            elif key in final_info and not final_info[key]:  # Only update if current value is empty
                final_info[key] = value
    
    # Count non-empty fields
    non_empty = sum(1 for v in final_info.values() if v)
    print(f"Found {non_empty} non-empty fields in final merged result")
    
    # Add constitutional articles
    final_info["constitutional_articles"] = unique_articles

    # Process and return the final information
    non_empty = {k: v for k, v in final_info.items() if v}
    print(f"\nExtracted {len(non_empty)} fields with content")
    
    if non_empty:
        print("\nExtracted Information:")
        for k, v in non_empty.items():
            if isinstance(v, list):
                print(f"\n{k.replace('_', ' ').title()}:")
                for item in v:
                    print(f"- {item}")
            else:
                print(f"\n{k.replace('_', ' ').title()}: {v}")
    
    return final_info

def extract_laws_and_sections(text):
    pattern = re.compile(r'\bSection\s+\d+\b', re.IGNORECASE)
    sections = pattern.findall(text)
    return sections

def extract_key_points_with_llama(text, max_points=5):
    """Extract key points from text using the modern LLM via get_llm()"""
    try:
        llm = get_llm()
        if llm is None:
            return fallback_key_point_extraction(text, max_points)

        prompt = f"Extract {max_points} key points from the following text as a list:\n\n{text}"
        response = llm.invoke(prompt)

        if isinstance(response, str):
            lines = response.strip().split('\n')
            key_points = [re.sub(r'^[\d\.\-\*\s]*', '', line).strip() for line in lines if line.strip()]
            return key_points[:max_points]
        return fallback_key_point_extraction(text, max_points)
            
    except Exception as e:
        logging.error(f'LLM key point extraction failed: {e}')
        return fallback_key_point_extraction(text, max_points)

def fallback_key_point_extraction(text, max_points=5):
    """Fallback method to extract key points when LLM is not available"""
    try:
        # Simple fallback: split into sentences and return first few
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences[:max_points] if s.strip()]
    except:
        return ["Key point extraction failed."]

def fallback_key_point_extraction(text, max_points=5):
    """Fallback method to extract key points using basic NLP techniques"""
    doc = nlp(text)
    
    # Extract sentences with high importance based on noun chunks and verb actions
    important_sentences = []
    for sent in doc.sents:
        noun_chunks = list(sent.noun_chunks)
        verb_count = len([token for token in sent if token.pos_ == 'VERB'])
        
        if noun_chunks or verb_count > 1:
            important_sentences.append(sent.text)
    
    return important_sentences[:max_points]

def analyze_document_structure(text):
    """Analyze document structure and formatting patterns"""
    structure_features = {
        'has_title': bool(re.search(r'^\s*[A-Z][A-Z\s]+\s*$', text[:100], re.MULTILINE)),
        'has_date': bool(re.search(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', text)),
        'has_signature_block': bool(re.search(r'(?:signed|witnessed|executed)\s+by', text.lower())),
        'has_whereas_clauses': bool(re.search(r'whereas', text.lower())),
        'has_defined_terms': bool(re.search(r'defined\s+terms', text.lower())),
        'has_consideration': bool(re.search(r'consideration', text.lower())),
        'has_termination': bool(re.search(r'termination', text.lower())),
        'has_governing_law': bool(re.search(r'governing\s+law', text.lower())),
        'has_arbitration': bool(re.search(r'arbitration', text.lower())),
        'has_force_majeure': bool(re.search(r'force\s+majeure', text.lower()))
    }
    return structure_features

def identify_document_type(text, legal_keywords):
    try:
        # Convert text to lowercase for case-insensitive matching
        text = text.lower()
        
        # Initialize keyword counts
        keyword_counts = defaultdict(int)
        
        # Count occurrences of keywords for each document type
        for doc_type, keywords in legal_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text:
                    keyword_counts[doc_type] += 1
        
        # Extract sections from the text
        sections = extract_laws_and_sections(text)
        
        # Determine the most likely document type
        if keyword_counts:
            max_type = max(keyword_counts, key=keyword_counts.get)
            return max_type, sections, keyword_counts
        else:
            return 'Unknown Document Type', sections, keyword_counts
            
    except Exception as e:
        # If any error occurs, return unknown type
        return 'Unknown Document Type', [], defaultdict(int)

def extract_dates(text):
    date_patterns = [
        r"(?:January|February|March|April|May|June|July|August|September|October|November|December)[\s\n]*\d{1,2},[\s\n]*\d{4}",
        r"\d{1,2}[\s\n]*(?:January|February|March|April|May|June|July|August|September|October|November|December)[,\s\n]*\d{4}",
        r"\d{4}[-/]\d{2}[-/]\d{2}",
        r"\d{1,2}[-/]\d{2}[-/]\d{4}",
        r"\d{2}[-/]\d{2}[-/]\d{4}",
        r"\d{1,2}[\s\n]*(?:January|February|March|April|May|June|July|August|September|October|November|December)[\s\n]*\d{4}",
    ]
    
    dates_found = set()
    for pattern in date_patterns:
        dates_found.update(match.group() for match in re.finditer(pattern, text))
    return list(dates_found)

def extract_names(text):
    doc = nlp(text)
    filtered_names = set()
    
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text.strip()
            normalized_name = ' '.join(name.split())
            if len(normalized_name.split()) > 1 and not re.match(r'\b(?:This|Witness|Whereof|Sealed|Witnesseth|Principal|a|b|c|d)\b', normalized_name):
                filtered_names.add(normalized_name)
    
    filtered_names = list(filtered_names)
    filtered_names = [name for name in filtered_names if re.match(r'^[A-Za-z\s]+$', name)]
    return filtered_names

def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text