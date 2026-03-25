# Utility LLM-based extraction functions for text_processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from modules.llm_model import get_llm
import logging
import json

def extract_key_information(text):
    print("\n=== Starting Key Information Extraction ===")
    print(f"Text length: {len(text)}")
    logging.info(f"Starting key information extraction for text of length {len(text)}")
    
    # Get the LLM
    print("Initializing LLM...")
    llm = get_llm()
    if llm is None:
        print("Error: Failed to initialize LLM")
        logging.error("Failed to initialize LLM")
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
    print("LLM initialized successfully")
    print("Starting text analysis...")
    # Split text if it's too long
    print("Splitting text into manageable chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,  # Further reduced chunk size
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    # Process only first few chunks for key info
    key_info_chunks = chunks[:2] if len(chunks) > 3 else chunks
    print(f"Processing {len(key_info_chunks)} chunks for key information")
    key_info_dicts = []
    articles_list = []
    
    # Process chunks for key information
    print("Processing chunks for key information...\n")
    chunk_results = []
    for i, chunk in enumerate(key_info_chunks, 1):  
        status = f"Processing chunk {i} of {len(key_info_chunks)}..."
        print(status)
        print(f"Chunk length: {len(chunk)} characters")
        
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
                    "text": chunk
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
                        'chunk_text': chunk[:200] + '...' if len(chunk) > 200 else chunk,
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
                articles = articles_chain.invoke({"text": chunk})
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

def extract_unique_articles(articles_list):
    """
    Process articles lists to extract unique article mentions
    Args:
        articles_list (list): List of articles text blocks
        list: Unique article mentions
    """
    all_articles = []
    for articles_text in articles_list:
        lines = articles_text.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("- Article") or line.startswith("Article"):
                # Clean up the article mention
                article = line.replace("- ", "").strip()
                if article not in all_articles:
                    all_articles.append(article)
    return all_articles
