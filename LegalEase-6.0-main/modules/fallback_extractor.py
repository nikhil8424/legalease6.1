"""
Fallback key information extraction using regex patterns and basic NLP
"""

import re
from collections import defaultdict
import logging

def extract_key_information_fallback(text):
    """
    Fallback extraction using regex patterns and basic NLP
    """
    print("=== Starting Fallback Key Information Extraction ===")

    # Initialize result with default values
    result = {
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

    # Extract title (first line or heading)
    lines = text.split('\n')
    # Extract title (first line or heading)
lines = text.split('\n')
for line in lines[:15]:  # Check first 15 lines instead of 10
    line = line.strip()
    if line and len(line) > 5 and not line.isdigit():  # Reduced minimum length
        # Look for title patterns - more flexible
        if (re.match(r'^[A-Z][A-Z\s&.,-]+$', line) or
            'AGREEMENT' in line.upper() or
            'MEMORANDUM' in line.upper() or
            'OFFICE' in line.upper() or
            'CIRCULAR' in line.upper() or
            'NOTIFICATION' in line.upper()):
            result["title"] = line
            break

# If no title found, try to extract from common patterns
if not result["title"]:
    title_patterns = [
        r'(?:OFFICE\s+)?MEMORANDUM',
        r'(?:OFFICE\s+)?ORDER',
        r'CIRCULAR\s+NO',
        r'NOTIFICATION',
        r'SUBJECT[:\s]*(.+?)(?:\n|$)',
    ]
    for pattern in title_patterns:
        matches = re.findall(pattern, text.upper())
        if matches:
            result["title"] = matches[0].strip()
            break
# Extract parties - more comprehensive patterns
party_patterns = [
    r'PARTY|PARTIES|BETWEEN\s+(.+?)\s+AND\s+(.+?)(?:\s|$)',
    r'(.+?)\s+AND\s+(.+?)\s+(?:AGREEMENT|CONTRACT)',
    r'WITNESSETH\s+THAT\s+(.+?)\s+AND\s+(.+?)',
    r'THE\s+(.+?)\s+AND\s+THE\s+(.+?)',
    r'MINISTRY\s+OF\s+(.+?)(?:\s|$)',
    r'GOVERNMENT\s+OF\s+(.+?)(?:\s|$)',
    r'DEPARTMENT\s+OF\s+(.+?)(?:\s|$)',
]

for pattern in party_patterns:
    matches = re.findall(pattern, text.upper())
    if matches:
        for match in matches:
            if isinstance(match, tuple):
                result["parties"].extend([p.strip() for p in match if p.strip()])
            else:
                result["parties"].append(match.strip())
        break
    # Extract dates
    date_patterns = [
        r'EFFECTIVE\s+DATE[:\s]*([A-Za-z0-9\s,.-]+)',
        r'DATED[:\s]*([A-Za-z0-9\s,.-]+)',
        r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
        r'([A-Za-z]+\s+\d{1,2},?\s+\d{4})',
    ]

    for pattern in date_patterns:
        matches = re.findall(pattern, text.upper())
        if matches:
            result["effective_date"] = matches[0].strip()
            break

    # Extract sections based on keywords
    sections = {
        "purpose_objective": ["PURPOSE", "OBJECTIVE", "WHEREAS", "RECITALS"],
        "obligations_responsibilities": ["OBLIGATIONS", "RESPONSIBILITIES", "DUTIES", "SHALL"],
        "terms_conditions": ["TERMS", "CONDITIONS", "PROVISIONS"],
        "duration_term": ["TERM", "DURATION", "PERIOD"],
        "termination_clause": ["TERMINATION", "CANCELLATION", "EXPIRY"],
        "confidentiality_clause": ["CONFIDENTIAL", "PRIVACY", "NON-DISCLOSURE"],
        "jurisdiction_governing_law": ["GOVERNING LAW", "JURISDICTION", "APPLICABLE LAW"],
        "dispute_resolution": ["DISPUTE", "ARBITRATION", "MEDIATION"],
        "payment_terms": ["PAYMENT", "COMPENSATION", "FEES", "CHARGES"],
        "force_majeure": ["FORCE MAJEURE", "ACT OF GOD"],
        "amendments": ["AMENDMENT", "MODIFICATION", "CHANGE"],
        "signatures_witnesses": ["SIGNATURE", "EXECUTION", "WITNESS"],
        "annexures_attachments_schedules": ["ANNEXURE", "ATTACHMENT", "SCHEDULE", "EXHIBIT"],
        "legal_clauses_boilerplate": ["SEVERABILITY", "NOTICE", "WAIVER", "ENTIRE AGREEMENT"],
    }

    for field, keywords in sections.items():
        for keyword in keywords:
            if keyword in text.upper():
                # Find the sentence/paragraph containing the keyword
                pattern = rf'[^.]*{re.escape(keyword)}[^.]*\.'
                matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
                if matches:
                    result[field] = matches[0].strip()
                    break

    # Extract constitutional articles
    article_pattern = r'Article\s+\d+[A-Za-z]*'
    articles = re.findall(article_pattern, text)
    if articles:
        result["constitutional_articles"] = list(set(articles))

    # Clean up results
    result["parties"] = list(set(result["parties"]))
    for key, value in result.items():
        if isinstance(value, str):
            result[key] = value.strip()

    # Count non-empty fields
    non_empty = {k: v for k, v in result.items() if v}
    print(f"✓ Fallback extraction completed, found {len(non_empty)} fields")

    if non_empty:
        print("\nExtracted Information:")
        for k, v in non_empty.items():
            if isinstance(v, list):
                print(f"\n{k.replace('_', ' ').title()}:")
                for item in v:
                    print(f"- {item}")
            else:
                print(f"\n{k.replace('_', ' ').title()}: {v}")

    return result

def extract_key_information_fixed(text):
    """
    Extract key information with LLM fallback to regex-based extraction
    """
    print("\n=== Starting Key Information Extraction with Fallback ===")
    print(f"Text length: {len(text)}")

    # Try LLM-based extraction first
    try:
        from modules.text_processing_llm_utils import extract_key_information
        llm_result = extract_key_information(text)
        # Check if LLM returned meaningful results (not all empty)
        non_empty_fields = {k: v for k, v in llm_result.items() if v}
        if len(non_empty_fields) > 2:  # If we got more than 2 meaningful fields
            print(f"✓ LLM extraction successful, found {len(non_empty_fields)} fields")
            return llm_result
        else:
            print(f"⚠ LLM extraction returned only {len(non_empty_fields)} fields, using fallback")
    except Exception as e:
        print(f"⚠ LLM extraction failed: {str(e)}, using fallback")

    # Fallback to regex-based extraction
    print("Using regex-based fallback extraction...")
    return extract_key_information_fallback(text)