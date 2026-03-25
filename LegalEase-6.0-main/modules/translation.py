# modules/translation.py
from deep_translator import GoogleTranslator, MyMemoryTranslator, LibreTranslator
from typing import Optional, Dict, List, Tuple
import logging
import time
from functools import lru_cache
import re
from langdetect import detect, DetectorFactory
import concurrent.futures
from tqdm import tqdm
from modules.config import LANGUAGE_MAP
import os
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set seed for consistent language detection
DetectorFactory.seed = 0

class TranslationError(Exception):
    """Custom exception for translation errors"""
    pass

class TranslationService:
    def __init__(self):
        self.services = {
            'google': GoogleTranslator,
            'mymemory': MyMemoryTranslator,
            'libre': LibreTranslator
        }
        self.max_retries = 3
        self.retry_delay = 2  # seconds
        self.max_batch_size = 1000  # characters
        self.cache_size = 1000  # number of translations to cache

    @lru_cache(maxsize=1000)
    def detect_language(self, text: str) -> str:
        """Detect the language of the text"""
        try:
            if not text.strip():
                return 'en'
            return detect(text)
        except Exception as e:
            logger.warning(f"Language detection failed: {str(e)}")
            return 'en'

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks while preserving paragraphs"""
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) < self.max_batch_size:
                current_chunk += paragraph + '\n\n'
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + '\n\n'
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    def translate_with_retry(self, text: str, target_lang: str = 'en', 
                           source_lang: Optional[str] = None,
                           service: str = 'google') -> str:
        """Translate text with retry mechanism"""
        if not text.strip():
            return text

        if not source_lang:
            source_lang = self.detect_language(text)

        if source_lang == target_lang:
            return text

        translator_class = self.services.get(service)
        if not translator_class:
            raise TranslationError(f"Invalid translation service: {service}")

        for attempt in range(self.max_retries):
            try:
                translator = translator_class(source=source_lang, target=target_lang)
                return translator.translate(text)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise TranslationError(f"Translation failed after {self.max_retries} attempts: {str(e)}")
                logger.warning(f"Translation attempt {attempt + 1} failed: {str(e)}")
                time.sleep(self.retry_delay)

    def translate_batch(self, texts: List[str], target_lang: str = 'en',
                       source_lang: Optional[str] = None,
                       service: str = 'google') -> List[str]:
        """Translate a batch of texts in parallel"""
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for text in texts:
                future = executor.submit(
                    self.translate_with_retry,
                    text,
                    target_lang,
                    source_lang,
                    service
                )
                futures.append(future)
            
            for future in tqdm(concurrent.futures.as_completed(futures), 
                             total=len(futures),
                             desc="Translating"):
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error(f"Translation failed: {str(e)}")
                    results.append("")
        
        return results

    def translate_text(self, text: str, target_lang: str = 'en',
                      source_lang: Optional[str] = None,
                      service: str = 'google') -> str:
        """Main translation function with progress tracking"""
        try:
            if not text.strip():
                return text

            # Split text into manageable chunks
            chunks = self.split_text(text)
            logger.info(f"Split text into {len(chunks)} chunks for translation")

            # Translate chunks in parallel
            translated_chunks = self.translate_batch(
                chunks,
                target_lang,
                source_lang,
                service
            )

            # Combine translated chunks
            return '\n\n'.join(translated_chunks).strip()

        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            raise TranslationError(f"Translation failed: {str(e)}")

# Create a singleton instance
translation_service = TranslationService()

def get_language_name(lang_code):
    """Get language name from language code"""
    return LANGUAGE_MAP.get(f'name_{lang_code}', 'Unknown')

def get_iso639_1(lang_code):
    """Get ISO 639-1 code from language code"""
    return LANGUAGE_MAP.get(f'iso639_1_{lang_code}', 'en')

def get_iso639_3(lang_code):
    """Get ISO 639-3 code from language code"""
    return LANGUAGE_MAP.get(f'iso639_3_{lang_code}', 'eng')

def translate_text(text, source_lang, target_lang):
    """Translate text from source language to target language"""
    try:
        # Get ISO codes for translation
        source_iso = get_iso639_1(source_lang)
        target_iso = get_iso639_1(target_lang)
        
        # Create translation service instance
        translator = TranslationService()
        
        # Translate the text using the service
        translated_text = translator.translate_text(
            text=text,
            target_lang=target_iso,
            source_lang=source_iso,
            service='google'  # Default to Google Translate
        )
        
        # Post-processing to clean up translation artifacts
        if translated_text:
            # Remove double spaces
            translated_text = re.sub(r'\s+', ' ', translated_text)
            # Fix sentence endings (ensure proper spacing after periods)
            translated_text = re.sub(r'\.(\w)', r'. \1', translated_text)
            # Remove any leading/trailing whitespace
            translated_text = translated_text.strip()
        
        return translated_text
        
    except Exception as e:
        logging.error(f"Error translating text: {str(e)}")
        return text

def translate_to_english(text: str, source_lang: Optional[str] = None) -> str:
    """Translate text to English with enhanced error handling"""
    try:
        if not text.strip():
            logger.warning("Empty text provided for translation")
            return text

        # Create translation service instance
        translator = TranslationService()
        
        # Map language codes to Google Translate codes
        lang_map = {
            'mar': 'mr',  # Marathi
            'hin': 'hi',  # Hindi
            'eng': 'en',  # English
            'English': 'en',
            'Hindi': 'hi',
            'Marathi': 'mr'
        }
        
        # Convert source language to Google Translate code
        if source_lang:
            original_lang = source_lang
            source_lang = lang_map.get(source_lang, source_lang)
            logger.info(f"Using provided language: {original_lang} -> {source_lang}")
        else:
            # If source language not provided, detect it
            detected_lang = translator.detect_language(text)
            source_lang = lang_map.get(detected_lang, detected_lang)
            logger.info(f"Detected language: {detected_lang} -> {source_lang}")
        
        logger.info(f"Translating from {source_lang} to English")
        logger.info(f"Input text length: {len(text)}")
        logger.info(f"First 100 characters of input: {text[:100]}")
        
        # If the text is already in English, return as is
        if source_lang == 'en':
            logger.info("Text is already in English, skipping translation")
            return text
        
        # For Marathi text, ensure proper encoding
        if source_lang == 'mr':
            try:
                text = text.encode('utf-8').decode('utf-8')
            except:
                pass
        
        # Split text into manageable chunks
        chunks = translator.split_text(text)
        logger.info(f"Split text into {len(chunks)} chunks for translation")
        
        # Translate chunks in parallel
        translated_chunks = []
        for chunk in chunks:
            try:
                # Try Google Translate first
                translated = translator.translate_with_retry(
                    chunk,
                    target_lang='en',
                    source_lang=source_lang,
                    service='google'
                )
                translated_chunks.append(translated)
            except Exception as e:
                logger.error(f"Google Translate failed: {str(e)}")
                try:
                    # Fallback to MyMemory
                    translated = translator.translate_with_retry(
                        chunk,
                        target_lang='en',
                        source_lang=source_lang,
                        service='mymemory'
                    )
                    translated_chunks.append(translated)
                except Exception as e:
                    logger.error(f"MyMemory translation failed: {str(e)}")
                    translated_chunks.append("")  # Add empty string for failed chunks
        
        # Combine translated chunks
        translated_text = '\n\n'.join(translated_chunks)
        logger.info(f"Translation completed. Original length: {len(text)}, Translated length: {len(translated_text)}")
        logger.info(f"First 100 characters of translated text: {translated_text[:100]}")
        
        return translated_text
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        raise TranslationError(f"Failed to translate text: {str(e)}")

def translate_to_language(text: str, target_lang: str, 
                         source_lang: Optional[str] = None, 
                         save_to_db: bool = True) -> str:
    """Translate text to specified language"""
    try:
        # Maintain backward compatibility with file_handlers.py
        if isinstance(text, bytes):
            text = text.decode('utf-8')
            
        # Translate the text
        translated_text = translation_service.translate_text(text, target_lang, source_lang)
        
        # Optionally save to database
        if save_to_db:
            save_to_database(translated_text)
        
        return translated_text
    except Exception as e:
        logger.error(f"Translation to {target_lang} failed: {str(e)}")
        return f"Translation failed: {str(e)}"

# Maintain backward compatibility with existing code
def translate(text: str, target_lang: str = 'en', 
             source_lang: Optional[str] = None) -> str:
    """Backward compatible translation function"""
    return translate_to_language(text, target_lang, source_lang)

def split_image_into_parts(image: Image.Image, num_parts: int = 5) -> List[Image.Image]:
    """Split an image into equal parts vertically"""
    width, height = image.size
    part_height = height // num_parts
    parts = []
    
    for i in range(num_parts):
        top = i * part_height
        bottom = (i + 1) * part_height if i < num_parts - 1 else height
        part = image.crop((0, top, width, bottom))
        parts.append(part)
    
    return parts

def perform_ocr_on_image(image: Image.Image, lang: str = 'hin+mar') -> str:
    """Perform OCR on an image with specified language"""
    try:
        # Convert image to grayscale for better OCR
        if image.mode != 'L':
            image = image.convert('L')
        
        # Perform OCR with specified language
        text = pytesseract.image_to_string(image, lang=lang)
        return text.strip()
    except Exception as e:
        logging.error(f"OCR failed: {str(e)}")
        return ""

def verify_translation(google_translation: str, source_lang: str, target_lang: str) -> Tuple[str, Dict[str, str]]:
    """
    Verify Google translation using MyMemory and LibreTranslator.
    Returns the final translation and verification results.
    """
    translator = TranslationService()
    verification_results = {}
    
    # Get MyMemory translation
    try:
        mymemory_translation = translator.translate_text(
            google_translation,
            target_lang=target_lang,
            source_lang=source_lang,
            service='mymemory'
        )
        verification_results['mymemory'] = mymemory_translation
    except Exception as e:
        logger.error(f"MyMemory verification failed: {str(e)}")
        verification_results['mymemory'] = "Verification failed"
    
    # Get LibreTranslator translation
    try:
        libre_translation = translator.translate_text(
            google_translation,
            target_lang=target_lang,
            source_lang=source_lang,
            service='libre'
        )
        verification_results['libre'] = libre_translation
    except Exception as e:
        logger.error(f"LibreTranslator verification failed: {str(e)}")
        verification_results['libre'] = "Verification failed"
    
    # For now, we'll return the Google translation as the final result
    # In the future, we could implement more sophisticated verification logic
    return google_translation, verification_results

def process_pdf_and_translate(pdf_path: str, source_lang: str = 'hi', 
                            target_lang: str = 'en') -> Tuple[str, Dict[str, str]]:
    """
    Process a PDF file by converting to images, splitting into parts,
    performing OCR, and translating the text with verification.
    
    Args:
        pdf_path: Path to the PDF file
        source_lang: Source language code (default: 'hi' for Hindi)
        target_lang: Target language code (default: 'en' for English)
    
    Returns:
        Tuple containing:
        - Final translated text
        - Dictionary with verification results from other services
    """
    try:
        # Convert PDF to images
        images = convert_from_path(pdf_path)
        all_text = []
        
        # Process each page
        for page_num, image in enumerate(images, 1):
            logging.info(f"Processing page {page_num}")
            
            # Split image into parts
            parts = split_image_into_parts(image)
            
            # Perform OCR on each part
            page_text = []
            for part_num, part in enumerate(parts, 1):
                logging.info(f"Processing part {part_num} of page {page_num}")
                text = perform_ocr_on_image(part)
                page_text.append(text)
            
            # Combine text from all parts
            combined_text = '\n'.join(page_text)
            all_text.append(combined_text)
        
        # Combine text from all pages
        full_text = '\n\n'.join(all_text)
        
        # Clean up the text
        full_text = re.sub(r'\s+', ' ', full_text)  # Remove extra whitespace
        full_text = full_text.strip()
        
        # First translate using Google
        logging.info("Translating with Google Translate...")
        google_translation = translate_text(full_text, source_lang, target_lang)
        
        # Verify the translation with other services
        logging.info("Verifying translation with other services...")
        final_translation, verification_results = verify_translation(
            google_translation,
            source_lang,
            target_lang
        )
        
        return final_translation, verification_results
        
    except Exception as e:
        logging.error(f"Error processing PDF: {str(e)}")
        raise TranslationError(f"Failed to process PDF: {str(e)}")