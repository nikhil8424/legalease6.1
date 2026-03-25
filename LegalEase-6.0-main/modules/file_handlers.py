# modules/file_handlers.py
#If you give this module a file path:
#For a regular PDF (with digital text): It extracts the text directly.
#For a scanned PDF (images inside): It converts pages into images and performs OCR.
#For image files: It extracts text using OCR.
#If the PDF is encrypted: It alerts the user.
#If all methods fail: It throws a descriptive error.
#
#upto 4 pages all good
#more than 4 pages  -  it will indivisually convert all pages to img and then do the incivisial ocr
#not much effective OCR in done on pdf more tha 4 pages


import PyPDF2 #read and extracts text from pdf files
from pdf2image import convert_from_path#har page ko image me convert kr dega(usefull for scanned pdf)
from PIL import Image#image manipulation k liye
import os
import sys
import fitz  # PyMuPDF
import logging
from datetime import datetime
from typing import Optional, List
import numpy as np
import cv2
from langdetect import detect
import tempfile
import shutil
import concurrent.futures
from functools import partial
from modules.models import TextProcessor
from modules.config import LANGUAGE_MAP, TESSERACT_CONFIG
import pytesseract
from PIL import ImageEnhance
import re
from modules.marathi_hindi_ocr import process_pdf_pages as extract_indic_text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize TextProcessor
text_processor = TextProcessor()

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Set Tesseract data directory
os.environ['TESSDATA_PREFIX'] = TESSERACT_CONFIG['tessdata_dir']

def get_tesseract_lang(lang_code):
    """Convert language code to Tesseract language code"""
    return LANGUAGE_MAP.get(f'tesseract_{lang_code}', 'eng')

def preprocess_image(image):
    """
    Image preprocessing optimized for English text.
    """
    try:
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter
        bilateral = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Apply morphological operations
        kernel = np.ones((1, 1), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(morph, None, 7, 7, 21)
        
        # Convert back to PIL Image
        return Image.fromarray(denoised)
        
    except Exception as e:
        logger.error(f"Error in image preprocessing: {str(e)}")
        return image

def detect_language(text: str) -> str:
    """Detect the language of the text"""
    try:
        if text.strip():
            return detect(text)
        return 'eng'  # Default to English
    except:
        return 'eng'

def check_pdf_encryption(pdf_path: str) -> bool:
    """Check if PDF is password protected"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            if pdf_reader.is_encrypted:
                return True
        return False
    except Exception as e:
        logger.error(f"Error checking PDF encryption: {str(e)}")
        return False

def split_image_into_three_parts(image, overlap=100):
    """
    Split an image into exactly 3 vertical parts with overlap for better OCR processing.
    
    Args:
        image: PIL Image object
        overlap: Number of pixels to overlap between parts
        
    Returns:
        List of (part, position) tuples where position is (x, y) of top-left corner
    """
    try:
        width, height = image.size
        
        # Calculate the height of each part
        part_height = height // 3
        
        parts = []
        
        # Create three parts with overlap
        for i in range(3):
            # Calculate boundaries with overlap
            top = max(0, i * part_height - overlap)
            bottom = min(height, (i + 1) * part_height + overlap)
            
            # Crop the part
            part = image.crop((0, top, width, bottom))
            parts.append((part, (0, top)))
        
        return parts
    except Exception as e:
        logger.error(f"Error splitting image into three parts: {str(e)}")
        return [(image, (0, 0))]  # Return original image if splitting fails

def is_valid_marathi_text(text):
    """
    Validate if the extracted text contains valid Marathi characters.
    """
    # Marathi Unicode range
    marathi_range = r'[\u0900-\u097F]'
    # Check if text contains any Marathi characters
    return bool(re.search(marathi_range, text))

def extract_text_from_image_chunks(image_path, language='mar'):
    """
    Extract text from an image by processing it in three vertical parts.
    """
    try:
        # Load and preprocess the image
        image = Image.open(image_path)
        processed_image = preprocess_image(image)
        
        # Split image into three parts
        parts = split_image_into_three_parts(processed_image)
        logger.info(f"Split image into {len(parts)} parts")
        
        # Get Tesseract language code
        tess_lang = get_tesseract_lang(language)
        
        # Process each part
        all_text = []
        for part, (x, y) in parts:
            try:
                # Save the processed part temporarily for debugging
                temp_path = f"temp_part_{y}.png"
                part.save(temp_path)
                
                # Try different OCR configurations
                configs = [
                    f'--oem {TESSERACT_CONFIG["oem"]} --dpi {TESSERACT_CONFIG["dpi"]} --psm 4 -l {tess_lang}',
                    f'--oem {TESSERACT_CONFIG["oem"]} --dpi {TESSERACT_CONFIG["dpi"]} --psm 6 -l {tess_lang}',
                    f'--oem {TESSERACT_CONFIG["oem"]} --dpi {TESSERACT_CONFIG["dpi"]} --psm 3 -l {tess_lang}'
                ]
                
                best_text = ""
                best_confidence = 0
                
                for config in configs:
                    try:
                        # Perform OCR
                        text = pytesseract.image_to_string(
                            part,
                            config=config,
                            lang=tess_lang
                        )
                        
                        # Get confidence score
                        data = pytesseract.image_to_data(
                            part,
                            config=config,
                            lang=tess_lang,
                            output_type=pytesseract.Output.DICT
                        )
                        
                        # Calculate average confidence
                        confidences = [float(conf) for conf in data['conf'] if float(conf) > 0]
                        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                        
                        # Validate text and update best result
                        if avg_confidence > best_confidence and is_valid_marathi_text(text):
                            best_text = text
                            best_confidence = avg_confidence
                            
                    except Exception as e:
                        logger.warning(f"Error with config {config}: {str(e)}")
                        continue
                
                if best_text.strip():
                    all_text.append((best_text, y))
                    logger.info(f"Part at y={y} processed successfully with confidence: {best_confidence:.2f}")
                else:
                    logger.warning(f"No valid text extracted from part at y={y}")
                    
            except Exception as e:
                logger.warning(f"Error processing part at y={y}: {str(e)}")
                continue
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        # Sort parts by vertical position (top to bottom)
        all_text.sort(key=lambda x: x[1])
        
        # Combine text from all parts
        combined_text = "\n".join(text for text, _ in all_text)
        
        # Validate final text
        if not is_valid_marathi_text(combined_text):
            logger.warning("Final combined text does not contain valid Marathi characters")
            return ""
        
        logger.info(f"Extracted text length from parts: {len(combined_text)}")
        return combined_text.strip()
        
    except Exception as e:
        logger.error(f"Error in chunked text extraction: {str(e)}")
        return ""

def extract_text_from_image(image_path):
    """
    Extract text from an image with optimized processing for English.
    """
    try:
        # Load and preprocess the image
        image = Image.open(image_path)
        processed_image = preprocess_image(image)
        
        # Perform OCR with optimized settings for English
        text = pytesseract.image_to_string(
            processed_image,
            config=f'--oem {TESSERACT_CONFIG["oem"]} --dpi {TESSERACT_CONFIG["dpi"]} --psm 3 -l eng',
            lang='eng'
        )
        
        return text.strip()
        
    except Exception as e:
        logger.error(f"Error in text extraction: {str(e)}")
        return ""
def save_to_database(text: str, database_path: str = 'uploads/database/db.txt'):
    """Save text to optimized RAG database"""
    try:
        # Use the new optimized RAG database instead of plain text file
        from modules.rag_database import rag_db
        
        # Add document to RAG database with metadata
        doc_metadata = {
            'source': 'file_upload',
            'added_at': datetime.now().isoformat(),
            'text_length': len(text)
        }
        
        doc_id = rag_db.add_document(text, doc_metadata)
        logger.info(f'Text added to RAG database with ID: {doc_id}')
        
        # Also save to legacy text file for backward compatibility
        os.makedirs(os.path.dirname(database_path), exist_ok=True)
        with open(database_path, 'a', encoding='utf-8') as f:  # Append mode
            f.write(f'\n\n--- Document {doc_id} ---\n')
            f.write(text)
            f.write('\n--- End Document ---\n')
            
    except Exception as e:
        logger.error(f'Failed to save text to database: {e}')

def pdf_to_text(pdf_path: str) -> str:
    """Extract text from PDF using multiple methods"""
    try:
        logger.info("Attempting to extract text from PDF...")
        text = ''
        
        # Try PyPDF2 first
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            logger.info(f"PDF has {len(pdf_reader.pages)} pages")
            
            for i, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text() or ''
                text += page_text
                logger.info(f"Page {i+1} text length: {len(page_text)}")
        
        # If no text found, try PyMuPDF with proper error handling
        if not text.strip():
            logger.info("No text found with PyPDF2, trying PyMuPDF...")
            try:
                doc = fitz.open(pdf_path)
                for i, page in enumerate(doc):
                    page_text = page.get_text()
                    text += page_text
                    logger.info(f"Page {i+1} text length (PyMuPDF): {len(page_text)}")
            except Exception as e:
                logger.warning(f"PyMuPDF failed: {str(e)}")
        
        logger.info(f"Total extracted text length: {len(text)}")
        return text
    except Exception as e:
        logger.error(f"Error reading PDF: {str(e)}")
        raise Exception(f"Error reading PDF: {str(e)}")

def process_single_image(image_path, language):
    """Process a single image with OCR in parallel"""
    try:
        return extract_text_from_image(image_path)
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")
        return ""

def pdf_to_images_and_text(pdf_path: str, language: str = 'eng') -> str:
    """Convert PDF to images and extract text with OCR"""
    try:
        logger.info(f"Converting PDF to images for language: {language}")
        
        # Create temporary directory for images
        temp_dir = tempfile.mkdtemp()
        try:
            # Convert with optimized settings
            images = convert_from_path(
                pdf_path,
                dpi=300,
                thread_count=4,
                grayscale=True,
                size=(2400, None),
                fmt='png',
                use_pdftocairo=True
            )
            logger.info(f"PDF converted to {len(images)} images")
            
            # Process each page
            text = ""
            for i, image in enumerate(images):
                temp_image_path = os.path.join(temp_dir, f"page_{i+1}.png")
                image.save(temp_image_path, "PNG")
                
                # Use appropriate OCR based on language
                if language in ['en', 'eng']:
                    page_text = extract_text_from_image(temp_image_path)
                else:
                    page_text = extract_indic_text(temp_image_path, language)
                
                text += page_text + "\n\n"
                logger.info(f"Processed page {i+1}")
            
            return text.strip()
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir)
    except Exception as e:
        logger.error(f"Error converting PDF to images: {str(e)}")
        raise Exception(f"Error converting PDF to images: {str(e)}")

def extract_text(file_path: str, language_code: Optional[str] = None) -> str:
    """Main function to extract text from files with language-specific processing"""
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise Exception(f"File not found: {file_path}")
    
    try:
        # Map language codes
        source_lang = LANGUAGE_MAP.get(language_code, 'en')
        logger.info(f"Processing with language: {source_lang}")
        
        # Check file type
        if file_path.lower().endswith('.pdf'):
            # Check if PDF is password protected
            if check_pdf_encryption(file_path):
                raise Exception("PDF is password protected. Please remove the password and try again.")
            
            # First try regular text extraction
            logger.info("Attempting regular text extraction...")
            try:
                text = pdf_to_text(file_path)
                logger.info(f"Regular text extraction result length: {len(text)}")
                
                # If no text was extracted, try OCR
                if not text.strip():
                    logger.info(f"No text found in regular extraction, attempting OCR...")
                    if source_lang in ['en', 'eng']:
                        text = pdf_to_images_and_text(file_path, source_lang)
                    else:
                        text = extract_indic_text(file_path, source_lang)
                    logger.info(f"OCR result length: {len(text)}")
                
                if not text.strip():
                    raise Exception("No text could be extracted from the PDF using any method.")
                
                return text
            except Exception as e:
                logger.error(f"Error in PDF processing: {str(e)}")
                # If PDF processing fails, try direct OCR
                logger.info("Falling back to direct OCR...")
                if source_lang in ['en', 'eng']:
                    return pdf_to_images_and_text(file_path, source_lang)
                else:
                    return extract_indic_text(file_path, source_lang)
            
        elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            logger.info(f"Processing image with OCR")
            # Use appropriate OCR based on language
            if source_lang in ['en', 'eng']:
                text = extract_text_from_image(file_path)
            else:
                text = extract_indic_text(file_path, source_lang)
            logger.info(f"Image OCR result length: {len(text)}")
            return text
        else:
            raise ValueError("Only PDF and image files are supported.")
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise Exception(f"Error processing file: {str(e)}")