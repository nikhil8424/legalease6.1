import os
import logging
import re
import numpy as np
import cv2
from PIL import Image
import pytesseract
import tempfile
import shutil
from typing import Optional
from modules.config import LANGUAGE_MAP, TESSERACT_CONFIG
from pdf2image import convert_from_path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_tesseract_lang(lang_code):
    """Convert language code to Tesseract language code"""
    # Map language codes to Tesseract language codes
    lang_map = {
        'mar': 'mar',
        'hin': 'hin',
        'mr': 'mar',
        'hi': 'hin'
    }
    return lang_map.get(lang_code, 'mar')

def preprocess_image(image):
    """
    Enhanced image preprocessing for better OCR results, especially for Marathi text.
    """
    try:
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Handle grayscale images
        if len(image.shape) == 2:  # If already grayscale
            gray = image
        else:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while preserving edges
        bilateral = cv2.bilateralFilter(gray, d=15, sigmaColor=75, sigmaSpace=75)
        
        # Apply adaptive thresholding with optimized parameters for Marathi
        thresh = cv2.adaptiveThreshold(
            bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 21, 11
        )
        
        # Apply morphological operations to clean up the image
        kernel = np.ones((2, 2), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(morph, None, 10, 7, 21)
        
        # Apply sharpening
        kernel_sharp = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel_sharp)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast_enhanced = clahe.apply(sharpened)
        
        # Apply additional noise reduction
        denoised_final = cv2.fastNlMeansDenoising(contrast_enhanced, None, 7, 7, 21)
        
        # Convert back to PIL Image
        return Image.fromarray(denoised_final)
        
    except Exception as e:
        logger.error(f"Error in image preprocessing: {str(e)}")
        return image

def is_valid_marathi_text(text):
    """
    Validate if the extracted text contains valid Marathi characters.
    """
    # Marathi Unicode range
    marathi_range = r'[\u0900-\u097F]'
    # Check if text contains any Marathi characters
    return bool(re.search(marathi_range, text))

def split_image_into_eight_parts(image):
    """
    Split an image into exactly 8 vertical parts with increased overlap for better OCR processing.
    """
    try:
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Calculate the height of each part
        part_height = height // 8
        overlap = 200  # Increased overlap for better text capture
        
        parts = []
        
        # Create eight parts with overlap
        for i in range(8):
            # Calculate boundaries with overlap
            top = max(0, i * part_height - overlap)
            bottom = min(height, (i + 1) * part_height + overlap)
            
            # Crop the part
            part = image[top:bottom, :]
            parts.append((Image.fromarray(part), top))
        
        return parts
    except Exception as e:
        logger.error(f"Error splitting image into eight parts: {str(e)}")
        return [(Image.fromarray(image), 0)]  # Return original image if splitting fails

def extract_text_from_image_chunks(image_path, language='mar'):
    """
    Extract text from an image by processing it in six vertical parts with enhanced OCR configurations.
    """
    try:
        # Load and preprocess the image
        image = Image.open(image_path)
        processed_image = preprocess_image(image)
        
        # Split image into six parts
        parts = split_image_into_eight_parts(processed_image)
        logger.info(f"Split image into {len(parts)} parts")
        
        # Get Tesseract language code
        tess_lang = get_tesseract_lang(language)
        
        # Process each part
        all_text = []
        for part, y in parts:
            try:
                # Save the processed part temporarily
                temp_path = f"temp_part_{y}.png"
                part.save(temp_path)
                
                # Try different OCR configurations
                configs = [
                    f'--oem {TESSERACT_CONFIG["oem"]} --dpi {TESSERACT_CONFIG["dpi"]} --psm 4 -l {tess_lang}',
                    f'--oem {TESSERACT_CONFIG["oem"]} --dpi {TESSERACT_CONFIG["dpi"]} --psm 6 -l {tess_lang}',
                    f'--oem {TESSERACT_CONFIG["oem"]} --dpi {TESSERACT_CONFIG["dpi"]} --psm 3 -l {tess_lang}',
                    f'--oem {TESSERACT_CONFIG["oem"]} --dpi {TESSERACT_CONFIG["dpi"]} --psm 1 -l {tess_lang}',
                    f'--oem {TESSERACT_CONFIG["oem"]} --dpi {TESSERACT_CONFIG["dpi"]} --psm 11 -l {tess_lang}'
                ]
                
                best_text = ""
                best_confidence = 0
                min_confidence = 60.0  # Minimum confidence threshold
                
                for config in configs:
                    try:
                        # Perform OCR with specific configuration for Marathi
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
                        if avg_confidence > best_confidence and avg_confidence >= min_confidence and is_valid_marathi_text(text):
                            best_text = text
                            best_confidence = avg_confidence
                            
                    except Exception as e:
                        logger.warning(f"Error with config {config}: {str(e)}")
                        continue
                
                if best_text.strip() and best_confidence >= min_confidence:
                    all_text.append((best_text, y))
                    logger.info(f"Part at y={y} processed successfully with confidence: {best_confidence:.2f}")
                else:
                    logger.warning(f"No valid text extracted from part at y={y} (confidence: {best_confidence:.2f})")
                    
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

def process_pdf_pages(file_path, language='mar'):
    """
    Process a file (PDF or image) by performing OCR.
    """
    try:
        logger.info(f"Processing file: {file_path}")
        
        # Check if file is PDF or image
        if file_path.lower().endswith('.pdf'):
            # Create temporary directory for images
            temp_dir = tempfile.mkdtemp()
            try:
                # Convert PDF to images with optimized settings
                images = convert_from_path(
                    file_path,
                    dpi=300,
                    thread_count=4,
                    grayscale=True,
                    size=(2400, None),
                    fmt='png',
                    use_pdftocairo=True
                )
                logger.info(f"PDF converted to {len(images)} images")
                
                # Process each page
                all_text = []
                for i, image in enumerate(images):
                    temp_image_path = os.path.join(temp_dir, f"page_{i+1}.png")
                    image.save(temp_image_path, "PNG")
                    
                    # Extract text from the page using six-part processing
                    page_text = extract_text_from_image_chunks(temp_image_path, language)
                    all_text.append(page_text)
                    logger.info(f"Processed page {i+1}")
                
                # Combine text from all pages
                final_text = "\n\n".join(all_text)
                logger.info(f"Total extracted text length: {len(final_text)}")
                
                return final_text.strip()
                
            finally:
                # Clean up temporary directory
                shutil.rmtree(temp_dir)
        else:
            # Process single image file
            logger.info("Processing single image file")
            return extract_text_from_image_chunks(file_path, language)
            
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise Exception(f"Error processing file: {str(e)}") 