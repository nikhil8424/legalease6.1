# modules/config.py
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
UPLOADS_DIR = BASE_DIR / 'uploads'
TEMPLATES_DIR = BASE_DIR / 'templates'
STATIC_DIR = BASE_DIR / 'static'
CACHE_DIR = BASE_DIR / 'huggingface_cache'

# Create necessary directories
for directory in [UPLOADS_DIR, CACHE_DIR]:
    directory.mkdir(exist_ok=True)

# Flask configuration
class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'your_secret_key')
    UPLOAD_FOLDER = str(UPLOADS_DIR)
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    TEMPLATES_AUTO_RELOAD = True

# API Keys
# Development API key - replace with your own key in production
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Model configurations
MODEL_CONFIGS = {
    'summarization': {
        'model': 'facebook/bart-large-cnn',
        'max_length': 500,
        'min_length': 30
    },
    'paraphrasing': {
        'model': 't5-small',
        'max_length': 100
    }
}

# Language configurations
LANGUAGE_MAP = {
    # ISO 639-1 codes (2-letter)
    'en': 'en',  # English
    'hi': 'hi',  # Hindi
    'mr': 'mr',  # Marathi
    
    # ISO 639-3 codes (3-letter)
    'eng': 'en',
    'hin': 'hi',
    'mar': 'mr',
    
    # Language names
    'English': 'en',
    'Hindi': 'hi',
    'Marathi': 'mr',
    
    # Tesseract language codes
    'tesseract_en': 'eng',
    'tesseract_hi': 'hin',
    'tesseract_mr': 'mar'
}

# Tesseract OCR configuration
TESSERACT_CONFIG = {
    'dpi': 300,
    'oem': 1,  # LSTM OCR Engine Mode
    'psm': 4,  # Assume a single column of text of variable sizes
    'tessdata_dir': "C:/Program Files/Tesseract-OCR/tessdata",
    'tesseract_cmd': r'C:\Program Files\Tesseract-OCR\tesseract.exe'
} 