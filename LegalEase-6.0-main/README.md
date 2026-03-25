# Legalese - Legal Document Analysis Assistant

Legalese is an AI-powered application designed to simplify and analyze legal documents. It helps users understand complex legal texts by providing simplified versions, summaries, and key insights.

## Features

- **Document Upload**: Support for PDF, PNG, and JPG files
- **Text Extraction**: Accurate extraction of text from various document formats
- **Language Support**: Multiple language support including English, Hindi, and Marathi
- **Document Analysis**:
  - **Identify**: Detects document type and key sections
  - **Key Points**: Extracts important dates, organizations, cities, and names
  - **Plain Language**: Converts complex legal jargon into simpler language
  - **Summary**: Generates concise summaries of legal documents
- **Chat Interface**: Interactive chat for document-related queries
- **Responsive Design**: Works on both desktop and mobile devices
- **Dark Mode**: User-friendly dark mode support

## Technical Stack

- **Backend**:
  - Flask (Web Framework)
  - spaCy (NLP Processing)
  - NLTK (Text Processing)
  - Transformers (AI Models)
  - PyPDF2 (PDF Processing)
  - OpenCV (Image Processing)

- **Frontend**:
  - HTML5
  - CSS3
  - JavaScript
  - Font Awesome Icons

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/legalese.git
cd legalese
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required NLTK data:
```python
import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
```

5. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

## System Requirements

- Python 3.8 or higher
- At least 4GB RAM
- Tesseract OCR Engine (for OCR functionality)
- poppler-utils (for pdf2image)
- GPU recommended for better performance with AI models

## Usage Flow

1. **Document Upload**:
   - Click "Upload" button
   - Select document (PDF, PNG, or JPG)
   - Choose document language
   - Click "Upload" to process

2. **Document Analysis**:
   - **Identify**: Click to detect document type and sections
   - **Key Points**: Click to extract important information
   - **Plain Language**: Click to get simplified version
   - **Summary**: Click to get document summary

3. **Chat Interface**:
   - Type questions about the document
   - Get AI-powered responses
   - Clear chat history as needed

## Privacy Policy

- All uploaded documents are handled with strict confidentiality
- Documents are processed locally when possible
- No files or information are shared with third parties without explicit consent
- Users can request data deletion at any time

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For support or inquiries:
- Email: guptanikhil8424@gmail.com
- Phone: +91 9167534578

## Acknowledgments

- spaCy for NLP processing
- Transformers for AI models
- NLTK for text processing
- Flask for web framework
- All other open-source libraries used in this project "# LegalEase-6.0" 
