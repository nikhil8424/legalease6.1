# app.py
from flask import Flask, render_template, request, flash, jsonify
from werkzeug.utils import secure_filename
import os
import time
import nltk
import spacy
from flask_cors import CORS

# Import from our modules
from modules.file_handlers import extract_text
from modules.text_processing import (
    identify_document_type,
    extract_dates,
    extract_names,
    clean_text,
    extract_key_information   # <-- Add this line
)
from modules.translation import translate_to_english
from modules.text_simplifier import text_simplifier
from data_list import MAHARASHTRA_CITIES, legal_keywords
from modules.fast_rag_chatbot import chat_response, add_document, get_stats
from modules.enhanced_fallback_extractor import extract_key_information_fixed
# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['SECRET_KEY'] = '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'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route("/", methods=["GET"])
def mainpage():
    return render_template("index.html")

@app.route("/getstarted", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/features", methods=["GET"])
def features():
    return render_template("features.html")

@app.route("/contact", methods=["GET"])
def contact():
    return render_template("contact.html")

@app.route("/upload", methods=["POST"])
def upload():
    if 'file' not in request.files:
        print("No file part in request")
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        print("No file selected")
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            print(f"Saving file to: {filepath}")
            file.save(filepath)
            print("File saved successfully")
            
            # Get the selected language
            language = request.form.get('language', 'English')
            print(f"Selected language: {language}")
            
            # Check if Tesseract is installed
            tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            if not os.path.exists(tesseract_path):
                raise Exception("Tesseract OCR is not installed. Please install it from https://github.com/UB-Mannheim/tesseract/wiki")
            
            # Extract text from the file using the selected language
            print(f"Starting text extraction with language: {language}")
            text = extract_text(filepath, language)
            print(f"Text extracted successfully. Length: {len(text)}")
            
            if not text.strip():
                print("Warning: Extracted text is empty")
                return jsonify({
                    'error': 'No text could be extracted from the file. The file might be scanned or image-based.',
                    'details': 'Try using a different file or ensure the file contains readable text.'
                }), 400
            
            # If the text is not in English, translate it
            if language != "English":
                print(f"Translating from {language} to English...")
                try:
                    text = translate_to_english(text, language)
                    print("Translation completed")
                except Exception as e:
                    print(f"Translation error: {str(e)}")
                    return jsonify({
                        'error': 'Translation failed',
                        'details': str(e)
                    }), 500
            
            response = jsonify({
                'translated_text': text,
                'success': True
            })
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
            
        except Exception as e:
            print(f"Error processing file: {str(e)}")
            error_message = str(e)
            
            if "Tesseract" in error_message:
                return jsonify({
                    'error': 'Tesseract OCR is not installed or not properly configured.',
                    'details': 'Please install Tesseract OCR from https://github.com/UB-Mannheim/tesseract/wiki and make sure it is in your PATH.'
                }), 500
            elif "password protected" in error_message.lower():
                return jsonify({
                    'error': 'The file is password protected.',
                    'details': 'Please remove the password from the file and try again.'
                }), 400
            else:
                return jsonify({
                    'error': 'Error processing file',
                    'details': error_message
                }), 500
        finally:
            # Clean up the uploaded file
            try:
                os.remove(filepath)
            except:
                pass

@app.route("/process", methods=["POST"])
def process():
    try:
        action = request.form.get('action')
        text = request.form.get('translated_text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'})
            
        if action == 'Identify':
            # Identify document type and extract sections
            doc_type, sections, keyword_counts = identify_document_type(text, legal_keywords)
            return jsonify({
                'action': 'Identify',
                'document_type': doc_type,
                'sections': sections,
                'keyword_counts': keyword_counts
            })
        elif action == 'Key Points':
            # Use advanced LLM-based key information extraction
            key_info = extract_key_information_fixed(text)
            key_info['action'] = 'Key Points'
            return jsonify(key_info)
        elif action == 'Plain Language':
            # Simplify the text
            simplified_text = text_simplifier.simplify_text(text)
            return jsonify({
                'action': 'Plain Language',
                'simplified_text': simplified_text
            })
        elif action == 'Summary':
            # Generate summary using TextSimplifier
            try:
                summary = text_simplifier.summarize_text(text)
                return jsonify({
                    'action': 'Summary',
                    'summary': summary
                })
            except Exception as e:
                print(f"Summarization error: {str(e)}")
                return jsonify({
                    'error': 'Summarization failed',
                    'details': str(e)
                }), 500
        else:
            return jsonify({'error': 'Invalid action'})
            
    except Exception as e:
        return jsonify({'error': str(e)})

# Optimized RAG Chatbot Endpoints
@app.route("/chat", methods=["POST"])
def chat():
    """Fast RAG chatbot endpoint."""
    try:
        data = request.get_json() if request.is_json else request.form
        query = data.get('message', '').strip()
        
        if not query:
            return jsonify({
                'error': 'Please provide a message',
                'response': 'Please ask me a question about the uploaded documents.'
            }), 400
        
        # Generate response using optimized RAG chatbot
        response = chat_response(query)
        
        return jsonify({
            'response': response,
            'success': True
        })
        
    except Exception as e:
        print(f"Chat error: {str(e)}")
        return jsonify({
            'error': 'Chat processing failed',
            'response': 'I\'m having trouble processing your request. Please try again.'
        }), 500

@app.route("/chat/add_document", methods=["POST"])
def add_document_to_chat():
    """Add document to RAG database for chatbot."""
    try:
        text = request.form.get('translated_text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Add document to RAG database
        doc_id = add_document(text, {
            'source': 'upload',
            'timestamp': str(time.time()),
            'length': len(text)
        })
        
        return jsonify({
            'success': True,
            'message': 'Document added to knowledge base',
            'doc_id': doc_id
        })
        
    except Exception as e:
        print(f"Add document error: {str(e)}")
        return jsonify({
            'error': 'Failed to add document',
            'details': str(e)
        }), 500

@app.route("/chat/stats", methods=["GET"])
def chat_stats():
    """Get chatbot and database statistics."""
    try:
        stats = get_stats()
        return jsonify({
            'stats': stats,
            'success': True
        })
    except Exception as e:
        print(f"Stats error: {str(e)}")
        return jsonify({
            'error': 'Failed to get statistics',
            'details': str(e)
        }), 500

if __name__ == "__main__":
    app.run(debug=True)