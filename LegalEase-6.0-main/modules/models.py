# modules/models.py
import os
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from modules.config import MODEL_CONFIGS, CACHE_DIR

class TextProcessor:
    def __init__(self):
        self.summarizer = None
        self.paraphraser = None
        self.tokenizer = None
        self.model = None
        
    def initialize_models(self):
        """Initialize all required models"""
        # Set cache directory for Hugging Face models
        os.environ['TRANSFORMERS_CACHE'] = str(CACHE_DIR)
        
        # Initialize summarization model
        self.summarizer = pipeline(
            "summarization",
            model=MODEL_CONFIGS['summarization']['model'],
            tokenizer=MODEL_CONFIGS['summarization']['model']
        )
        
        # Initialize paraphrasing model
        self.paraphraser = pipeline(
            "text2text-generation",
            model=MODEL_CONFIGS['paraphrasing']['model'],
            tokenizer=MODEL_CONFIGS['paraphrasing']['model']
        )
        
        # Initialize translation model
        self.tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
    
    def summarize_text(self, text, max_length=500, min_length=30):
        """Summarize the given text"""
        if not self.summarizer:
            self.initialize_models()
            
        summary = self.summarizer(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )
        return summary[0]['summary_text']
    
    def paraphrase_text(self, text, max_length=100):
        """Paraphrase the given text"""
        if not self.paraphraser:
            self.initialize_models()
            
        paraphrased = self.paraphraser(
            text,
            max_length=max_length,
            num_return_sequences=1
        )
        return paraphrased[0]['generated_text']
    
    def translate_text(self, text, target_lang):
        """Translate text to target language"""
        if not self.tokenizer or not self.model:
            self.initialize_models()
            
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        outputs = self.model.generate(**inputs)
        translated = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return translated[0] 