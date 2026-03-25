# modules/text_simplifier.py3
#summery simplify doc
from collections import Counter
from functools import lru_cache

import nltk
import torch
from nltk.corpus import wordnet
from transformers import LlamaTokenizer, LlamaForCausalLM
import spacy
import logging
from modules.cache_manager import cached_processing

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Note: You'll need to install these dependencies:
# pip install transformers torch accelerate bitsandbytes
# Also, you need Hugging Face authentication for Llama models

# Download required NLTK data
try:
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except Exception as e:
    logger.error(f"Error downloading NLTK data: {e}")

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    nlp.add_pipe('sentencizer')  # Ensure sentence boundaries are set
except Exception as e:
    logger.error(f"Error loading spaCy model: {e}")
    nlp = None

class TextSimplifier:
    def __init__(self):
        self.pos_map = {'NOUN': 'n', 'VERB': 'v', 'ADJ': 'a', 'ADV': 'r'}
        self.models_initialized = False
        self.common_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'if', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])

    def _fallback_short_summary(self, text, max_sentences=3):
        """
        Fallback method for generating a short summary if spaCy fails.
        
        Args:
            text (str): Input text to summarize
            max_sentences (int, optional): Maximum number of sentences in summary. Defaults to 3.
        
        Returns:
            str: Shortened summary
        """
        sentences = nltk.sent_tokenize(text)
        return ' '.join(sentences[:max_sentences])

    def generate_short_summary(self, text, max_sentences=3, max_chars=300):
        """
        Generate a concise, short summary of the input text.
        Args:
            text (str): Input text to summarize
            max_sentences (int, optional): Maximum number of sentences in summary. Defaults to 3.
            max_chars (int, optional): Maximum number of characters in the summary. Defaults to 300.
        Returns:
            str: Shortened summary
        """
        try:
            if not nlp:
                logger.warning("spaCy model not loaded. Using fallback summary method.")
                return self._fallback_short_summary(text, max_sentences)

            doc = nlp(text)
            sentences = list(doc.sents)
            if not sentences:
                return self._fallback_short_summary(text, max_sentences)
            
            doc = nlp(text)
            
            # Score sentences based on importance
            sentence_scores = {}
            for sent in doc.sents:
                # Score based on word frequency
                for token in sent:
                    if token.pos_ in ['NOUN', 'VERB', 'ADJ']:
                        sentence_scores[sent] = sentence_scores.get(sent, 0) + 1
            
            # Sort sentences by score
            ranked_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
            
            # Select top sentences
            summary_sentences = ranked_sentences[:max_sentences]
            
            # Preserve original order
            summary_sentences.sort(key=lambda sent: sent.start)
            
            return ' '.join([sent.text for sent in summary_sentences])
        
        except Exception as e:
            logger.error(f"Error generating short summary: {e}")
            return self._fallback_short_summary(text, max_sentences)

    def _fallback_short_summary(self, text, max_sentences=3):
        """
        Fallback method for generating a short summary if spaCy fails.
        
        Args:
            text (str): Input text to summarize
            max_sentences (int, optional): Maximum number of sentences in summary. Defaults to 3.
        
        Returns:
            str: Shortened summary
        """
        sentences = nltk.sent_tokenize(text)
        return ' '.join(sentences[:max_sentences])
        self.models_initialized = False
        self.initialize_models()
        self.common_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'if', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
    
    def initialize_models(self):
        try:
            logger.info("Initializing local Llama models...")
            import torch
            from transformers import LlamaTokenizer, LlamaForCausalLM

            # Local model path
            model_path = r'C:\Users\NIKHIL GUPTA\.llama\checkpoints\Llama-2-7b'
            
            # Check for GPU availability
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # Load tokenizer and model
            self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
            self.llama_model = LlamaForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
                device_map='auto'  # Automatic device placement
            )
            
            # Ensure model is in evaluation mode
            self.llama_model.eval()
            
            self.models_initialized = True
            logger.info(f"Local Llama model initialized successfully on {device}")
        except Exception as e:
            logger.error(f"Error initializing local Llama models: {e}")
            self.tokenizer = None
            self.llama_model = None
            self.models_initialized = False

    @lru_cache(maxsize=1000)
    def find_simple_synonym(self, word, pos_tag):
        try:
            if len(word) <= 3 or word in self.common_words:
                return word
                
            try:
                synsets = wordnet.synsets(word, pos=self.pos_map.get(pos_tag, 'n'))
            except Exception as e:
                logger.error(f"Error getting synsets for {word}: {e}")
                return word
                
            if not synsets:
                return word
                
            all_lemmas = []
            for syn in synsets:
                try:
                    lemmas = [lemma.name().replace('_', ' ') for lemma in syn.lemmas()]
                    all_lemmas.extend(lemmas)
                except Exception as e:
                    logger.error(f"Error processing lemma for {word}: {e}")
                    continue
            
            if not all_lemmas:
                return word
                
            word_counter = Counter(all_lemmas)
            for synonym, count in word_counter.most_common():
                if len(synonym) < len(word) and synonym != word:
                    return synonym
                    
            return word
            
        except Exception as e:
            logger.error(f"Error in find_simple_synonym for {word}: {e}")
            return word

    def paraphrase_sentence(self, sentence):
        if self.paraphraser:
            try:
                paraphrased = self.paraphraser(sentence, max_length=50, num_return_sequences=1)
                return paraphrased[0]['generated_text']
            except Exception as e:
                logger.error(f"Paraphrasing error: {e}")
                return sentence
        return sentence

    def simplify_sentence(self, sentence):
        try:
            doc = nlp(sentence)
            simplified_words = []
            
            for token in doc:
                if token.pos_ == 'PROPN' or token.text.lower() in ['court', 'judge', 'law', 'legal', 'statute', 'regulation']:
                    simplified_words.append(token.text)
                    continue
                    
                if token.text.lower() in ["don't", "can't", "won't", "isn't", "aren't"]:
                    simplified_words.append(token.text)
                    continue
                    
                if token.pos_ in self.pos_map:
                    simplified_word = self.find_simple_synonym(token.text.lower(), token.pos_)
                    if len(simplified_word) <= len(token.text) or simplified_word in ['be', 'have', 'do', 'make', 'take', 'give']:
                        simplified_words.append(simplified_word)
                    else:
                        simplified_words.append(token.text)
                else:
                    simplified_words.append(token.text)
                    
            simplified_sentence = ' '.join(simplified_words)
            simplified_sentence = simplified_sentence.replace(' ,', ',').replace(' .', '.').replace(' ?', '?').replace(' !', '!')
            
            return simplified_sentence
            
        except Exception as e:
            logger.error(f"Error in simplify_sentence: {e}")
            return sentence

    def simplify_with_llama(self, text, max_length=500):
        """Use Llama AI to simplify legal language"""
        try:
            # Import Llama model from text_processing
            from modules.text_processing import llama_model, llama_tokenizer
            
            if llama_model is None or llama_tokenizer is None:
                return self.simplify_sentence(text)
            
            # Construct a prompt for simplification
            prompt = f"""Simplify the following legal text into plain, easy-to-understand language. 
            Convert complex legal terms into simple everyday language:

{text}

Simplified version:"""
            
            # Tokenize and generate simplified text
            inputs = llama_tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
            outputs = llama_model.generate(**inputs, max_length=max_length, num_return_sequences=1)
            simplified_text = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the simplified version
            simplified_version = simplified_text.split('Simplified version:')[-1].strip()
            
            return simplified_version or text
        
        except ImportError:
            logger.warning("Llama model not available. Falling back to standard simplification.")
            return self.simplify_sentence(text)
        
        except Exception as e:
            logger.error(f"Llama simplification error: {e}")
            return self.simplify_sentence(text)

    def legal_term_translator(self, term):
        """Translate complex legal terms into simple language"""
        legal_term_map = {
            'plaintiff': 'person filing the lawsuit',
            'defendant': 'person being sued',
            'tort': 'civil wrong that causes harm',
            'statute': 'law passed by legislature',
            'jurisdiction': 'area where a court has power',
            'prima facie': 'at first sight, based on first impression',
            'pro bono': 'free legal work',
            'subpoena': 'legal order to appear in court',
            'affidavit': 'written statement made under oath',
            'habeas corpus': 'legal action to report unlawful detention'
        }
        
        # Try Llama AI translation if available
        try:
            from modules.text_processing import llama_model, llama_tokenizer
            
            if llama_model and llama_tokenizer:
                prompt = f"""Explain the legal term '{term}' in simple, everyday language that a person without legal training can understand."""
                
                inputs = llama_tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
                outputs = llama_model.generate(**inputs, max_length=100, num_return_sequences=1)
                translation = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                return translation.strip() or legal_term_map.get(term.lower(), term)
        except Exception:
            pass
        
        # Fallback to predefined translations
        return legal_term_map.get(term.lower(), term)

    def simplify_text(self, text, complexity_level='medium'):
        """
        Simplify text using local Llama model's text generation capabilities.
        
        :param text: Input text to simplify
        :param complexity_level: Level of simplification (low, medium, high)
        :return: Simplified text
        """
        if not self.models_initialized:
            logger.error("Local Llama model not initialized")
            return text
        
        try:
            # Prepare prompt for text simplification
            prompt_templates = {
                'low': "Explain the following text in very simple words, keeping most of the original meaning:\n{text}\n\nSimplified explanation:",
                'medium': "Rewrite the following text in simpler language, making it easier to understand:\n{text}\n\nSimpler version:",
                'high': "Break down this text into the simplest possible language, removing complex terms:\n{text}\n\nUltra-simplified version:"
            }
            
            # Select prompt based on complexity level
            prompt = prompt_templates.get(complexity_level, prompt_templates['medium']).format(text=text)
            
            # Prepare inputs
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
            
            # Move inputs to the same device as the model
            inputs = {k: v.to(self.llama_model.device) for k, v in inputs.items()}
            
            # Generate simplified text
            with torch.no_grad():  # Disable gradient computation
                outputs = self.llama_model.generate(
                    **inputs, 
                    max_length=512,  # Limit output length
                    num_return_sequences=1,
                    do_sample=True,  # Use sampling for more diverse output
                    temperature=0.7,  # Control randomness
                    top_p=0.9,  # Nucleus sampling
                    no_repeat_ngram_size=2  # Reduce repetition
                )
            
            # Decode the generated text
            simplified = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the simplified part (after the prompt)
            simplified = simplified.replace(prompt, '').strip()
            
            return simplified
        except Exception as e:
            logger.error(f"Error simplifying text with local Llama model: {e}")
            return text

    @cached_processing('summarization')
    def summarize_text(self, text: str, max_length: int = 500, min_length: int = 30) -> str:
        """
        Optimized summarization with caching and adaptive chunking.
        Falls back to generate_short_summary if model is unavailable or fails.
        """
        import traceback
        try:
            if not text or not isinstance(text, str):
                logger.error(f"Summarization error: Invalid input text. Text: {text}")
                return "Error: Invalid input text"
            
            # Use optimized chunking for summarization
            from modules.text_chunker import chunk_text_for_processing
            chunks = chunk_text_for_processing(text, processing_type="summarization")
            
            if len(chunks) == 1:
                # Single chunk - process directly
                try:
                    from modules.model_manager import model_manager
                    summarizer = model_manager.get_model('summarizer')
                    if summarizer:
                        result = summarizer(chunks[0].content, 
                                          max_length=max_length, 
                                          min_length=min_length)
                        return result[0]['summary_text'] if result else self.generate_short_summary(text)
                except Exception as e:
                    logger.error(f"Model summarization failed: {e}")
                    
            # Multiple chunks - summarize each and combine
            summaries = []
            for chunk in chunks:
                chunk_summary = self.generate_short_summary(chunk.content, max_sentences=2)
                if chunk_summary:
                    summaries.append(chunk_summary)
            
            # Combine summaries
            combined_summary = ' '.join(summaries)
            if len(combined_summary) > max_length:
                return self.generate_short_summary(combined_summary, max_sentences=3)
            
            return combined_summary or self.generate_short_summary(text)
            
        except Exception as e:
            logger.error(f"Summarization error: {e}")
            logger.error(traceback.format_exc())
            return self.generate_short_summary(text)

# Create a singleton instance
text_simplifier = TextSimplifier()