# modules/llm_model.py
"""
Optimized LLM model management with lazy loading and performance optimization.
Integrates with the ModelManager for efficient memory usage.
"""
import logging
from modules.model_manager import model_manager

logger = logging.getLogger(__name__)

# Legacy compatibility - these will use lazy loading now
llama_model = None
llama_tokenizer = None

def get_llm():
    """
    Returns a LangChain-compatible LLM object with lazy loading.
    Now uses ModelManager for efficient memory management.
    """
    try:
        from langchain_huggingface import HuggingFacePipeline
        from transformers import pipeline
        
        # Get model from model manager (lazy loaded)
        model_data = model_manager.get_model('llm')
        
        if model_data is None:
            logger.warning("LLM model not available")
            return None
        
        # Create pipeline with loaded model
        pipe = pipeline(
            "text-generation",
            model=model_data['model'],
            tokenizer=model_data['tokenizer'],
            max_new_tokens=150,  # Generate max 150 new tokens
            do_sample=True,
            temperature=0.7,
            pad_token_id=model_data['tokenizer'].eos_token_id,
            return_full_text=False,  # Only return generated text
        )
        
        return HuggingFacePipeline(pipeline=pipe, model_kwargs={'trust_remote_code': True})
        
    except ImportError as e:
        logger.error(f'Required libraries not available: {e}')
        return None
    except Exception as e:
        logger.error(f'Error creating LLM: {e}')
        return None

def get_model_components():
    """
    Get model and tokenizer components directly (for legacy compatibility).
    """
    global llama_model, llama_tokenizer
    
    try:
        model_data = model_manager.get_model('llm')
        if model_data:
            llama_model = model_data['model']
            llama_tokenizer = model_data['tokenizer']
            return llama_model, llama_tokenizer
    except Exception as e:
        logger.error(f'Error getting model components: {e}')
    
    return None, None

# Initialize on first access
def _ensure_models_loaded():
    """Ensure models are loaded for legacy compatibility."""
    global llama_model, llama_tokenizer
    if llama_model is None or llama_tokenizer is None:
        llama_model, llama_tokenizer = get_model_components()
