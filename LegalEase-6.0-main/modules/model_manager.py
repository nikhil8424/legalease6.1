# modules/model_manager.py
"""
Performance-optimized model manager with lazy loading and memory management.
This module ensures models are only loaded when needed and manages memory efficiently.
"""

import logging
import threading
import time
import gc
from typing import Optional, Dict, Any
from functools import lru_cache
import torch

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Singleton model manager that implements lazy loading and memory optimization.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self._models = {}
        self._model_configs = {}
        self._last_used = {}
        self._loading_locks = {}
        self._max_idle_time = 300  # 5 minutes
        self._cleanup_thread = None
        self._initialized = True
        
        # Start cleanup thread
        self._start_cleanup_thread()
        logger.info("ModelManager initialized")
    
    def _start_cleanup_thread(self):
        """Start background thread to clean up unused models."""
        if self._cleanup_thread is None or not self._cleanup_thread.is_alive():
            self._cleanup_thread = threading.Thread(target=self._cleanup_models, daemon=True)
            self._cleanup_thread.start()
    
    def _cleanup_models(self):
        """Background cleanup of unused models to free memory."""
        while True:
            try:
                current_time = time.time()
                models_to_remove = []
                
                for model_name, last_used in self._last_used.items():
                    if current_time - last_used > self._max_idle_time:
                        models_to_remove.append(model_name)
                
                for model_name in models_to_remove:
                    self._unload_model(model_name)
                    logger.info(f"Unloaded unused model: {model_name}")
                
                # Run garbage collection
                if models_to_remove:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in model cleanup: {e}")
                time.sleep(60)
    
    def _unload_model(self, model_name: str):
        """Unload a specific model from memory."""
        if model_name in self._models:
            del self._models[model_name]
            del self._last_used[model_name]
            if model_name in self._loading_locks:
                del self._loading_locks[model_name]
    
    def register_model(self, name: str, loader_func: callable, config: Dict[str, Any] = None):
        """Register a model with its loader function."""
        self._model_configs[name] = {
            'loader': loader_func,
            'config': config or {}
        }
        self._loading_locks[name] = threading.Lock()
        logger.info(f"Registered model: {name}")
    
    def get_model(self, name: str):
        """Get model with lazy loading."""
        if name not in self._model_configs:
            raise ValueError(f"Model '{name}' not registered")
        
        # Update last used time
        self._last_used[name] = time.time()
        
        # Return if already loaded
        if name in self._models:
            return self._models[name]
        
        # Load model with thread safety
        with self._loading_locks[name]:
            # Double-check after acquiring lock
            if name in self._models:
                return self._models[name]
            
            logger.info(f"Loading model: {name}")
            start_time = time.time()
            
            try:
                loader_func = self._model_configs[name]['loader']
                config = self._model_configs[name]['config']
                
                model = loader_func(**config)
                self._models[name] = model
                
                load_time = time.time() - start_time
                logger.info(f"Model '{name}' loaded successfully in {load_time:.2f}s")
                
                return model
                
            except Exception as e:
                logger.error(f"Failed to load model '{name}': {e}")
                raise
    
    def is_model_loaded(self, name: str) -> bool:
        """Check if a model is currently loaded."""
        return name in self._models
    
    def force_unload(self, name: str):
        """Force unload a specific model."""
        if name in self._models:
            self._unload_model(name)
            logger.info(f"Force unloaded model: {name}")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            'loaded_models': list(self._models.keys()),
            'memory_usage_mb': memory_info.rss / 1024 / 1024,
            'gpu_memory_mb': torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        }

# Global instance
model_manager = ModelManager()

# Model loader functions
def load_llm_model(**config):
    """Load LLM model with error handling."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        
        model_name = config.get('model_name', 'facebook/opt-350m')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            device_map='auto' if torch.cuda.is_available() else None
        )
        
        return {
            'model': model,
            'tokenizer': tokenizer,
            'device': device
        }
    except Exception as e:
        logger.error(f"Failed to load LLM model: {e}")
        return None

def load_summarization_model(**config):
    """Load summarization model."""
    try:
        from transformers import pipeline
        
        model_name = config.get('model_name', 'facebook/bart-large-cnn')
        return pipeline('summarization', model=model_name)
    except Exception as e:
        logger.error(f"Failed to load summarization model: {e}")
        return None

def load_spacy_model(**config):
    """Load spaCy model."""
    try:
        import spacy
        
        model_name = config.get('model_name', 'en_core_web_sm')
        return spacy.load(model_name)
    except Exception as e:
        logger.error(f"Failed to load spaCy model: {e}")
        return None

# Register models
model_manager.register_model(
    'llm', 
    load_llm_model, 
    {'model_name': 'facebook/opt-350m'}
)

model_manager.register_model(
    'summarizer', 
    load_summarization_model, 
    {'model_name': 'facebook/bart-large-cnn'}
)

model_manager.register_model(
    'spacy', 
    load_spacy_model, 
    {'model_name': 'en_core_web_sm'}
)
