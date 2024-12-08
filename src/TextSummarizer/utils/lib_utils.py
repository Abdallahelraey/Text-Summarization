import nltk
from pathlib import Path
from typing import List, Optional, Set
import os
from TextSummarizer.logging import logger

def setup_nltk_environment(data_dir: Optional[str] = None) -> str:
    if data_dir:
        nltk_data_dir = Path(data_dir)
        nltk_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Set NLTK_DATA environment variable
        os.environ['NLTK_DATA'] = str(nltk_data_dir)
        
        # Clear existing paths and add custom path
        nltk.data.path = [str(nltk_data_dir)]
        return str(nltk_data_dir)
    return nltk.data.find('.').parent

def download_nltk_models(data_dir: Optional[str] = None, 
                        models: Optional[List[str]] = None) -> Set[str]:
    default_models = [
        'punkt',           # Sentence tokenizer
        'stopwords',       # Common stopwords
        'averaged_perceptron_tagger',  # POS tagger
        'wordnet',         # Lexical database
        'words'            # Word lists
    ]
    
    # Setup environment if custom directory provided
    if data_dir:
        setup_nltk_environment(data_dir)
    
    models_to_download = models if models else default_models
    downloaded_models = set()
    
    for model in models_to_download:
        try:
            nltk.download(model, 
                         download_dir=data_dir if data_dir else None,
                         quiet=True)
            downloaded_models.add(model)
            logger.info(f"Successfully downloaded '{model}' to {data_dir if data_dir else 'default location'}")
        except Exception as e:
            logger.info(f"Failed to download '{model}': {e}")
    
    return downloaded_models