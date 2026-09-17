"""Data preprocessing utilities"""

import re
from typing import List


def preprocess_text(text: str) -> str:
    """
    Preprocess text for medical chatbot
    
    Args:
        text: Raw text
        
    Returns:
        Cleaned text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters but keep medical terms
    text = re.sub(r'[^\w\s\-\.]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def tokenize_data(texts: List[str], tokenizer, max_length: int = 512) -> dict:
    """
    Tokenize texts for model input
    
    Args:
        texts: List of text strings
        tokenizer: Tokenizer from transformers
        max_length: Maximum sequence length
        
    Returns:
        Tokenized data
    """
    encodings = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    return encodings
