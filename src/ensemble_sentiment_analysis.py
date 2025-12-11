# ensemble_sentiment_analysis.py
#
# Classifies financial text as positive ("UP"), negative ("DOWN"), or neutral ("NEUTRAL").
#
# Implements a voting ensemble including a TF-IDF + Logistic Regression Classifier, VADER, and FinBERT.
#
# In the case that all models yield a different result, VADER breaks the tie
# because it has demonstrated higher accuracy so far.

import torch
import numpy as np
import pickle
import os
import logging
from typing import List, Tuple, Optional
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL = os.path.join(BASE_DIR, "tfidf_lr_model.pkl")

labels = ["UP", "DOWN", "NEUTRAL"]

# Global model cache for FinBERT (singleton pattern)
_finbert_model = None
_finbert_tokenizer = None
_vader_analyzer = None
_tfidf_model_cache = None

def get_finbert_model():
    """Get or load FinBERT model and tokenizer (cached singleton)."""
    global _finbert_model, _finbert_tokenizer
    if _finbert_model is None or _finbert_tokenizer is None:
        logger.info("Loading FinBERT model (first time, will be cached)...")
        try:
            _finbert_model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
            _finbert_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
            logger.info("FinBERT model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading FinBERT model: {e}")
            raise
    return _finbert_model, _finbert_tokenizer

def get_vader_analyzer():
    """Get or create VADER analyzer (cached singleton)."""
    global _vader_analyzer
    if _vader_analyzer is None:
        _vader_analyzer = SentimentIntensityAnalyzer()
    return _vader_analyzer

def get_tfidf_model():
    """Get or load TF-IDF model (cached singleton)."""
    global _tfidf_model_cache
    if _tfidf_model_cache is None:
        logger.info("Loading TF-IDF model (first time, will be cached)...")
        try:
            with open(BASE_MODEL, "rb") as f:
                _tfidf_model_cache = pickle.load(f)
            logger.info("TF-IDF model loaded successfully")
        except FileNotFoundError:
            logger.warning(f"TF-IDF model file not found: {BASE_MODEL}")
            return None
        except Exception as e:
            logger.error(f"Error loading TF-IDF model: {e}")
            return None
    return _tfidf_model_cache

"""
VADER Sentiment
"""
def analyze_sentiment_vader(text: str) -> str:
    """Analyze sentiment using VADER."""
    if not text or not text.strip():
        return "NEUTRAL"
    
    try:
        analyzer = get_vader_analyzer()
        scores = analyzer.polarity_scores(text)
        polarity = scores["compound"]

        if polarity > 0.05:
            return "UP"
        elif polarity < -0.05:
            return "DOWN"
        else:
            return "NEUTRAL"
    except Exception as e:
        logger.error(f"Error in VADER analysis: {e}")
        return "NEUTRAL"

"""
FinBERT Sentiment
"""
def analyze_sentiment_finbert(text: str) -> str:
    """Analyze sentiment using FinBERT (with caching)."""
    if not text or not text.strip():
        return "NEUTRAL"

    try:
        model, tokenizer = get_finbert_model()
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).numpy()[0]
        max_index = np.argmax(probabilities)
        sentiment = labels[max_index]

        return sentiment
    except Exception as e:
        logger.error(f"Error in FinBERT analysis: {e}")
        return "NEUTRAL"

"""
Base Model Sentiment (TF-IDF + Logistic Regression)
"""
def analyze_sentiment_base(text: str) -> Optional[str]:
    """Analyze sentiment using TF-IDF + Logistic Regression."""
    if not text or not text.strip():
        return "NEUTRAL"
    
    try:
        saved = get_tfidf_model()
        if saved is None:
            logger.warning("TF-IDF model not available, skipping")
            return None
        
        vectorizer = saved["vectorizer"]
        model = saved["model"]

        X_tfidf = vectorizer.transform([text])
        pred = model.predict(X_tfidf)[0]

        return pred
    except Exception as e:
        logger.error(f"Error in TF-IDF analysis: {e}")
        return None

"""
Voting Ensemble
"""
def analyze_sentiment(text: str) -> str:
    """
    Analyze sentiment using ensemble of VADER and FinBERT.
    VADER breaks ties if models disagree.
    """
    if not text or not text.strip():
        return "NEUTRAL"
    
    try:
        # Get votes from each model
        vader_vote = analyze_sentiment_vader(text)
        finbert_vote = analyze_sentiment_finbert(text)
        
        # Optional: Try to get TF-IDF vote (may be None if model unavailable)
        base_vote = analyze_sentiment_base(text)
        
        # Build votes list (exclude None values)
        votes = [v for v in [base_vote, vader_vote, finbert_vote] if v is not None]
        
        if not votes:
            logger.warning("No valid votes from any model, returning NEUTRAL")
            return "NEUTRAL"
        
        # Count votes per label
        vote_counts = {label: votes.count(label) for label in labels}
        
        # Find max votes
        max_votes = max(vote_counts.values())
        candidates = [label for label, count in vote_counts.items() if count == max_votes]

        # VADER breaks tie if multiple candidates
        if len(candidates) > 1:
            final_label = vader_vote
        else:
            final_label = candidates[0]

        return final_label
    except Exception as e:
        logger.error(f"Error in ensemble sentiment analysis: {e}")
        return "NEUTRAL"

"""
Batch Processing
"""
def analyze_sentiment_batch(texts: List[str], batch_size: int = 8) -> List[str]:
    """
    Analyze sentiment for multiple texts efficiently using batch processing.
    
    Args:
        texts: List of text strings to analyze
        batch_size: Number of texts to process in each batch (for FinBERT)
    
    Returns:
        List of sentiment labels
    """
    if not texts:
        return []
    
    results = []
    model, tokenizer = get_finbert_model()
    
    # Process in batches for FinBERT (most expensive operation)
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_results = []
        
        try:
            # Filter empty texts
            valid_batch = []
            valid_indices = []
            for j, text in enumerate(batch):
                if text and text.strip():
                    valid_batch.append(text)
                    valid_indices.append(j)
                else:
                    batch_results.append(("NEUTRAL", j))
            
            if not valid_batch:
                # All empty, return NEUTRAL for all
                results.extend(["NEUTRAL"] * len(batch))
                continue
            
            # Batch FinBERT processing
            inputs = tokenizer(
                valid_batch, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=True
            )
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1).numpy()
            
            # Process each text in batch
            finbert_votes = []
            for prob in probabilities:
                max_index = np.argmax(prob)
                finbert_votes.append(labels[max_index])
            
            # Get VADER and TF-IDF votes for each text
            for idx, text_item in enumerate(valid_batch):
                vader_vote = analyze_sentiment_vader(text_item)
                finbert_vote = finbert_votes[idx]
                base_vote = analyze_sentiment_base(text_item)
                
                # Ensemble voting
                votes = [v for v in [base_vote, vader_vote, finbert_vote] if v is not None]
                vote_counts = {label: votes.count(label) for label in labels}
                max_votes = max(vote_counts.values())
                candidates = [label for label, count in vote_counts.items() if count == max_votes]
                
                if len(candidates) > 1:
                    final_label = vader_vote
                else:
                    final_label = candidates[0]
                
                batch_results.append((final_label, valid_indices[idx]))
            
            # Fill in results for this batch
            batch_final = ["NEUTRAL"] * len(batch)
            for label, idx in batch_results:
                batch_final[idx] = label
            
            # Fill empty texts with NEUTRAL
            for j, text in enumerate(batch):
                if not text or not text.strip():
                    batch_final[j] = "NEUTRAL"
            
            results.extend(batch_final)
                
        except Exception as e:
            logger.error(f"Error processing batch starting at index {i}: {e}")
            # Fallback to individual processing for this batch
            for text_item in batch:
                try:
                    results.append(analyze_sentiment(text_item))
                except:
                    results.append("NEUTRAL")
    
    return results
