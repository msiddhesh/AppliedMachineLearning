import sklearn
import joblib
import numpy as np
from typing import Tuple
import os

def score(text: str, model: sklearn.base.BaseEstimator = None, threshold: float = 0.5) -> Tuple[bool, float]:
    """
    Score a text using a trained model and determine if it's spam based on a threshold.
    
    Args:
        text (str): The text to score
        model (sklearn.base.BaseEstimator, optional): A trained sklearn model. If None, loads the best model
        threshold (float): The threshold for classification (0 to 1)
        
    Returns:
        Tuple[bool, float]: A tuple containing:
            - prediction (bool): True if the text is classified as spam, False otherwise
            - propensity (float): The probability score between 0 and 1
    """
    assert type(text) == str
    assert ((type(threshold) == float) or type(threshold) == int) and (0 <= threshold <= 1)
    
    if model is None:
        # Use a relative path that will work in Docker
        model_path = os.path.join(os.path.dirname(__file__), 'best_spam_model.joblib')
        model = joblib.load(model_path)
        
    propensity = model.predict_proba([text])[0][1]  # Get probability of class 1 (spam)

    # Make prediction based on threshold
    prediction = propensity >= threshold
    
    return prediction, propensity