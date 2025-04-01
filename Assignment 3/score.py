import sklearn
import joblib
import numpy as np
from typing import Tuple

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
    # Ensure text is a string
    if not isinstance(text, str):
        raise TypeError("Text must be a string")
    
    # Ensure threshold is a float between 0 and 1
    if not isinstance(threshold, float) or threshold < 0 or threshold > 1:
        raise ValueError("Threshold must be a float between 0 and 1")
    
    # Load model if not provided
    if model is None:
        try:
            model = joblib.load(r'E:\Sem 4\AML ASSIGNEMNT\AppliedMachineLearning\Assignment 3\best_spam_model.joblib')
        except FileNotFoundError:
            raise ValueError("Model not provided and default model file 'best_model.joblib' not found")
    
    # Get the probability prediction from the model
    try:
        # Assuming the model has predict_proba method
        propensity = model.predict_proba([text])[0][1]  # Get probability of class 1 (spam)
    except:
        # Fallback for models without predict_proba
        try:
            propensity = float(model.decision_function([text])[0])
            # Scale to 0-1 range if necessary using sigmoid function
            propensity = 1 / (1 + np.exp(-propensity))
        except:
            raise ValueError("Model must support either predict_proba or decision_function")
    
    # Make prediction based on threshold
    prediction = propensity >= threshold
    
    return prediction, propensity