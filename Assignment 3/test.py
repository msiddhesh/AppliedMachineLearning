import pytest
from score import score
import joblib
import requests
import os
import sys
import signal
import time
import subprocess
import warnings
import numpy
warnings.filterwarnings("ignore")

# Load model using joblib
MODEL_PATH = r"E:\Sem 4\AML ASSIGNEMNT\AppliedMachineLearning\Assignment 3\best_spam_model.joblib"
model = joblib.load(MODEL_PATH)

def test_smoke_test():
    try:
        score("Example", model, 0.5)
    except Exception as e:
        pytest.fail(f"score function raised an exception: {e} (Smoke test failed)")
    
    assert type(score("Example", model, 0.5)) == tuple, f"Expected 2 outputs, received 1 (smoke test failed)"
    assert len(score("Example", model, 0.5)) == 2, f"Expected 2 outputs, received {len(score('Example', model, 0.5))} (smoke test failed)"

def test_format_test():
    text = "Example"
    threshold = 0.7
    prediction, probability = score(text, model, threshold)
    assert type(prediction) == numpy.bool_
    
    try:
        float(probability)
    except Exception as e:
        pytest.fail(f"score function raised an exception: {e} (Format test failed)")

def test_prediction_0_or_1():
    text = "Example"
    threshold = 0.7
    prediction, _ = score(text, model, threshold)
    assert int(prediction) in (0, 1)

def test_propensity_between_0_and_1():
    text = "Example"
    threshold = 0.7
    _, propensity = score(text, model, threshold)
    assert 0 <= propensity <= 1

def test_when_threshold_0_prediction_always_1():
    text_1 = "Be there tonight"
    threshold = 0
    prediction, _ = score(text_1, model, threshold)
    assert int(prediction) == 1
    
    text_2 = "Get a chance to go on a vacation to Hawaii"
    threshold = 0
    prediction, _ = score(text_2, model, threshold)
    assert int(prediction) == 1

def test_when_threshold_1_prediction_always_0():
    text_1 = "Be there tonight"
    threshold = 1
    prediction, _ = score(text_1, model, threshold)
    assert prediction == 0
    
    text_2 = "Get a chance to go on a vacation to Hawaii"
    threshold = 1
    prediction, _ = score(text_2, model, threshold)
    assert prediction == 0

def test_obvious_spam_gives_prediction_1():
    text = '''Just apply to this lucky draw and get a chance to send
              your child to foreign universities like Stanford and Harvard. Don't be late. 
              Offer valid for a limited time only.'''
    prediction, _ = score(text, model, threshold = 0.5)
    assert int(prediction) == 1

def test_obvious_non_spam_gives_prediction_0():
    text = "Don't be late for tomorrow's meeting"
    threshold = 0.4
    prediction, _ = score(text, model, threshold)
    assert int(prediction) == 0

def test_flask():

    process = subprocess.Popen(["python", r"E:\Sem 4\AML ASSIGNEMNT\AppliedMachineLearning\Assignment 3/app.py"], stdout=subprocess.PIPE)

    time.sleep(2)

    payload = {"text": "Hello, congratulations! You have won a prize."}
    response = requests.post("http://127.0.0.1:5000/", data=payload)

    assert response.status_code == 200

    data = response.json()
    assert 'prediction' in data
    assert 'propensity' in data

    process.terminate()

import pytest
from app import app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_homepage(client):
    """Test if the homepage loads correctly."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Spam Classifier" in response.data  

def test_prediction_form_data(client):
    """Test POST request with form-data."""
    response = client.post("/", data={"text": "You have won a free prize! Call now!"})
    assert response.status_code == 200
    assert "prediction" in response.json
    assert "propensity" in response.json

def test_prediction_json(client):
    """Test POST request with JSON data."""
    response = client.post("/", json={"text": "Congratulations! You have won a lottery."})
    assert response.status_code == 200
    assert "prediction" in response.json
    assert "propensity" in response.json

def test_missing_text(client):
    """Test POST request with missing text field."""
    response = client.post("/", json={})  # No 'text' key in JSON
    assert response.status_code == 400
    assert response.json == {"error": "No input text provided"}
