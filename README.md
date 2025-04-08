# 📘 Applied Machine Learning

This repository contains coursework for the **Applied Machine Learning** course, covering the complete ML lifecycle — from prototyping to deployment and CI/CD using industry-standard tools like DVC, MLflow, Flask, Docker, and Git hooks.

---

## 📦 Assignment 1: Prototype  
**Goal:** Build a prototype for SMS spam classification.

- Loaded and preprocessed the [UCI SMS Spam Collection dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection).
- Split data into train/validation/test sets and saved as CSV files.
- Trained baseline models and evaluated using accuracy, precision, recall, and AUCPR.
- Validated and selected the best-performing model.

---

## 🔍 Assignment 2: Experiment Tracking  
**Goal:** Track data and model experiments.

- Used **DVC** to version control raw and split data files.
- Compared target distributions across data versions using different random seeds.
- Tracked experiments using **MLflow**, including model metrics and parameters.
- Built and registered three benchmark models, compared using AUCPR.

---

## 🧪 Assignment 3: Testing & Model Serving  
**Goal:** Add testing and serve model via API.

- Wrote a `score()` function to predict spam from text input using a trained model.
- Implemented unit tests with **pytest** to validate function output, types, and edge cases.
- Created a **Flask API** with a `/score` endpoint that returns prediction and propensity.
- Wrote integration tests for the API and generated coverage reports.

---

## 🐳 Assignment 4: Containerization & CI  
**Goal:** Dockerize the app and automate testing.

- Created a `Dockerfile` to containerize the Flask app and run it with all dependencies.
- Wrote a test script to build the image, run the container, query the `/score` endpoint, and shut down cleanly.
- Configured a **pre-commit Git hook** to automatically run `test.py` before any commit to the `main` branch.

---

## 🛠️ Tools & Libraries  
- `scikit-learn`, `pandas`, `joblib`, `DVC`, `MLflow`, `Flask`, `pytest`, `Docker`, `Git`

---

