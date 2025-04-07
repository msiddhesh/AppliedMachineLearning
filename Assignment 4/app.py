from flask import Flask, request, jsonify
import joblib
import os
from score import score  # Ensure score.py is in the same directory
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load the trained model using a relative path
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best_spam_model.joblib')

# Initialize model variable
model = None
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Warning: Could not load model at startup: {str(e)}")
    # We'll try to load it again when handling requests

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Handle both JSON and form-data inputs
            if request.is_json:
                data = request.get_json()
                if "text" not in data or not data["text"].strip():
                    return jsonify({"error": "No input text provided"}), 400
                text = data["text"].strip()
            else:
                text = request.form.get("text", "").strip()
                if not text:
                    return jsonify({"error": "No input text provided"}), 400

            # Get prediction using the score function
            # The score function will load the model if needed
            prediction, probability = score(text, model, 0.55)

            return jsonify({
                "prediction": "SPAM" if prediction else "HAM",
                "propensity": float(probability)
            })
        
        except Exception as e:
            return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

    return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Spam Classifier</title>
        </head>
        <body>
            <h1>Spam Classifier</h1>
            <form action="/" method="post">
                <label for="text">Enter Text:</label><br>
                <input type="text" id="text" name="text" required><br><br>
                <input type="submit" value="Submit">
            </form>
        </body>
        </html>
    """

if __name__ == '__main__':
    # In Docker we need to listen on 0.0.0.0
    app.run(host='0.0.0.0', port=5000, debug=True)