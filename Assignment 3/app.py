from flask import Flask, request, jsonify
import joblib
from score import score  # Ensure score.py is in the same directory or correctly referenced
import warnings
warnings.filterwarnings("ignore")


app = Flask(__name__)

# Load the trained model using joblib
MODEL_PATH = r'E:\Sem 4\AML ASSIGNEMNT\AppliedMachineLearning\Assignment 3\best_spam_model.joblib'
model = joblib.load(MODEL_PATH)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Handle both JSON and form-data inputs
            if request.is_json:
                data = request.get_json()
                if "text" not in data or not data["text"].strip():
                    return jsonify({"error": "No input text provided"}), 400  # 🔥 Fix here
                text = data["text"].strip()
            else:
                text = request.form.get("text", "").strip()
                if not text:
                    return jsonify({"error": "No input text provided"}), 400  # 🔥 Fix here

            # Ensure model is loaded
            if model is None:
                return jsonify({"error": "Model not loaded"}), 500

            # Get prediction
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
    app.run(debug=True)