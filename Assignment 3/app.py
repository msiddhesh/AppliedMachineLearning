from flask import Flask, request, jsonify
import joblib
from score import score

app = Flask(__name__)

# Load the trained model using joblib
MODEL_PATH = r'E:\Sem 4\AML ASSIGNEMNT\AppliedMachineLearning\Assignment 3\best_spam_model.joblib'

try:
    model = joblib.load(MODEL_PATH)
    print("Loaded trained model from 'best_spam_model.joblib'")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    print("Will use the model loading logic in score.py instead")

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text = request.form.get('text', '')
        
        if not text:
            return jsonify({"error": "No input text provided"}), 400
        
        prediction, probability = score(text, model, 0.55)

        response = {
            "prediction": "HAM" if bool(prediction)== False else "SPAM",  # Ensure JSON serializability
            "propensity": float(probability)  # Ensure JSON serializability
        }
        
        return jsonify(response)
    
    return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Spam Classifier</title>
            <style>
                body {
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                    flex-direction: column;
                }
                h1 {
                    text-align: center;
                    margin-top: 20px;
                }
                form {
                    text-align: center;
                }
            </style>
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