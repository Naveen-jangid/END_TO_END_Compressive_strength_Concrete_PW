import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Load trained model
model = joblib.load("best_rf_model.pkl")

# Get the correct feature names from the model
expected_features = model.feature_names_in_

# Initialize Flask app
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to Concrete Strength Prediction API"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Remove extra spaces from keys
        corrected_data = {k.strip(): v for k, v in data.items()}

        # If the model expects "fine_aggregate " (with space), rename it
        if "fine_aggregate" in corrected_data and "fine_aggregate " in expected_features:
            corrected_data["fine_aggregate "] = corrected_data.pop("fine_aggregate")

        # Convert to DataFrame
        df = pd.DataFrame([corrected_data])

        # **Force columns to match training order**
        df = df[expected_features]  # ✅ Ensure order is correct

        # Make prediction
        prediction = model.predict(df)[0]

        return jsonify({"predicted_compressive_strength": prediction}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)