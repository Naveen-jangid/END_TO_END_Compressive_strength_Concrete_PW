from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("best_rf_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from form
        features = [float(request.form.get(key, 0)) for key in [
            "cement", "slag", "flyash", "water", "superplasticizer", "coarseagg", "fineagg", "age"
        ]]
        
        print("Received Input Features:", features)  # Debugging
        
        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features_array)[0]
        
        print("Predicted Strength:", prediction)  # Debugging
        
        return render_template('index.html', prediction_text=f'Predicted Strength: {prediction:.2f} MPa')

    except Exception as e:
        print("Error:", e)
        return render_template('index.html', prediction_text="Error in prediction")

if __name__ == '__main__':
    app.run(debug=True)
