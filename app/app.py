from flask import Flask, request, jsonify, send_from_directory
import xgboost as xgb
import numpy as np

app = Flask(__name__, static_folder='static')

# Load the XGBoost model
model = xgb.XGBClassifier()
model.load_model('xgboost_00.json')

# Define a dictionary to map predictions to labels
prediction_mapping = {0: "normal", 1: "suspect", 2: "pathological"}

# Route to serve the HTML form
@app.route('/')
def serve_html():
    return send_from_directory(app.static_folder, 'index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array(data['features'])  # Get features from request
    prediction = model.predict(features)  # Model prediction
    
    # Map numeric predictions to their respective string labels
    mapped_prediction = [prediction_mapping[p] for p in prediction]
    
    return jsonify({'prediction': mapped_prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
